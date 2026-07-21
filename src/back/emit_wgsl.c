// WGSL emission: IR → WGSL text for StereoKit's WebGPU backend. See
// emit_wgsl.h for the output contract. Structure mirrors emit_spirv.c where
// the problems are shared; WGSL needs none of the id/section machinery, and
// entry IO maps to attributed structs instead of SROA'd variables.
//
// Matrices follow the SPIR-V backend's convention (emit_spirv.c:3): HLSL row i
// is WGSL column i (matNxM columns occupy the same bytes as HLSL rows when
// strides match), and mul() swaps its operands to compensate.
//
// A stage using anything browser WebGPU can't express is skipped: `skip()`
// records one located warning, emission stops, and the blob's text stays NULL.

#include "emit_wgsl.h"

#include "../tables/intrinsics.h"
#include "../tables/semantics.h"
#include "../tables/formats.h"
#include "../sema/layout.h"
#include "../sema/const_eval.h"
#include "../ir/ir_operands.h"
#include "../front/ast.h"
#include "../util/array.h"
#include "../../vendor/spirv.h"
#include "../../vendor/GLSL.std.450.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define WGSL_VIEW_INDEX_SPEC_ID 999 // SKSC_WGSL_VIEW_INDEX_SPEC_ID (sksc_file.h)
#define SLOT_TEXTURE   100
#define SLOT_READWRITE 200
#define SLOT_INPUT_ATT 300
#define SLOT_SAMPLER   400

typedef struct wgsl_t {
	svsl_arena_t         *arena;
	const svsl_program_t *prog;
	const svsl_ir_func_t *fn;
	svsl_diag_list_t     *diags;

	svsl_array_t(char) out;
	int32_t            indent;

	bool skipped; // one warning already recorded; all emission becomes a no-op

	// prescan facts
	uint8_t *buffer_used;    // per prog->buffers entry
	uint8_t *res_used;       // per prog->resources entry
	uint8_t *res_cmp;        // texture: 1 = SampleCmp'd, 2 = also plainly sampled
	uint8_t *res_access;     // storage image: bit 0 read, bit 1 written
	uint8_t *struct_used;    // per prog->types.structs entry
	int32_t *sampler_pair;   // standalone sampler → first texture it samples, -1
	const char **const_texts;// per const-global: materialized initializer, NULL = unused/int
	// WGSL atomics are type-level: these mark the storage leaves that must
	// declare atomic<T> (and whose plain loads/stores become atomicLoad/Store)
	uint8_t  *res_atomic;    // object-form structured buffer: its elements
	uint8_t  *wg_atomic;     // workgroup variable (scalar or array-of-scalar)
	uint8_t **buf_atomic;    // per buffer → per member

	// single-use inlining: a value referenced exactly once folds into its use
	// site instead of a `let _N` (see inline_analyze) — inline_expr holds its
	// expression text once emission reaches it
	uint8_t     *inline_ok;
	const char **inline_expr;
	bool     uses_view_index;
	bool     uses_f16;
	bool     uses_derivatives;
	bool     uses_subpass;   // subpass reads fetch at the fragment's own pixel...
	const char *frag_pos;    // ...through this position expression ("in.pos")

	const struct io_list_t *in_io; // entry inputs; read by the param copy-in

	svsl_array_t(svsl_wgsl_sampler_t) samplers;
} wgsl_t;

// ---- text building ---------------------------------------------------------------

static const char *sfmt(wgsl_t *e, const char *fmt, ...) {
	// fast path: most expressions are short, so format once into a stack
	// buffer and copy; only oversized results pay the measure+format double
	char    stack[256];
	va_list args, args2;
	va_start(args, fmt);
	va_copy(args2, args);
	int32_t n = vsnprintf(stack, sizeof(stack), fmt, args);
	va_end(args);
	char *buf = svsl_arena_alloc(e->arena, (size_t)n + 1);
	if (n < (int32_t)sizeof(stack)) memcpy(buf, stack, (size_t)n + 1);
	else                            vsnprintf(buf, (size_t)n + 1, fmt, args2);
	va_end(args2);
	return buf;
}

static void wraw(wgsl_t *e, const char *text) {
	for (const char *c = text; *c; c++)
		svsl_array_push(e->arena, &e->out, *c);
}

// one statement/declaration line at the current indent
static void wln(wgsl_t *e, const char *fmt, ...) {
	if (e->skipped) return;
	for (int32_t i = 0; i < e->indent; i++)
		svsl_array_push(e->arena, &e->out, '\t');
	va_list args, args2;
	va_start(args, fmt);
	va_copy(args2, args);
	int32_t n   = vsnprintf(NULL, 0, fmt, args);
	char   *buf = svsl_arena_alloc(e->arena, (size_t)n + 1);
	vsnprintf(buf, (size_t)n + 1, fmt, args2);
	va_end(args);
	va_end(args2);
	wraw(e, buf);
	svsl_array_push(e->arena, &e->out, '\n');
}

static void skip(wgsl_t *e, svsl_loc_t loc, const char *fmt, ...) {
	if (e->skipped) return;
	e->skipped = true;
	va_list args;
	va_start(args, fmt);
	int32_t n   = vsnprintf(NULL, 0, fmt, args);
	va_end(args);
	char *msg = svsl_arena_alloc(e->arena, (size_t)n + 1);
	va_start(args, fmt);
	vsnprintf(msg, (size_t)n + 1, fmt, args);
	va_end(args);
	svsl_diag_add(e->arena, e->diags, svsl_severity_warning, loc,
	              "WGSL: %s; the stage is skipped and the .sks carries no WGSL for it", msg);
}

// ---- identifiers -----------------------------------------------------------------

// WGSL keywords and predeclared names we must not collide with. Modest set:
// anything colliding gets a trailing underscore.
static bool wgsl_reserved(const char *s) {
	static const char *words[] = {
		"alias", "break", "case", "const", "continue", "continuing", "default",
		"diagnostic", "discard", "else", "enable", "false", "fn", "for", "if",
		"let", "loop", "override", "requires", "return", "struct", "switch",
		"true", "var", "while", "bool", "f16", "f32", "i32", "u32", "sampler",
		"array", "atomic", "ptr", "bitcast", "dot", "cross", "length", "min",
		"max", "abs", "all", "any", "select", "normalize", "distance", "mix",
		"in", "out", "ref", "mod", "smoothstep",
	};
	for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++)
		if (strcmp(s, words[i]) == 0) return true;
	if (strncmp(s, "vec", 3) == 0 || strncmp(s, "mat", 3) == 0 ||
	    strncmp(s, "texture", 7) == 0)
		return true;
	return false;
}

static const char *ident(wgsl_t *e, svsl_str_t name) {
	int32_t len = name.len < 0 ? 0 : name.len;
	char   *buf = svsl_arena_alloc(e->arena, (size_t)len + 2);
	for (int32_t i = 0; i < len; i++) {
		char c = name.ptr[i];
		bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		          (c >= '0' && c <= '9') || c == '_';
		buf[i] = ok ? c : '_';
	}
	buf[len] = '\0';
	if (len == 0 || (buf[0] >= '0' && buf[0] <= '9')) return sfmt(e, "x%s", buf);
	if (wgsl_reserved(buf)) { buf[len] = '_'; buf[len + 1] = '\0'; }
	return buf;
}

// ---- types -----------------------------------------------------------------------

static const char *scalar_name(wgsl_t *e, svsl_scalar_ s, svsl_loc_t loc) {
	switch (s) {
	case svsl_scalar_bool:    return "bool";
	case svsl_scalar_int32:   return "i32";
	case svsl_scalar_uint32:  return "u32";
	case svsl_scalar_float32: return "f32";
	case svsl_scalar_half:    return "f32"; // relaxed half is float32 in buffers and math
	case svsl_scalar_float16: e->uses_f16 = true; return "f16";
	default:
		skip(e, loc, "the type uses a scalar width WGSL has no equivalent for");
		return "f32";
	}
}

static const char *struct_name_w(wgsl_t *e, int32_t struct_index) {
	return ident(e, e->prog->types.structs.items[struct_index].name);
}

static const char *type_name_w(wgsl_t *e, svsl_type_id_t id, svsl_loc_t loc);
static bool        ptr_is_atomic(wgsl_t *e, uint32_t id);

// one folded scalar as a WGSL literal; false when the value can't be spelled
static bool wgsl_scalar_literal(wgsl_t *e, svsl_scalar_ scalar, double v, const char **out) {
	switch (scalar) {
	case svsl_scalar_bool:   *out = v != 0 ? "true" : "false"; return true;
	case svsl_scalar_int32:  *out = sfmt(e, "%lldi", (long long)(int64_t)v); return true;
	case svsl_scalar_uint32: *out = sfmt(e, "%lluu", (unsigned long long)(uint64_t)(int64_t)v); return true;
	case svsl_scalar_half:
	case svsl_scalar_float32: {
		float f = (float)v;
		if (f != f || f > 3.4e38f || f < -3.4e38f) return false; // WGSL can't spell NaN/Inf
		return *out = sfmt(e, "%.9gf", f), true;
	}
	case svsl_scalar_float16:
		e->uses_f16 = true;
		return *out = sfmt(e, "%.9gh", (float)v), true;
	default: return false;
	}
}

// materializes a const-global initializer as a WGSL const-expression; NULL
// when the fold fails (the caller skips the stage, naming the global)
static const char *const_global_text(wgsl_t *e, const struct svsl_ast_expr_t *expr,
                                     svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	const char        *lit;
	if (t->kind == svsl_type_scalar) {
		double v;
		if (!svsl_const_eval_num(e->prog, expr, &v)) return NULL;
		return wgsl_scalar_literal(e, t->scalar, v, &lit) ? lit : NULL;
	}
	if (t->kind == svsl_type_vector) {
		double vals[4];
		if (!svsl_const_eval_vec(e->prog, expr, t->count, vals)) return NULL;
		const char *args = "";
		for (int32_t i = 0; i < t->count; i++) {
			if (!wgsl_scalar_literal(e, t->scalar, vals[i], &lit)) return NULL;
			args = sfmt(e, "%s%s%s", args, i ? ", " : "", lit);
		}
		return sfmt(e, "%s(%s)", type_name_w(e, type, (svsl_loc_t){0}), args);
	}
	// arrays (and matrices as rows-of-vectors) from init lists / ctor forms
	const svsl_ast_expr_t **items = NULL;
	int32_t                 count = 0;
	if (expr->kind == svsl_expr_init_list) { items = (const svsl_ast_expr_t **)expr->init_list.items; count = expr->init_list.count; }
	else if (expr->kind == svsl_expr_ctor) { items = (const svsl_ast_expr_t **)expr->ctor.args; count = expr->ctor.arg_count; }
	else return NULL;

	if (t->kind == svsl_type_array && count == t->array_count) {
		const char *args = "";
		for (int32_t i = 0; i < count; i++) {
			const char *item = const_global_text(e, items[i], t->elem);
			if (!item) return NULL;
			args = sfmt(e, "%s%s%s", args, i ? ", " : "", item);
		}
		return sfmt(e, "%s(%s)", type_name_w(e, type, (svsl_loc_t){0}), args);
	}
	if (t->kind == svsl_type_matrix && count == t->rows * t->cols) {
		const char *args = "";
		for (int32_t r = 0; r < t->rows; r++) { // HLSL rows are the WGSL columns
			const char *col = "";
			for (int32_t c = 0; c < t->cols; c++) {
				double v;
				if (!svsl_const_eval_num(e->prog, items[r * t->cols + c], &v) ||
				    !wgsl_scalar_literal(e, t->scalar, v, &lit)) return NULL;
				col = sfmt(e, "%s%s%s", col, c ? ", " : "", lit);
			}
			args = sfmt(e, "%s%svec%d<%s>(%s)", args, r ? ", " : "", t->cols,
			            scalar_name(e, t->scalar, (svsl_loc_t){0}), col);
		}
		return sfmt(e, "%s(%s)", type_name_w(e, type, (svsl_loc_t){0}), args);
	}
	return NULL;
}

// packed structs expose logical bit fields; their physical members are all
// backing uint32 words that share a source name — number them instead
static const char *struct_member_name(wgsl_t *e, int32_t struct_index, int32_t m) {
	const svsl_struct_info_t *info = &e->prog->types.structs.items[struct_index];
	if (info->packed) return sfmt(e, "_w%d", m);
	return ident(e, info->members.items[m].name);
}

static const char *type_name_w(wgsl_t *e, svsl_type_id_t id, svsl_loc_t loc) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, id);
	switch (t->kind) {
	case svsl_type_scalar: return scalar_name(e, t->scalar, loc);
	case svsl_type_vector: return sfmt(e, "vec%d<%s>", t->count, scalar_name(e, t->scalar, loc));
	case svsl_type_matrix: // HLSL rows are the WGSL columns
		return sfmt(e, "mat%dx%d<%s>", t->rows, t->cols, scalar_name(e, t->scalar, loc));
	case svsl_type_array:
		if (t->array_count == 0) return sfmt(e, "array<%s>", type_name_w(e, t->elem, loc));
		return sfmt(e, "array<%s, %d>", type_name_w(e, t->elem, loc), t->array_count);
	case svsl_type_struct: return struct_name_w(e, t->struct_index);
	default:
		skip(e, loc, "an opaque resource type appeared as a value");
		return "f32";
	}
}

// ---- WGSL memory layout (gauge only: we never restructure, we verify) ------------

// Natural WGSL alignment/size for host-shareable types; `uniform` applies the
// address-space rounding rules. Returns false when the type can't live in a
// host-shareable buffer at all.
static bool wgsl_layout(wgsl_t *e, svsl_type_id_t id, bool uniform,
                        uint32_t *out_align, uint32_t *out_size);

static uint32_t round_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

static bool wgsl_member_offsets(wgsl_t *e, const svsl_struct_info_t *info, bool uniform,
                                uint32_t *out_offsets, uint32_t *out_align, uint32_t *out_size) {
	uint32_t offset = 0, align = 1;
	for (int32_t m = 0; m < info->members.count; m++) {
		uint32_t ma, ms;
		if (!wgsl_layout(e, info->members.items[m].type, uniform, &ma, &ms)) return false;
		if (uniform) {
			const svsl_type_t *mt = svsl_type_get(&e->prog->types, info->members.items[m].type);
			if (mt->kind == svsl_type_struct || mt->kind == svsl_type_array)
				ma = round_up(ma, 16);
		}
		offset = round_up(offset, ma);
		if (out_offsets) out_offsets[m] = offset;
		offset += ms;
		if (ma > align) align = ma;
	}
	*out_align = align;
	*out_size  = round_up(offset, align);
	return true;
}

static bool wgsl_layout(wgsl_t *e, svsl_type_id_t id, bool uniform,
                        uint32_t *out_align, uint32_t *out_size) {
	const svsl_type_t *t  = svsl_type_get(&e->prog->types, id);
	uint32_t           sc = t->scalar == svsl_scalar_float16 ? 2 : 4;
	if ((t->kind == svsl_type_scalar || t->kind == svsl_type_vector) &&
	    t->scalar == svsl_scalar_bool)
		return false; // bool is not host-shareable in WGSL; declare it uint
	switch (t->kind) {
	case svsl_type_scalar: *out_align = sc; *out_size = sc; return true;
	case svsl_type_vector:
		*out_align = t->count == 2 ? sc * 2 : sc * 4;
		*out_size  = sc * (uint32_t)t->count;
		return true;
	case svsl_type_matrix: { // rows HLSL-rows = columns; column vec size = cols
		uint32_t ca = t->cols == 2 ? sc * 2 : sc * 4;
		*out_align = ca;
		*out_size  = ca * (uint32_t)t->rows;
		return true;
	}
	case svsl_type_array: {
		uint32_t ea, es;
		if (!wgsl_layout(e, t->elem, uniform, &ea, &es)) return false;
		// the element stride is natural — WGSL does NOT round it up; a uniform
		// array whose natural stride isn't a 16-multiple is simply invalid
		uint32_t stride = round_up(es, ea);
		if (uniform && stride % 16 != 0) return false;
		*out_align = ea;
		*out_size  = stride * (uint32_t)(t->array_count > 0 ? t->array_count : 1);
		return true;
	}
	case svsl_type_struct:
		return wgsl_member_offsets(e, &e->prog->types.structs.items[t->struct_index],
		                           uniform, NULL, out_align, out_size);
	default: return false;
	}
}

// Verifies that WGSL's automatic layout reproduces the svsl layout for one
// buffer; skips the stage (with the offending member named) when it can't.
static void check_buffer_layout(wgsl_t *e, const svsl_buffer_t *buf) {
	bool uniform = buf->kind == svsl_block_uniform;
	if (buf->layout == svsl_layout_pack1 || buf->layout == svsl_layout_pack8) {
		skip(e, (svsl_loc_t){0}, "buffer '%.*s' uses a %s layout, which WGSL cannot express "
		     "(no scalar block layout)", buf->name.len, buf->name.ptr,
		     buf->layout == svsl_layout_pack1 ? "pack1/scalar" : "pack8/relaxed");
		return;
	}
	for (int32_t m = 0; m < buf->members.count && !e->skipped; m++) {
		const svsl_buf_member_t *member = &buf->members.items[m];
		// recompute the WGSL offset of every member from scratch
		uint32_t offset = 0;
		for (int32_t k = 0; k <= m; k++) {
			uint32_t ma, ms;
			if (!wgsl_layout(e, buf->members.items[k].type, uniform, &ma, &ms)) {
				skip(e, buf->members.items[k].loc, "member '%.*s' of buffer '%.*s' has no valid "
				     "WGSL %s layout", buf->members.items[k].name.len, buf->members.items[k].name.ptr,
				     buf->name.len, buf->name.ptr, uniform ? "uniform" : "storage");
				return;
			}
			if (uniform) {
				const svsl_type_t *mt = svsl_type_get(&e->prog->types, buf->members.items[k].type);
				if (mt->kind == svsl_type_struct || mt->kind == svsl_type_array)
					ma = round_up(ma, 16);
			}
			offset = round_up(offset, ma);
			if (k < m) offset += ms;
		}
		if (offset != member->offset) {
			skip(e, member->loc, "member '%.*s' of buffer '%.*s' sits at offset %u, but WGSL's "
			     "%s layout puts it at %u — restructure the buffer (float4-sized members always agree)",
			     member->name.len, member->name.ptr, buf->name.len, buf->name.ptr,
			     member->offset, uniform ? "uniform" : "storage", offset);
			return;
		}
	}
}

// ---- constants -------------------------------------------------------------------

static float bits_to_f32(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }

static const char *const_str(wgsl_t *e, const svsl_ir_inst_t *inst) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, inst->type);
	uint32_t bits = inst->args[0];
	switch (t->scalar) {
	case svsl_scalar_bool:   return bits ? "true" : "false";
	case svsl_scalar_int32:  // INT32_MIN has no literal spelling (the minus binds separately)
		return bits == 0x80000000u ? "i32(-2147483648)" : sfmt(e, "%di", (int32_t)bits);
	case svsl_scalar_uint32: return sfmt(e, "%uu", bits);
	case svsl_scalar_half:
	case svsl_scalar_float32: {
		if ((bits & 0x7F800000u) == 0x7F800000u) {
			skip(e, inst->loc, "a NaN/Inf float constant appears here, which WGSL cannot spell");
			return "0f";
		}
		return sfmt(e, "%.9gf", bits_to_f32(bits));
	}
	case svsl_scalar_float16: {
		e->uses_f16 = true;
		// f16 constants carry their f32 value in args[0] like the SPIR-V path
		return sfmt(e, "%.9gh", bits_to_f32(bits));
	}
	default:
		skip(e, inst->loc, "a constant of a scalar width WGSL has no equivalent for");
		return "0";
	}
}

// ---- resources -------------------------------------------------------------------

static const char *res_name(wgsl_t *e, int32_t res) {
	return ident(e, e->prog->resources.items[res].name);
}

// the sampler resource paired with a texture (same s-slot + space), or -1 —
// the same pairing rule sks_write uses to fuse samplers away on Vulkan
static int32_t tex_paired_sampler(wgsl_t *e, int32_t tex_res) {
	const svsl_resource_t *tex = &e->prog->resources.items[tex_res];
	if (tex->sampler_slot < 0) return -1;
	for (int32_t k = 0; k < e->prog->resources.count; k++) {
		const svsl_resource_t *smp = &e->prog->resources.items[k];
		if (smp->kind == svsl_res_sampler && smp->bind.slot == tex->sampler_slot &&
		    smp->bind.space == tex->bind.space) return k;
	}
	return -1;
}

// canonical name of a texture's paired sampler: the paired sampler resource's
// own name when one exists, else "<texture>_sampler" (synthesized)
static const char *paired_sampler_name(wgsl_t *e, int32_t tex_res) {
	int32_t smp = tex_paired_sampler(e, tex_res);
	return smp >= 0 ? res_name(e, smp) : sfmt(e, "%s_sampler", res_name(e, tex_res));
}

// Whether a texture binds as texture_depth_* — decided by its paired sampler's
// DECLARED type, because the runtime derives the bind group layout from that
// declaration (meta shape bit 5), never from per-stage usage. Usage that
// contradicts the declaration skips the stage in the prescan.
static bool tex_is_depth(wgsl_t *e, int32_t tex_res) {
	int32_t smp = tex_paired_sampler(e, tex_res);
	return smp >= 0 &&
	       svsl_type_get(&e->prog->types, e->prog->resources.items[smp].type)->is_comparison;
}

// resolves the sampler operand of a tex op: an explicit sampler resource, or
// the texture's paired sampler
static const char *sampler_ref(wgsl_t *e, int32_t tex_res, uint32_t sampler_res) {
	if (sampler_res != SVSL_IR_NONE) return res_name(e, (int32_t)sampler_res);
	return paired_sampler_name(e, tex_res);
}

// true when this sampler resource is paired with some texture (it then binds
// through the texture's declaration, never standalone)
static bool sampler_is_paired(wgsl_t *e, int32_t sampler_res) {
	const svsl_resource_t *smp = &e->prog->resources.items[sampler_res];
	for (int32_t k = 0; k < e->prog->resources.count; k++) {
		const svsl_resource_t *tex = &e->prog->resources.items[k];
		if (tex->kind == svsl_res_texture && tex->sampler_slot == smp->bind.slot &&
		    tex->bind.space == smp->bind.space) return true;
	}
	return false;
}

static const char *wgsl_texel_format(uint32_t spv_format) {
	switch (spv_format) {
	case SpvImageFormatRgba8:       return "rgba8unorm";
	case SpvImageFormatRgba8Snorm:  return "rgba8snorm";
	case SpvImageFormatRgba8ui:     return "rgba8uint";
	case SpvImageFormatRgba8i:      return "rgba8sint";
	case SpvImageFormatRgba16f:     return "rgba16float";
	case SpvImageFormatRgba16ui:    return "rgba16uint";
	case SpvImageFormatRgba16i:     return "rgba16sint";
	case SpvImageFormatR32f:        return "r32float";
	case SpvImageFormatR32ui:       return "r32uint";
	case SpvImageFormatR32i:        return "r32sint";
	case SpvImageFormatRg32f:       return "rg32float";
	case SpvImageFormatRg32ui:      return "rg32uint";
	case SpvImageFormatRg32i:       return "rg32sint";
	case SpvImageFormatRgba32f:     return "rgba32float";
	case SpvImageFormatRgba32ui:    return "rgba32uint";
	case SpvImageFormatRgba32i:     return "rgba32sint";
	default:                        return NULL;
	}
}

// ---- value / lvalue expressions --------------------------------------------------

static const char *val(wgsl_t *e, uint32_t id);

// literal value of a constant instruction (chain indices into structs)
static bool const_literal(wgsl_t *e, uint32_t id, int64_t *out) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	if (in->op == svsl_ir_const) {
		const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
		*out = t->scalar == svsl_scalar_int32 ? (int64_t)(int32_t)in->args[0] : (int64_t)in->args[0];
		return true;
	}
	return false;
}

// pointer-producing ops resolve to a path string rooted at a var-backed name
static const char *lvalue(wgsl_t *e, uint32_t id, svsl_type_id_t *out_type) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	switch (in->op) {
	case svsl_ir_var:
	case svsl_ir_param:
		*out_type = in->type;
		return sfmt(e, "_%u", id);
	case svsl_ir_ptr: {
		svsl_ref_ kind = (svsl_ref_)in->args[0];
		int32_t   a = (int32_t)in->args[1], b = (int32_t)in->args[2];
		switch (kind) {
		case svsl_ref_buffer_member: {
			const svsl_buffer_t *buf = &e->prog->buffers.items[a];
			*out_type = buf->members.items[b].type;
			const char *base = ident(e, buf->name);
			if (buf->kind == svsl_block_storagebuffer) // declared via its resource entry
				for (int32_t r = 0; r < e->prog->resources.count; r++)
					if (e->prog->resources.items[r].buffer_index == a) base = res_name(e, r);
			return sfmt(e, "%s.%s", base, ident(e, buf->members.items[b].name));
		}
		case svsl_ref_resource: // object-form structured buffer
			*out_type = e->prog->resources.items[a].type;
			return res_name(e, a);
		case svsl_ref_workgroup:
			*out_type = e->prog->workgroup_vars.items[a].type;
			return ident(e, e->prog->workgroup_vars.items[a].name);
		case svsl_ref_const_global:
			*out_type = e->prog->const_globals.items[a].type;
			return ident(e, e->prog->const_globals.items[a].name);
		case svsl_ref_spec_const:
			*out_type = e->prog->spec_consts.items[a].type;
			return ident(e, e->prog->spec_consts.items[a].name);
		default:
			skip(e, in->loc, "a storage reference kind the WGSL backend doesn't handle yet (%d)", kind);
			*out_type = SVSL_TYPE_NONE;
			return "_bad";
		}
	}
	case svsl_ir_chain: {
		svsl_type_id_t base_type;
		const char    *path = lvalue(e, in->args[0], &base_type);
		for (uint32_t i = 0; i < in->aux_count && !e->skipped; i++) {
			uint32_t           idx = e->fn->aux.items[in->aux + i];
			const svsl_type_t *t   = svsl_type_get(&e->prog->types, base_type);
			int64_t            lit;
			switch (t->kind) {
			case svsl_type_struct: {
				if (!const_literal(e, idx, &lit)) {
					skip(e, in->loc, "a dynamic struct member index");
					break;
				}
				const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
				path      = sfmt(e, "%s.%s", path, struct_member_name(e, t->struct_index, (int32_t)lit));
				base_type = info->members.items[lit].type;
				break;
			}
			case svsl_type_array:
				path      = sfmt(e, "%s[%s]", path, val(e, idx));
				base_type = t->elem;
				break;
			case svsl_type_buffer: // structured buffer element
				path      = sfmt(e, "%s[%s]", path, val(e, idx));
				base_type = t->elem;
				break;
			case svsl_type_matrix: // index selects an HLSL row = WGSL column
				path      = sfmt(e, "%s[%s]", path, val(e, idx));
				base_type = svsl_type_vector_id((svsl_types_t *)&e->prog->types, t->scalar, t->cols);
				break;
			case svsl_type_vector:
				path      = sfmt(e, "%s[%s]", path, val(e, idx));
				base_type = svsl_type_scalar_id((svsl_types_t *)&e->prog->types, t->scalar);
				break;
			default:
				skip(e, in->loc, "an address chain through a type the WGSL backend can't index");
				break;
			}
		}
		*out_type = base_type;
		return path;
	}
	default:
		skip(e, in->loc, "an instruction used as a pointer that isn't one (op %d)", in->op);
		*out_type = SVSL_TYPE_NONE;
		return "_bad";
	}
}

static const char *val(wgsl_t *e, uint32_t id) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	if (e->inline_expr && e->inline_expr[id]) return e->inline_expr[id];
	switch (in->op) {
	case svsl_ir_const:      return const_str(e, in);
	case svsl_ir_spec_const: return ident(e, e->prog->spec_consts.items[in->args[0]].name);
	case svsl_ir_undef:      return sfmt(e, "%s()", type_name_w(e, in->type, in->loc));
	default:                 return sfmt(e, "_%u", id);
	}
}

// vec4 → the instruction's narrower result, as a swizzle suffix
static const char *shrink4(wgsl_t *e, const char *expr, svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	int32_t count = t->kind == svsl_type_vector ? t->count : 1;
	static const char *sw[5] = { "", ".x", ".xy", ".xyz", "" };
	return count == 4 ? expr : sfmt(e, "%s%s", expr, sw[count]);
}

// ---- prescan: usage facts + unsupported features ---------------------------------

// marks every struct reachable from a type as used (declaration emission)
static void mark_struct(wgsl_t *e, svsl_type_id_t id) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, id);
	if (t->kind == svsl_type_array || t->kind == svsl_type_buffer ||
	    t->kind == svsl_type_texture || t->kind == svsl_type_image) {
		if (t->elem != SVSL_TYPE_NONE) mark_struct(e, t->elem);
		return;
	}
	if (t->kind != svsl_type_struct || e->struct_used[t->struct_index]) return;
	e->struct_used[t->struct_index] = 1;
	const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
	for (int32_t m = 0; m < info->members.count; m++)
		mark_struct(e, info->members.items[m].type);
}

static void prescan_type(wgsl_t *e, svsl_type_id_t id, svsl_loc_t loc) {
	if (id == SVSL_TYPE_NONE) return;
	const svsl_type_t *t = svsl_type_get(&e->prog->types, id);
	if (t->kind == svsl_type_scalar || t->kind == svsl_type_vector || t->kind == svsl_type_matrix) {
		switch (t->scalar) {
		case svsl_scalar_float64:
			skip(e, loc, "double-precision floats are used, and WGSL has no f64"); return;
		case svsl_scalar_int64: case svsl_scalar_uint64:
			skip(e, loc, "64-bit integers are used, and WGSL has no 64-bit types"); return;
		case svsl_scalar_int16: case svsl_scalar_uint16:
		case svsl_scalar_int8:  case svsl_scalar_uint8:
			skip(e, loc, "small integer types are used, and WGSL has no 8/16-bit integers"); return;
		case svsl_scalar_float16: e->uses_f16 = true; return;
		default: return;
		}
	}
	mark_struct(e, id);
}

static void prescan(wgsl_t *e) {
	const svsl_program_t *prog = e->prog;
	const svsl_ir_func_t *fn   = e->fn;

	if (fn->entry->wave_size > 0 || prog->wave_size > 0) {
		skip(e, fn->entry->func->loc, "[wave_size] pins the subgroup size, which WebGPU can't control");
		return;
	}
	for (int32_t i = 0; i < prog->spec_consts.count; i++)
		if (prog->spec_consts.items[i].id == WGSL_VIEW_INDEX_SPEC_ID) {
			svsl_diag_add(e->arena, e->diags, svsl_severity_error, prog->spec_consts.items[i].loc,
			              "spec constant id %d is reserved for sk_view_index when targeting WGSL; "
			              "pick another [[vk::constant_id]]", WGSL_VIEW_INDEX_SPEC_ID);
			e->skipped = true;
			return;
		}

	for (int32_t i = 0; i < fn->insts.count && !e->skipped; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		prescan_type(e, in->type, in->loc);
		if (e->skipped) return;

		switch ((svsl_ir_op_)in->op) {
		case svsl_ir_const:
			if (svsl_type_get(&prog->types, in->type)->scalar == svsl_scalar_float32 ||
			    svsl_type_get(&prog->types, in->type)->scalar == svsl_scalar_half)
				if ((in->args[0] & 0x7F800000u) == 0x7F800000u)
					skip(e, in->loc, "a NaN/Inf float constant, which WGSL cannot spell — "
					     "compute the value at runtime instead");
			break;
		case svsl_ir_spirv_asm:
			skip(e, in->loc, "inline SPIR-V assembly is inherently untranslatable"); break;
		case svsl_ir_atomic: {
			if ((in->args[3] >> 8) != 0) {
				skip(e, in->loc, "explicit atomic memory orders beyond relaxed have no WGSL "
				     "equivalent (WGSL atomics use a single implicit ordering)");
				break;
			}
			const svsl_type_t *vt = svsl_type_get(&prog->types, in->type);
			if (vt->kind != svsl_type_scalar ||
			    (vt->scalar != svsl_scalar_int32 && vt->scalar != svsl_scalar_uint32)) {
				skip(e, in->loc, "WGSL atomics only exist for 32-bit integers (no float atomics)");
				break;
			}
			// resolve the pointer to its storage root and mark the leaf atomic
			uint32_t p = in->args[0];
			while (fn->insts.items[p].op == svsl_ir_chain) p = fn->insts.items[p].args[0];
			const svsl_ir_inst_t *root = &fn->insts.items[p];
			if (root->op != svsl_ir_ptr) {
				skip(e, in->loc, "an atomic on function-local memory"); break;
			}
			switch ((svsl_ref_)root->args[0]) {
			case svsl_ref_resource: {
				const svsl_type_t *rt = svsl_type_get(&prog->types,
				                                      prog->resources.items[root->args[1]].type);
				const svsl_type_t *et = svsl_type_get(&prog->types, rt->elem);
				if (et->kind != svsl_type_scalar) {
					skip(e, in->loc, "atomics into structured-buffer struct members aren't "
					     "supported yet — use a scalar element type");
					break;
				}
				e->res_atomic[root->args[1]] = 1;
				break;
			}
			case svsl_ref_buffer_member: {
				const svsl_buffer_t *buf = &prog->buffers.items[root->args[1]];
				const svsl_type_t   *mt  = svsl_type_get(&prog->types,
				                                         buf->members.items[root->args[2]].type);
				if (mt->kind == svsl_type_array) mt = svsl_type_get(&prog->types, mt->elem);
				if (mt->kind != svsl_type_scalar) {
					skip(e, in->loc, "atomics into nested struct members aren't supported yet");
					break;
				}
				e->buf_atomic[root->args[1]][root->args[2]] = 1;
				break;
			}
			case svsl_ref_workgroup: {
				const svsl_type_t *wt = svsl_type_get(&prog->types,
				                                      prog->workgroup_vars.items[root->args[1]].type);
				if (wt->kind == svsl_type_array) wt = svsl_type_get(&prog->types, wt->elem);
				if (wt->kind != svsl_type_scalar) {
					skip(e, in->loc, "atomics into workgroup struct members aren't supported yet");
					break;
				}
				e->wg_atomic[root->args[1]] = 1;
				break;
			}
			default:
				skip(e, in->loc, "an atomic on a storage kind the WGSL backend can't retype");
				break;
			}
			break;
		}
		case svsl_ir_image_atomic:
			skip(e, in->loc, "image atomics aren't in core WebGPU"); break;
		case svsl_ir_ptr:
			if ((svsl_ref_)in->args[0] == svsl_ref_buffer_member) {
				const svsl_buffer_t *buf = &prog->buffers.items[in->args[1]];
				if (buf->kind == svsl_block_pushconstant) {
					skip(e, in->loc, "push constants have no WebGPU equivalent"); break;
				}
				if (buf->kind == svsl_block_storagebuffer) {
					// block-form storage buffers declare through their resource entry
					for (int32_t r = 0; r < prog->resources.count; r++)
						if (prog->resources.items[r].buffer_index == (int32_t)in->args[1])
							e->res_used[r] = 1;
				} else {
					e->buffer_used[in->args[1]] = 1;
				}
			}
			if ((svsl_ref_)in->args[0] == svsl_ref_resource) e->res_used[in->args[1]] = 1;
			break;
		case svsl_ir_tex: {
			int32_t tex     = (int32_t)in->args[0];
			int32_t method  = (int32_t)(in->args[2] & 0xFF);
			e->res_used[tex] = 1;
			if (in->args[1] != SVSL_IR_NONE) {
				e->res_used[in->args[1]] = 1;
				if (e->sampler_pair[in->args[1]] < 0) e->sampler_pair[in->args[1]] = tex;
			}
			const svsl_type_t *ot = svsl_type_get(&prog->types, prog->resources.items[tex].type);
			if (ot->kind == svsl_type_subpass)
				e->uses_subpass = true; // lowers to a textureLoad at this fragment's pixel
			else if (ot->kind == svsl_type_tileimage)
				skip(e, in->loc, "tile images have no WebGPU equivalent");
			else if (method == svsl_method_sample_weighted || method == svsl_method_box_filter ||
			         method == svsl_method_block_match)
				skip(e, in->loc, "QCOM image-processing ops have no WebGPU equivalent");
			else if (method == svsl_method_query_lod)
				skip(e, in->loc, "CalculateLevelOfDetail has no WGSL builtin");
			else if (method == svsl_method_sample_cmp || method == svsl_method_sample_cmp_level_zero ||
			         (method == svsl_method_gather && (int8_t)((in->args[2] >> 8) & 0xFF) == 4))
				e->res_cmp[tex] |= 1;
			else if (method == svsl_method_sample || method == svsl_method_sample_level ||
			         method == svsl_method_sample_bias || method == svsl_method_sample_grad ||
			         method == svsl_method_gather)
				e->res_cmp[tex] |= 2;
			if (method == svsl_method_sample || method == svsl_method_sample_bias)
				e->uses_derivatives = true;
			break;
		}
		case svsl_ir_image_load:
			e->res_used[in->args[0]] = 1; e->res_access[in->args[0]] |= 1; break;
		case svsl_ir_image_store:
			e->res_used[in->args[0]] = 1; e->res_access[in->args[0]] |= 2; break;
		case svsl_ir_intrinsic: {
			const svsl_intrinsic_t *row = svsl_intrinsic_get((int32_t)in->args[0]);
			switch ((svsl_emit_)row->emit) {
			case svsl_emit_subgroup:
			case svsl_emit_builtin_var:
				skip(e, in->loc, "wave/subgroup operations aren't available in browser WebGPU"); break;
			case svsl_emit_barrier_device: case svsl_emit_barrier_all:
			case svsl_emit_barrier_device_sync: case svsl_emit_barrier_all_sync:
				skip(e, in->loc, "device-scope memory barriers have no WGSL equivalent "
				     "(WGSL barriers are workgroup-scoped)"); break;
			case svsl_emit_tile_depth: case svsl_emit_tile_stencil:
				skip(e, in->loc, "tile-image reads have no WebGPU equivalent"); break;
			case svsl_emit_is_helper:
				skip(e, in->loc, "IsHelperLane has no WGSL builtin"); break;
			case svsl_emit_core:
				if (row->op[0] >= SpvOpDPdx && row->op[0] <= SpvOpFwidthCoarse)
					e->uses_derivatives = true;
				if (row->op[0] == SpvOpIsInf)
					skip(e, in->loc, "isinf cannot be expressed in WGSL (no Inf constants)");
				break;
			default: break;
			}
			break;
		}
		default: break;
		}
	}
	if (e->skipped) return;

	// depth-ness is a declaration (the runtime builds layouts from the paired
	// sampler's type), so per-stage usage must agree with it
	for (int32_t r = 0; r < prog->resources.count; r++) {
		if (!e->res_used[r] || prog->resources.items[r].kind != svsl_res_texture) continue;
		bool depth = tex_is_depth(e, r);
		if ((e->res_cmp[r] & 1) && !depth) {
			skip(e, prog->resources.items[r].loc, "texture '%.*s' is compare-sampled but has no "
			     "paired comparison sampler (same s-register); the WebGPU runtime binds depth "
			     "views by that pairing", prog->resources.items[r].name.len,
			     prog->resources.items[r].name.ptr);
			return;
		}
		if ((e->res_cmp[r] & 2) && depth) {
			skip(e, prog->resources.items[r].loc, "texture '%.*s' is paired with a comparison "
			     "sampler but also plain-sampled; a WGSL depth texture can't do both",
			     prog->resources.items[r].name.len, prog->resources.items[r].name.ptr);
			return;
		}
	}

	// buffers: layout gauge + used-resource types
	for (int32_t b = 0; b < prog->buffers.count && !e->skipped; b++)
		if (e->buffer_used[b]) {
			check_buffer_layout(e, &prog->buffers.items[b]);
			for (int32_t m = 0; m < prog->buffers.items[b].members.count; m++)
				prescan_type(e, prog->buffers.items[b].members.items[m].type,
				             prog->buffers.items[b].members.items[m].loc);
		}
	for (int32_t r = 0; r < prog->resources.count && !e->skipped; r++) {
		if (!e->res_used[r]) continue;
		const svsl_resource_t *res = &prog->resources.items[r];
		if (res->tile_attachment) { skip(e, res->loc, "tile attachments have no WebGPU equivalent"); break; }
		const svsl_type_t *t = svsl_type_get(&prog->types, res->type);
		if (t->kind == svsl_type_array) { skip(e, res->loc, "resource arrays aren't in the WGSL backend yet"); break; }
		if (t->elem != SVSL_TYPE_NONE) prescan_type(e, t->elem, res->loc);
		// block-form storage buffers reference their block's layout
		if (res->buffer_index >= 0) {
			e->buffer_used[res->buffer_index] = 0; // declared through the resource, not as a cbuffer
			check_buffer_layout(e, &prog->buffers.items[res->buffer_index]);
			for (int32_t m = 0; m < prog->buffers.items[res->buffer_index].members.count; m++)
				prescan_type(e, prog->buffers.items[res->buffer_index].members.items[m].type,
				             prog->buffers.items[res->buffer_index].members.items[m].loc);
		}
		// object-form structured buffers: element stride must match WGSL's
		if (t->kind == svsl_type_buffer && res->buffer_index < 0) {
			uint32_t ea, es;
			if (!wgsl_layout(e, t->elem, false, &ea, &es) || round_up(es, ea) != res->element_size) {
				skip(e, res->loc, "structured buffer '%.*s' has an element stride WGSL's storage "
				     "layout can't reproduce", res->name.len, res->name.ptr);
				break;
			}
		}
		if (t->kind == svsl_type_image) {
			uint32_t    fmt  = svsl_image_format_for(&prog->types, t);
			const char *name = wgsl_texel_format(fmt);
			uint8_t     acc  = e->res_access[r];
			if (!name) { skip(e, res->loc, "storage image '%.*s' uses a texel format WebGPU has no "
			                  "storage support for", res->name.len, res->name.ptr); break; }
			if (acc == 3 && fmt != SpvImageFormatR32f && fmt != SpvImageFormatR32ui && fmt != SpvImageFormatR32i) {
				skip(e, res->loc, "storage image '%.*s' is both read and written, which WebGPU only "
				     "allows on r32float/r32uint/r32sint", res->name.len, res->name.ptr);
				break;
			}
		}
	}

	// const globals: materialize each used initializer as a WGSL const-expression
	for (int32_t i = 0; i < fn->insts.count && !e->skipped; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op == svsl_ir_ptr && (svsl_ref_)in->args[0] == svsl_ref_const_global) {
			const svsl_global_t *g = &prog->const_globals.items[in->args[1]];
			if (e->const_texts[in->args[1]] || g->has_int) continue;
			const char *text = g->var && g->var->init
			                 ? const_global_text(e, g->var->init, g->type) : NULL;
			if (!text)
				skip(e, in->loc, "static const '%.*s' has an initializer the WGSL backend "
				     "can't fold to a constant", g->name.len, g->name.ptr);
			e->const_texts[in->args[1]] = text;
		}
		if (in->op == svsl_ir_ptr && (svsl_ref_)in->args[0] == svsl_ref_workgroup)
			prescan_type(e, prog->workgroup_vars.items[in->args[1]].type, in->loc);
	}
}

// ---- intrinsics ------------------------------------------------------------------

static const char *ext450_name(uint32_t op) {
	switch (op) {
	case GLSLstd450Sin:   return "sin";   case GLSLstd450Cos:   return "cos";
	case GLSLstd450Tan:   return "tan";   case GLSLstd450Asin:  return "asin";
	case GLSLstd450Acos:  return "acos";  case GLSLstd450Atan:  return "atan";
	case GLSLstd450Sinh:  return "sinh";  case GLSLstd450Cosh:  return "cosh";
	case GLSLstd450Tanh:  return "tanh";  case GLSLstd450Atan2: return "atan2";
	case GLSLstd450Pow:   return "pow";   case GLSLstd450Exp:   return "exp";
	case GLSLstd450Exp2:  return "exp2";  case GLSLstd450Log:   return "log";
	case GLSLstd450Log2:  return "log2";  case GLSLstd450Sqrt:  return "sqrt";
	case GLSLstd450InverseSqrt: return "inverseSqrt";
	case GLSLstd450Degrees: return "degrees"; case GLSLstd450Radians: return "radians";
	case GLSLstd450FAbs:  return "abs";   case GLSLstd450SAbs:  return "abs";
	case GLSLstd450FSign: return "sign";  case GLSLstd450SSign: return "sign";
	case GLSLstd450Floor: return "floor"; case GLSLstd450Ceil:  return "ceil";
	case GLSLstd450Trunc: return "trunc"; case GLSLstd450Round: return "round";
	case GLSLstd450RoundEven: return "round";
	case GLSLstd450Fract: return "fract";
	case GLSLstd450FMin: case GLSLstd450SMin: case GLSLstd450UMin: return "min";
	case GLSLstd450FMax: case GLSLstd450SMax: case GLSLstd450UMax: return "max";
	case GLSLstd450FClamp: case GLSLstd450SClamp: case GLSLstd450UClamp: return "clamp";
	case GLSLstd450FMix:  return "mix";   case GLSLstd450Step:  return "step";
	case GLSLstd450SmoothStep: return "smoothstep";
	case GLSLstd450Fma:   return "fma";
	case GLSLstd450Length: return "length"; case GLSLstd450Distance: return "distance";
	case GLSLstd450Normalize: return "normalize"; case GLSLstd450Cross: return "cross";
	case GLSLstd450Reflect: return "reflect"; case GLSLstd450Refract: return "refract";
	case GLSLstd450FaceForward: return "faceForward";
	case GLSLstd450Determinant: return "determinant";
	case GLSLstd450FindILsb: return "firstTrailingBit";
	case GLSLstd450FindSMsb: case GLSLstd450FindUMsb: return "firstLeadingBit";
	case GLSLstd450PackHalf2x16:   return "pack2x16float";
	case GLSLstd450UnpackHalf2x16: return "unpack2x16float";
	case GLSLstd450PackUnorm4x8:   return "pack4x8unorm";
	case GLSLstd450PackSnorm4x8:   return "pack4x8snorm";
	case GLSLstd450UnpackUnorm4x8: return "unpack4x8unorm";
	case GLSLstd450UnpackSnorm4x8: return "unpack4x8snorm";
	case GLSLstd450NMin: return "min"; case GLSLstd450NMax: return "max";
	default: return NULL;
	}
}

// zero/one splat matching a value's shape, for saturate/rcp lowerings
static const char *splat(wgsl_t *e, svsl_type_id_t type, const char *scalar, svsl_loc_t loc) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	if (t->kind == svsl_type_vector)
		return sfmt(e, "vec%d<%s>(%s)", t->count, scalar_name(e, t->scalar, loc), scalar);
	return sfmt(e, "%s(%s)", scalar_name(e, t->scalar, loc), scalar);
}

// argument list "a, b, c" from the instruction's aux operands; built linearly —
// re-formatting the accumulated prefix per operand was quadratic
static const char *aux_args(wgsl_t *e, const svsl_ir_inst_t *in) {
	svsl_array_t(char) buf = {0};
	for (uint32_t i = 0; i < in->aux_count; i++) {
		if (i) { svsl_array_push(e->arena, &buf, ','); svsl_array_push(e->arena, &buf, ' '); }
		for (const char *c = val(e, e->fn->aux.items[in->aux + i]); *c; c++)
			svsl_array_push(e->arena, &buf, *c);
	}
	svsl_array_push(e->arena, &buf, '\0');
	return buf.items;
}

// returns the value expression for an intrinsic, or NULL when it emitted a
// whole statement itself (barriers, clip)
static const char *intrinsic_expr(wgsl_t *e, const svsl_ir_inst_t *in) {
	const svsl_intrinsic_t *row = svsl_intrinsic_get((int32_t)in->args[0]);
	uint32_t a_count = in->aux_count;
	const char *a0 = a_count > 0 ? val(e, e->fn->aux.items[in->aux + 0]) : "";
	const char *a1 = a_count > 1 ? val(e, e->fn->aux.items[in->aux + 1]) : "";
	svsl_type_id_t a0_type = a_count > 0 ? e->fn->insts.items[e->fn->aux.items[in->aux]].type : SVSL_TYPE_NONE;

	switch ((svsl_emit_)row->emit) {
	case svsl_emit_ext450: {
		const char *name = ext450_name(row->op[0]);
		if (!name && row->op[1]) name = ext450_name(row->op[1]);
		if (!name) { skip(e, in->loc, "'%s' has no WGSL builtin", row->name); return "0"; }
		return sfmt(e, "%s(%s)", name, aux_args(e, in));
	}
	case svsl_emit_core:
		switch (row->op[0]) {
		case SpvOpDot:         return sfmt(e, "dot(%s, %s)", a0, a1);
		case SpvOpFRem:        return sfmt(e, "(%s %% %s)", a0, a1);
		case SpvOpDPdx:        return sfmt(e, "dpdx(%s)", a0);
		case SpvOpDPdy:        return sfmt(e, "dpdy(%s)", a0);
		case SpvOpFwidth:      return sfmt(e, "fwidth(%s)", a0);
		case SpvOpDPdxFine:    return sfmt(e, "dpdxFine(%s)", a0);
		case SpvOpDPdyFine:    return sfmt(e, "dpdyFine(%s)", a0);
		case SpvOpDPdxCoarse:  return sfmt(e, "dpdxCoarse(%s)", a0);
		case SpvOpDPdyCoarse:  return sfmt(e, "dpdyCoarse(%s)", a0);
		case SpvOpFwidthFine:  return sfmt(e, "fwidthFine(%s)", a0);
		case SpvOpFwidthCoarse:return sfmt(e, "fwidthCoarse(%s)", a0);
		case SpvOpBitCount:    return sfmt(e, "countOneBits(%s)", a0);
		case SpvOpBitReverse:  return sfmt(e, "reverseBits(%s)", a0);
		case SpvOpTranspose:   return sfmt(e, "transpose(%s)", a0);
		case SpvOpIsNan:       return sfmt(e, "(%s != %s)", a0, a0);
		default:
			skip(e, in->loc, "'%s' has no WGSL mapping yet", row->name);
			return "0";
		}
	case svsl_emit_rcp:      return sfmt(e, "(%s / %s)", splat(e, in->type, "1", in->loc), a0);
	case svsl_emit_saturate: return sfmt(e, "clamp(%s, %s, %s)", a0,
	                                     splat(e, in->type, "0", in->loc), splat(e, in->type, "1", in->loc));
	case svsl_emit_log10:    return sfmt(e, "(log(%s) * %s)", a0, splat(e, in->type, "0.43429448190325176", in->loc));
	case svsl_emit_ldexp: {  // HLSL ldexp exponent is float; WGSL wants i32
		const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
		const char *ei = t->kind == svsl_type_vector ? sfmt(e, "vec%d<i32>(%s)", t->count, a1)
		                                             : sfmt(e, "i32(%s)", a1);
		return sfmt(e, "ldexp(%s, %s)", a0, ei);
	}
	case svsl_emit_frexp_mant: return sfmt(e, "frexp(%s).fract", a0);
	case svsl_emit_frexp_exp:  return sfmt(e, "%s(frexp(%s).exp)", type_name_w(e, in->type, in->loc), a0);
	case svsl_emit_any:        return sfmt(e, "any(%s)", a0);
	case svsl_emit_all:        return sfmt(e, "all(%s)", a0);
	case svsl_emit_bitcast: {
		// asfloat of constant NaN/Inf bits is a WGSL shader-creation error when
		// const-evaluated; routing the bits through a var defers the bitcast to
		// runtime, where the value is legal — exactly what the source intends
		const svsl_type_t    *rt  = svsl_type_get(&e->prog->types, in->type);
		const svsl_ir_inst_t *src = a_count > 0 ? &e->fn->insts.items[e->fn->aux.items[in->aux]] : NULL;
		if (src && src->op == svsl_ir_const && rt->kind == svsl_type_scalar &&
		    (rt->scalar == svsl_scalar_float32 || rt->scalar == svsl_scalar_half) &&
		    (src->args[0] & 0x7F800000u) == 0x7F800000u) {
			int32_t idx = (int32_t)(in - e->fn->insts.items);
			wln(e, "var _bc%d : %s = %s;", idx, type_name_w(e, src->type, in->loc), a0);
			return sfmt(e, "bitcast<%s>(_bc%d)", type_name_w(e, in->type, in->loc), idx);
		}
		return sfmt(e, "bitcast<%s>(%s)", type_name_w(e, in->type, in->loc), a0);
	}
	case svsl_emit_f16tof32: {
		const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
		if (t->kind == svsl_type_vector) { skip(e, in->loc, "vector f16tof32 isn't mapped yet"); return "0"; }
		return sfmt(e, "unpack2x16float(%s).x", a0);
	}
	case svsl_emit_f32tof16: {
		const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
		if (t->kind == svsl_type_vector) { skip(e, in->loc, "vector f32tof16 isn't mapped yet"); return "0"; }
		return sfmt(e, "(pack2x16float(vec2<f32>(%s, 0.0)) & 0xffffu)", a0);
	}
	case svsl_emit_bitfield_extract:
		return sfmt(e, "extractBits(%s, u32(%s), u32(%s))", a0, a1,
		            a_count > 2 ? val(e, e->fn->aux.items[in->aux + 2]) : "0");
	case svsl_emit_bitfield_insert:
		return sfmt(e, "insertBits(%s, %s, u32(%s), u32(%s))", a0, a1,
		            a_count > 2 ? val(e, e->fn->aux.items[in->aux + 2]) : "0",
		            a_count > 3 ? val(e, e->fn->aux.items[in->aux + 3]) : "0");
	case svsl_emit_clip: { // clip(x): discard when any component < 0
		const svsl_type_t *t = svsl_type_get(&e->prog->types, a0_type);
		if (t->kind == svsl_type_vector)
			wln(e, "if (any(%s < %s)) { discard; }", a0, splat(e, a0_type, "0", in->loc));
		else
			wln(e, "if (%s < 0.0) { discard; }", a0);
		return NULL;
	}
	case svsl_emit_barrier_wg:
	case svsl_emit_barrier_wg_mem: // WGSL's nearest barrier is control + workgroup memory
		wln(e, "workgroupBarrier();");
		return NULL;
	default:
		skip(e, in->loc, "'%s' has no WGSL mapping yet", row->name);
		return "0";
	}
}

// ---- texture ops ------------------------------------------------------------------

static const char *tex_expr(wgsl_t *e, const svsl_ir_inst_t *in) {
	int32_t  tex     = (int32_t)in->args[0];
	uint32_t sampler = in->args[1];
	int32_t  method  = (int32_t)(in->args[2] & 0xFF);
	int32_t  channel = (int32_t)(int8_t)((in->args[2] >> 8) & 0xFF);
	const svsl_resource_t *res = &e->prog->resources.items[tex];
	const svsl_type_t     *ot  = svsl_type_get(&e->prog->types, res->type);
	bool pixel = e->fn->entry->stage == svsl_stage_pixel;

	const char *a0 = in->aux_count > 0 ? val(e, e->fn->aux.items[in->aux + 0]) : "";
	const char *a1 = in->aux_count > 1 ? val(e, e->fn->aux.items[in->aux + 1]) : "";
	const char *a2 = in->aux_count > 2 ? val(e, e->fn->aux.items[in->aux + 2]) : "";

	// structured Buffer<T>: element access / length, never an image op
	if (ot->kind == svsl_type_buffer) {
		if (method == svsl_method_get_dimensions)
			return sfmt(e, "arrayLength(&%s)", res_name(e, tex));
		if (e->res_atomic[tex]) // atomic-retyped elements read through atomicLoad
			return sfmt(e, "atomicLoad(&%s[%s])", res_name(e, tex), a0);
		return sfmt(e, "%s[%s]", res_name(e, tex), a0);
	}

	const char *t = res_name(e, tex);

	// SubpassInput[MS].Load(): the runtime lowers postfx chains to sequential
	// passes binding the previous stage as a plain texture, so a subpass read
	// is exactly a fetch at this fragment's own pixel
	if (ot->kind == svsl_type_subpass) {
		const char *coord = sfmt(e, "vec2<i32>(%s.xy)", e->frag_pos);
		if (ot->multisampled)
			return shrink4(e, sfmt(e, "textureLoad(%s, %s, %s)", t, coord, a0), in->type);
		return shrink4(e, sfmt(e, "textureLoad(%s, %s, 0)", t, coord), in->type);
	}

	const char *s = sampler_ref(e, tex, sampler);

	// arrayed textures split the packed HLSL coordinate into uv + array index,
	// which WGSL takes as its own argument right after the coordinate
	const char *uv = a0;
	if (ot->arrayed && ot->kind == svsl_type_texture) {
		int32_t n = ot->dim == svsl_texdim_cube ? 3 : ot->dim == svsl_texdim_1d ? 1 : 2;
		static const char *csw[4] = { "", ".x", ".xy", ".xyz" };
		static const char *lsw[4] = { ".y", ".y", ".z", ".w" };
		uv = sfmt(e, "%s%s, i32(%s%s)", a0, csw[n], a0, lsw[n]);
	}

	switch ((svsl_method_)method) {
	case svsl_method_sample:
		if (!pixel) return shrink4(e, sfmt(e, "textureSampleLevel(%s, %s, %s, 0.0)", t, s, uv), in->type);
		return shrink4(e, sfmt(e, "textureSample(%s, %s, %s)", t, s, uv), in->type);
	case svsl_method_sample_bias:
		if (!pixel) return shrink4(e, sfmt(e, "textureSampleLevel(%s, %s, %s, 0.0)", t, s, uv), in->type);
		return shrink4(e, sfmt(e, "textureSampleBias(%s, %s, %s, %s)", t, s, uv, a1), in->type);
	case svsl_method_sample_level:
		return shrink4(e, sfmt(e, "textureSampleLevel(%s, %s, %s, %s)", t, s, uv, a1), in->type);
	case svsl_method_sample_grad:
		return shrink4(e, sfmt(e, "textureSampleGrad(%s, %s, %s, %s, %s)", t, s, uv, a1, a2), in->type);
	case svsl_method_sample_cmp:
		if (!pixel) return sfmt(e, "textureSampleCompareLevel(%s, %s, %s, %s)", t, s, uv, a1);
		return sfmt(e, "textureSampleCompare(%s, %s, %s, %s)", t, s, uv, a1);
	case svsl_method_sample_cmp_level_zero:
		return sfmt(e, "textureSampleCompareLevel(%s, %s, %s, %s)", t, s, uv, a1);
	case svsl_method_gather:
		if (channel == 4) // GatherCmp
			return shrink4(e, sfmt(e, "textureGatherCompare(%s, %s, %s, %s)", t, s, uv, a1), in->type);
		return shrink4(e, sfmt(e, "textureGather(%d, %s, %s, %s)", channel < 0 ? 0 : channel, t, s, uv), in->type);
	case svsl_method_load: { // fetch: coordinate packs [dims..., layer,] mip/sample
		static const char *csw[4]  = { "", ".x", ".xy", ".xyz" };
		static const char *comp[5] = { ".x", ".y", ".z", ".w", "" };
		if (ot->multisampled) {
			if (ot->arrayed) { skip(e, in->loc, "WGSL has no multisampled array textures"); return "0"; }
			return shrink4(e, sfmt(e, "textureLoad(%s, %s, %s)", t, a0, a1), in->type);
		}
		int32_t n = ot->dim == svsl_texdim_1d ? 1 : ot->dim == svsl_texdim_3d ? 3 : 2;
		if (ot->arrayed) // layer sits between the dims and the mip
			return shrink4(e, sfmt(e, "textureLoad(%s, %s%s, %s%s, %s%s)",
			               t, a0, csw[n], a0, comp[n], a0, comp[n + 1]), in->type);
		return shrink4(e, sfmt(e, "textureLoad(%s, %s%s, %s%s)", t, a0, csw[n], a0, comp[n]), in->type);
	}
	case svsl_method_get_dimensions: {
		const char *tn = type_name_w(e, in->type, in->loc);
		if (channel == 1) // trailing level/sample-count query
			return sfmt(e, "%s(%s(%s))", tn, ot->multisampled ? "textureNumSamples" : "textureNumLevels", t);
		// SPIR-V's size query returns the layer count as the last component;
		// WGSL splits it into textureNumLayers, so arrayed queries recompose
		int32_t n = ot->dim == svsl_texdim_1d ? 1 : ot->dim == svsl_texdim_3d ? 3 : 2;
		if (ot->kind == svsl_type_image || ot->multisampled) {
			if (ot->arrayed)
				return sfmt(e, "%s(vec%d<u32>(textureDimensions(%s), textureNumLayers(%s)))",
				            tn, n + 1, t, t);
			return sfmt(e, "%s(textureDimensions(%s))", tn, t);
		}
		const char *lod = in->aux_count > 0 ? sfmt(e, "i32(%s)", a0) : "0";
		if (ot->arrayed)
			return sfmt(e, "%s(vec%d<u32>(textureDimensions(%s, %s), textureNumLayers(%s)))",
			            tn, n + 1, t, lod, t);
		return sfmt(e, "%s(textureDimensions(%s, %s))", tn, t, lod);
	}
	default:
		skip(e, in->loc, "a texture method the WGSL backend doesn't map yet (%d)", method);
		return "0";
	}
}

// storage-image store value must be a vec4
static const char *expand4(wgsl_t *e, uint32_t id, svsl_loc_t loc) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	const svsl_type_t    *t  = svsl_type_get(&e->prog->types, in->type);
	const char *v  = val(e, id);
	const char *sc = scalar_name(e, t->scalar, loc);
	int32_t count  = t->kind == svsl_type_vector ? t->count : 1;
	switch (count) {
	case 1:  return sfmt(e, "vec4<%s>(%s)", sc, v); // splat; single-channel formats read .x
	case 2:  return sfmt(e, "vec4<%s>(%s, 0, 0)", sc, v);
	case 3:  return sfmt(e, "vec4<%s>(%s, 0)", sc, v);
	default: return v;
	}
}

// ---- entry IO --------------------------------------------------------------------

// WGSL builtin name for a SPIR-V BuiltIn in this stage/direction, NULL = none
static const char *builtin_name(uint32_t builtin, bool input) {
	switch (builtin) {
	case SpvBuiltInPosition:             return "position"; // vs out
	case SpvBuiltInFragCoord:            return "position"; // ps in
	case SpvBuiltInVertexIndex:          return "vertex_index";
	case SpvBuiltInInstanceIndex:        return "instance_index";
	case SpvBuiltInFragDepth:            return "frag_depth";
	case SpvBuiltInFrontFacing:          return "front_facing";
	case SpvBuiltInSampleId:             return "sample_index";
	case SpvBuiltInSampleMask:           return "sample_mask";
	case SpvBuiltInGlobalInvocationId:   return "global_invocation_id";
	case SpvBuiltInLocalInvocationId:    return "local_invocation_id";
	case SpvBuiltInWorkgroupId:          return "workgroup_id";
	case SpvBuiltInLocalInvocationIndex: return "local_invocation_index";
	default:                             (void)input; return NULL;
	}
}

// one flattened IO field: a non-struct param, a struct-param member, the
// return value, or a return-struct member
typedef struct io_field_t {
	svsl_str_t     name;      // member name in the IO struct
	svsl_type_id_t type;
	svsl_str_t     semantic;
	int32_t        param;     // owning param, -1 for the return value
	int32_t        member;    // member within the param/return struct, -1 = whole
	int32_t        location;  // -1 = builtin
	const char    *builtin;   // WGSL builtin name; NULL = location-numbered
	bool           view_index;// SV_ViewID: fed from the sk_view_index override
} io_field_t;

typedef struct io_list_t {
	io_field_t fields[64];
	int32_t    count;
} io_list_t;

// builtins that WGSL declares with a fixed type; loads convert to the declared
// HLSL type when it's narrower (uint id : SV_DispatchThreadID)
static const char *builtin_decl_type(const char *builtin) {
	if (strcmp(builtin, "position") == 0)               return "vec4<f32>";
	if (strcmp(builtin, "front_facing") == 0)           return "bool";
	if (strcmp(builtin, "frag_depth") == 0)             return "f32";
	if (strcmp(builtin, "vertex_index") == 0 ||
	    strcmp(builtin, "instance_index") == 0 ||
	    strcmp(builtin, "sample_index") == 0 ||
	    strcmp(builtin, "sample_mask") == 0 ||
	    strcmp(builtin, "local_invocation_index") == 0) return "u32";
	return "vec3<u32>"; // the compute id builtins
}

static void io_add(wgsl_t *e, io_list_t *io, svsl_str_t name, svsl_type_id_t type,
                   svsl_str_t semantic, int32_t param, int32_t member,
                   int32_t *ref_location, svsl_sem_io_ sem_io, svsl_loc_t loc,
                   int32_t explicit_location) {
	if (io->count >= 64) { skip(e, loc, "more than 64 stage IO fields"); return; }
	io_field_t *f = &io->fields[io->count++];
	*f = (io_field_t){ .name = name, .type = type, .semantic = semantic,
	                   .param = param, .member = member, .location = -1 };

	svsl_semantic_info_t info;
	if (svsl_semantic_lookup(semantic, sem_io, &info) && info.is_builtin) {
		if (info.builtin == SpvBuiltInViewIndex) { f->view_index = true; e->uses_view_index = true; return; }
		if (info.builtin == SpvBuiltInLayer) {
			skip(e, loc, "SV_RenderTargetArrayIndex routes primitives to a layered-target slice, "
			     "which WebGPU cannot express — use SV_ViewID and multiview instead");
			return;
		}
		f->builtin = builtin_name(info.builtin, sem_io != svsl_sem_vs_out && sem_io != svsl_sem_ps_out);
		if (!f->builtin)
			skip(e, loc, "the '%.*s' semantic has no WGSL builtin", semantic.len, semantic.ptr);
		return;
	}
	// SV_TargetN pins its location; everything else numbers sequentially
	if (sem_io == svsl_sem_ps_out && svsl_semantic_lookup(semantic, sem_io, &info)) {
		f->location = info.target_index;
		return;
	}
	if (explicit_location >= 0) *ref_location = explicit_location;
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	if (t->kind == svsl_type_matrix || t->kind == svsl_type_array || t->kind == svsl_type_struct) {
		skip(e, loc, "matrix/array/nested-struct stage IO isn't mapped for WGSL yet");
		return;
	}
	f->location = (*ref_location)++;
}

// flattens the entry's params (or return type) into IO fields
static void io_collect_params(wgsl_t *e, io_list_t *io, svsl_sem_io_ sem_io) {
	const svsl_entry_t     *entry = e->fn->entry;
	const svsl_func_info_t *fi    = svsl_program_func_info(e->prog, entry->func);
	int32_t location = 0;
	for (int32_t p = 0; p < entry->func->param_count && !e->skipped; p++) {
		const svsl_ast_var_t *pv = entry->func->params[p];
		const svsl_type_t    *pt = svsl_type_get(&e->prog->types, fi->param_types[p]);
		if (pt->kind == svsl_type_struct) {
			const svsl_struct_info_t *info = &e->prog->types.structs.items[pt->struct_index];
			for (int32_t m = 0; m < info->members.count && !e->skipped; m++) {
				const svsl_member_t *mem = &info->members.items[m];
				io_add(e, io, mem->name, mem->type, mem->semantic, p, m,
				       &location, sem_io, mem->loc, mem->explicit_location);
			}
		} else {
			io_add(e, io, pv->name, fi->param_types[p], pv->semantic, p, -1,
			       &location, sem_io, pv->loc, -1);
		}
	}
}

static void io_collect_return(wgsl_t *e, io_list_t *io, svsl_sem_io_ sem_io) {
	const svsl_entry_t     *entry = e->fn->entry;
	const svsl_func_info_t *fi    = svsl_program_func_info(e->prog, entry->func);
	if (fi->return_type == SVSL_TYPE_NONE) return;
	const svsl_type_t *rt = svsl_type_get(&e->prog->types, fi->return_type);
	int32_t location = 0;
	if (rt->kind == svsl_type_struct) {
		const svsl_struct_info_t *info = &e->prog->types.structs.items[rt->struct_index];
		for (int32_t m = 0; m < info->members.count && !e->skipped; m++) {
			const svsl_member_t *mem = &info->members.items[m];
			io_add(e, io, mem->name, mem->type, mem->semantic, -1, m,
			       &location, sem_io, mem->loc, mem->explicit_location);
		}
	} else {
		io_add(e, io, svsl_str("result"), fi->return_type, entry->func->return_semantic,
		       -1, -1, &location, sem_io, entry->func->loc, -1);
	}
}

// integer varyings into a pixel stage must interpolate flat
static bool needs_flat(wgsl_t *e, const io_field_t *f) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, f->type);
	return t->scalar == svsl_scalar_int32 || t->scalar == svsl_scalar_uint32 || t->scalar == svsl_scalar_bool;
}

// `varying` marks the interpolated vs↔ps interface, where WGSL requires
// integer fields to declare @interpolate(flat) on both sides
static void emit_io_struct(wgsl_t *e, const char *name, const io_list_t *io, bool varying) {
	bool any = false;
	for (int32_t i = 0; i < io->count; i++)
		if (!io->fields[i].view_index) any = true;
	if (!any) return;
	wln(e, "struct %s {", name);
	e->indent++;
	for (int32_t i = 0; i < io->count; i++) {
		const io_field_t *f = &io->fields[i];
		if (f->view_index) continue;
		const char *mn = ident(e, f->name);
		if (f->builtin)
			wln(e, "@builtin(%s) %s : %s,", f->builtin, mn, builtin_decl_type(f->builtin));
		else if (varying && needs_flat(e, f))
			wln(e, "@location(%d) @interpolate(flat) %s : %s,", f->location, mn,
			    type_name_w(e, f->type, (svsl_loc_t){0}));
		else
			wln(e, "@location(%d) %s : %s,", f->location, mn, type_name_w(e, f->type, (svsl_loc_t){0}));
	}
	e->indent--;
	wln(e, "}");
	wln(e, "");
}

// the expression reading input field `f` from the entry's `in` parameter,
// converted to the field's declared HLSL type
static const char *io_read(wgsl_t *e, const io_field_t *f) {
	if (f->view_index) return "sk_view_index";
	const char *expr = sfmt(e, "in.%s", ident(e, f->name));
	if (!f->builtin) return expr;
	// convert the WGSL builtin's fixed type to the declared one when they differ
	const svsl_type_t *t = svsl_type_get(&e->prog->types, f->type);
	const char *decl = builtin_decl_type(f->builtin);
	if (strcmp(decl, "vec3<u32>") == 0) { // compute ids: allow narrower declarations
		static const char *sw[5] = { "", ".x", ".xy", ".xyz", "" };
		int32_t count = t->kind == svsl_type_vector ? t->count : 1;
		const char *narrowed = count == 3 ? expr : sfmt(e, "%s%s", expr, sw[count]);
		if (t->scalar != svsl_scalar_uint32)
			return sfmt(e, "%s(%s)", type_name_w(e, f->type, (svsl_loc_t){0}), narrowed);
		return narrowed;
	}
	if (strcmp(decl, "u32") == 0 && t->scalar != svsl_scalar_uint32)
		return sfmt(e, "%s(%s)", type_name_w(e, f->type, (svsl_loc_t){0}), expr);
	if (strcmp(decl, "bool") == 0 && t->scalar != svsl_scalar_bool)
		return sfmt(e, "%s(%s)", type_name_w(e, f->type, (svsl_loc_t){0}), expr);
	return expr;
}

// ---- statements ------------------------------------------------------------------

// pure computations may inline anywhere their operand names stay in scope;
// state-readers (loads, texture ops) additionally need a clear path to their
// materialization point, checked in inline_analyze
static bool pure_inlinable(wgsl_t *e, const svsl_ir_inst_t *in) {
	switch ((svsl_ir_op_)in->op) {
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul: case svsl_ir_div:
	case svsl_ir_rem: case svsl_ir_neg: case svsl_ir_bit_not: case svsl_ir_log_not:
	case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
	case svsl_ir_shl: case svsl_ir_shr:
	case svsl_ir_eq: case svsl_ir_ne: case svsl_ir_lt: case svsl_ir_le:
	case svsl_ir_gt: case svsl_ir_ge: case svsl_ir_log_and: case svsl_ir_log_or:
	case svsl_ir_convert: case svsl_ir_mat_mul:
	case svsl_ir_construct: case svsl_ir_extract: case svsl_ir_shuffle:
	case svsl_ir_extract_dynamic:
	case svsl_ir_bitfield_extract: case svsl_ir_bitfield_insert:
		return true;
	case svsl_ir_select: { // the matrix/array/struct form emits statements
		const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
		return t->kind == svsl_type_scalar || t->kind == svsl_type_vector;
	}
	case svsl_ir_intrinsic: { // barriers and clip are statements, the rest expressions
		svsl_emit_ em = (svsl_emit_)svsl_intrinsic_get((int32_t)in->args[0])->emit;
		return em != svsl_emit_barrier_wg && em != svsl_emit_barrier_wg_mem && em != svsl_emit_clip;
	}
	default:
		return false;
	}
}

// Decides which values fold into their (single) use site instead of becoming a
// `let _N`. sink[i] = the statement where i's text is finally evaluated,
// chased through inlined users and through chains (whose text materializes at
// their user). State-readers only inline when no side-effecting or
// control-flow instruction (svsl_ir_has_side_effects — the shared oracle)
// separates their definition from that sink.
static void inline_analyze(wgsl_t *e, bool struct_return) {
	const svsl_ir_func_t *fn = e->fn;
	int32_t n = fn->insts.count > 0 ? fn->insts.count : 1;
	e->inline_ok   = svsl_arena_alloc(e->arena, (size_t)n);
	e->inline_expr = svsl_arena_alloc(e->arena, (size_t)n * sizeof(const char *));
	int32_t *use_count = svsl_arena_alloc(e->arena, (size_t)n * sizeof(int32_t));
	int32_t *last_use  = svsl_arena_alloc(e->arena, (size_t)n * sizeof(int32_t));
	int32_t *sink      = svsl_arena_alloc(e->arena, (size_t)n * sizeof(int32_t));
	int32_t *cum       = svsl_arena_alloc(e->arena, (size_t)(n + 1) * sizeof(int32_t));
	for (int32_t i = 0; i < n; i++) last_use[i] = -1;

	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		cum[i + 1] = cum[i] + (svsl_ir_has_side_effects(in, &e->prog->types) ? 1 : 0);
		// struct returns and cmpxchg print an operand's text more than once —
		// weight 2 forces those into lets so evaluation isn't duplicated
		int32_t  w    = in->op == svsl_ir_atomic ||
		                (in->op == svsl_ir_return && struct_return) ? 2 : 1;
		uint32_t mask = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++) {
			if (!(mask & (1u << a)) || in->args[a] == SVSL_IR_NONE) continue;
			if (in->args[a] >= (uint32_t)fn->insts.count) continue;
			use_count[in->args[a]] += w;
			last_use [in->args[a]]  = i;
		}
		if (svsl_ir_aux_holds_values(in))
			for (uint32_t k = 0; k < in->aux_count; k++) {
				uint32_t v = fn->aux.items[in->aux + k];
				if (v >= (uint32_t)fn->insts.count) continue;
				use_count[v] += w;
				last_use [v]  = i;
			}
	}

	for (int32_t i = fn->insts.count - 1; i >= 0; i--) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		sink[i] = i;
		if (in->op == svsl_ir_chain || in->op == svsl_ir_ptr) {
			// pointer text materializes wherever its user does; several users
			// mean several materializations, which no scan can bound
			sink[i] = (use_count[i] == 1 && last_use[i] >= 0) ? sink[last_use[i]] : -1;
			continue;
		}
		if (use_count[i] != 1 || last_use[i] < 0) continue;
		int32_t s = sink[last_use[i]];
		if (s < 0) continue;
		if (pure_inlinable(e, in)) {
			e->inline_ok[i] = 1;
			sink[i] = s;
		} else if (in->op == svsl_ir_load || in->op == svsl_ir_tex || in->op == svsl_ir_image_load) {
			if (cum[s] - cum[i + 1] == 0) { // nothing observable between def and sink
				e->inline_ok[i] = 1;
				sink[i] = s;
			}
		}
	}
}

static const char *binop_token(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_add: return "+";  case svsl_ir_sub: return "-";
	case svsl_ir_mul: return "*";  case svsl_ir_div: return "/";
	case svsl_ir_rem: return "%";
	case svsl_ir_bit_and: case svsl_ir_log_and: return "&";
	case svsl_ir_bit_or:  case svsl_ir_log_or:  return "|";
	case svsl_ir_bit_xor: return "^";
	case svsl_ir_shl: return "<<"; case svsl_ir_shr: return ">>";
	case svsl_ir_eq: return "=="; case svsl_ir_ne: return "!=";
	case svsl_ir_lt: return "<";  case svsl_ir_le: return "<=";
	case svsl_ir_gt: return ">";  case svsl_ir_ge: return ">=";
	default: return NULL;
	}
}

// shift amounts must be u32-typed in WGSL
static const char *shift_rhs(wgsl_t *e, uint32_t id, svsl_type_id_t lhs_type) {
	const svsl_type_t *lt = svsl_type_get(&e->prog->types, lhs_type);
	const char *v = val(e, id);
	if (lt->kind == svsl_type_vector) return sfmt(e, "vec%d<u32>(%s)", lt->count, v);
	return sfmt(e, "u32(%s)", v);
}

static void emit_body(wgsl_t *e, const io_list_t *out_io, const char *out_struct);

// builds and returns the entry's output from the IR return value
static void emit_return(wgsl_t *e, const svsl_ir_inst_t *in,
                        const io_list_t *out_io, const char *out_struct) {
	if (in->args[0] == SVSL_IR_NONE) { wln(e, "return;"); return; }
	const char *v = val(e, in->args[0]);
	if (out_io->count == 1 && out_io->fields[0].member == -1) { // single attributed value
		wln(e, "return %s;", v);
		return;
	}
	// struct return: copy user-struct members into the attributed IO struct
	wln(e, "var _out : %s;", out_struct);
	for (int32_t i = 0; i < out_io->count; i++) {
		const io_field_t *f = &out_io->fields[i];
		wln(e, "_out.%s = %s.%s;", ident(e, f->name), v, ident(e, f->name));
	}
	wln(e, "return _out;");
}

static void emit_body(wgsl_t *e, const io_list_t *out_io, const char *out_struct) {
	const svsl_ir_func_t *fn = e->fn;
	// switch emission state: the aux literal list of each open switch
	struct { const svsl_ir_inst_t *inst; bool open_case, seen_default; } sw[8];
	int32_t sw_depth = 0;
	// loop nesting: whether each open loop has entered its continuing block
	// (the early-return wrapper loop has no loop_continue marker)
	bool    loop_cont[32];
	int32_t loop_depth = 0;

	inline_analyze(e, out_io->count > 1 || (out_io->count == 1 && out_io->fields[0].member >= 0));

	// locals hoist to function scope like SPIR-V's OpVariables: pointers cross
	// block boundaries (only pure values are block-local in this IR)
	for (int32_t i = 0; i < fn->insts.count && !e->skipped; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op == svsl_ir_var || in->op == svsl_ir_param)
			wln(e, "var _%d : %s;", i, type_name_w(e, in->type, in->loc));
	}

	for (int32_t i = 0; i < fn->insts.count && !e->skipped; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		svsl_ir_op_           op = (svsl_ir_op_)in->op;
		const char *a  = binop_token(op) && in->args[0] != SVSL_IR_NONE ? val(e, in->args[0]) : NULL;
		const char *ex = NULL; // value expression: inlines into its use, or becomes a let

		switch (op) {
		case svsl_ir_nop: case svsl_ir_const: case svsl_ir_spec_const: case svsl_ir_undef:
		case svsl_ir_ptr: case svsl_ir_chain:
		case svsl_ir_var: // declared in the hoisted prologue
			break; // no statement; resolved at use sites

		case svsl_ir_param: // copy the attributed input into its hoisted local
			for (int32_t f = 0; f < e->in_io->count; f++) {
				const io_field_t *fld = &e->in_io->fields[f];
				if (fld->param != (int32_t)in->args[0]) continue;
				if (fld->member < 0) wln(e, "_%d = %s;", i, io_read(e, fld));
				else                 wln(e, "_%d.%s = %s;", i, ident(e, fld->name), io_read(e, fld));
			}
			break;

		case svsl_ir_load: {
			svsl_type_id_t pt;
			const char *path = lvalue(e, in->args[0], &pt);
			ex = ptr_is_atomic(e, in->args[0]) // atomic-retyped leaves have no plain loads
			   ? sfmt(e, "atomicLoad(&%s)", path) : path;
			break;
		}
		case svsl_ir_store: {
			svsl_type_id_t pt;
			const char *path = lvalue(e, in->args[0], &pt);
			if (ptr_is_atomic(e, in->args[0]))
				wln(e, "atomicStore(&%s, %s);", path, val(e, in->args[1]));
			else
				wln(e, "%s = %s;", path, val(e, in->args[1]));
			break;
		}

		case svsl_ir_atomic: { // args[3] low byte is the op, in svsl_intr_atomic_ order
			static const char *ops[8] = { "atomicAdd", "atomicSub", "atomicMin", "atomicMax",
			                              "atomicAnd", "atomicOr", "atomicXor", "atomicExchange" };
			svsl_type_id_t pt;
			const char    *path    = lvalue(e, in->args[0], &pt);
			uint32_t       op_code = in->args[3] & 0xFF;
			if (op_code < 8) {
				wln(e, "let _%d = %s(&%s, %s);", i, ops[op_code], path, val(e, in->args[1]));
				break;
			}
			// compare-exchange: WGSL only has the weak form, which may fail
			// spuriously — retry until it either succeeds or genuinely mismatches
			wln(e, "var _%d : %s;", i, type_name_w(e, in->type, in->loc));
			wln(e, "loop {");
			e->indent++;
			wln(e, "let _r%d = atomicCompareExchangeWeak(&%s, %s, %s);", i, path,
			    val(e, in->args[2]), val(e, in->args[1]));
			wln(e, "if (_r%d.exchanged || _r%d.old_value != %s) { _%d = _r%d.old_value; break; }",
			    i, i, val(e, in->args[2]), i, i);
			e->indent--;
			wln(e, "}");
			break;
		}

		case svsl_ir_construct: {
			const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
			ex = t->kind == svsl_type_struct
			   ? sfmt(e, "%s(%s)", struct_name_w(e, t->struct_index), aux_args(e, in))
			   : sfmt(e, "%s(%s)", type_name_w(e, in->type, in->loc), aux_args(e, in));
			break;
		}
		case svsl_ir_extract: {
			const svsl_ir_inst_t *src = &fn->insts.items[in->args[0]];
			const svsl_type_t    *st  = svsl_type_get(&e->prog->types, src->type);
			ex = st->kind == svsl_type_struct
			   ? sfmt(e, "%s.%s", val(e, in->args[0]),
			          struct_member_name(e, st->struct_index, (int32_t)in->args[1]))
			   : sfmt(e, "%s[%u]", val(e, in->args[0]), in->args[1]);
			break;
		}
		case svsl_ir_insert: {
			const svsl_ir_inst_t *src = &fn->insts.items[in->args[0]];
			const svsl_type_t    *st  = svsl_type_get(&e->prog->types, src->type);
			wln(e, "var _%d = %s;", i, val(e, in->args[0]));
			if (st->kind == svsl_type_struct)
				wln(e, "_%d.%s = %s;", i,
				    struct_member_name(e, st->struct_index, (int32_t)in->args[1]),
				    val(e, in->args[2]));
			else
				wln(e, "_%d[%u] = %s;", i, in->args[1], val(e, in->args[2]));
			break;
		}
		case svsl_ir_shuffle: {
			static const char comp[4] = { 'x', 'y', 'z', 'w' };
			char sw_str[5] = {0};
			for (uint32_t c = 0; c < in->args[2] && c < 4; c++)
				sw_str[c] = comp[(in->args[1] >> (c * 4)) & 0xF];
			ex = sfmt(e, "%s.%s", val(e, in->args[0]), sw_str);
			break;
		}
		case svsl_ir_extract_dynamic:
			ex = sfmt(e, "%s[%s]", val(e, in->args[0]), val(e, in->args[1]));
			break;

		case svsl_ir_neg: {
			const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
			ex = t->scalar == svsl_scalar_uint32 // WGSL has no unary minus on u32
			   ? sfmt(e, "(%s - %s)", splat(e, in->type, "0", in->loc), val(e, in->args[0]))
			   : sfmt(e, "-(%s)", val(e, in->args[0]));
			break;
		}
		case svsl_ir_bit_not:
			ex = sfmt(e, "~(%s)", val(e, in->args[0])); break;
		case svsl_ir_log_not:
			ex = sfmt(e, "!(%s)", val(e, in->args[0])); break;

		case svsl_ir_shl: case svsl_ir_shr: {
			const svsl_ir_inst_t *lhs = &fn->insts.items[in->args[0]];
			ex = sfmt(e, "(%s %s %s)", val(e, in->args[0]), binop_token(op),
			          shift_rhs(e, in->args[1], lhs->type));
			break;
		}
		case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul: case svsl_ir_div:
		case svsl_ir_rem: case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
		case svsl_ir_eq: case svsl_ir_ne: case svsl_ir_lt: case svsl_ir_le:
		case svsl_ir_gt: case svsl_ir_ge: case svsl_ir_log_and: case svsl_ir_log_or: {
			// division by a constant zero (the 0.0/0.0 NaN idiom) const-evaluates
			// to a shader-creation error; a var makes the divisor runtime instead
			if (op == svsl_ir_div || op == svsl_ir_rem) {
				const svsl_ir_inst_t *rhs = &fn->insts.items[in->args[1]];
				const svsl_type_t    *rt  = rhs->op == svsl_ir_const
				                          ? svsl_type_get(&e->prog->types, rhs->type) : NULL;
				bool zero = rt && (rt->scalar == svsl_scalar_float32 || rt->scalar == svsl_scalar_half
				                   ? (rhs->args[0] & 0x7FFFFFFFu) == 0 : rhs->args[0] == 0);
				if (zero) {
					wln(e, "var _dz%d : %s = %s;", i, type_name_w(e, rhs->type, in->loc), val(e, in->args[1]));
					ex = sfmt(e, "(%s %s _dz%d)", a, binop_token(op), i);
					break;
				}
			}
			ex = sfmt(e, "(%s %s %s)", a, binop_token(op), val(e, in->args[1]));
			break;
		}

		case svsl_ir_select: {
			const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
			if (t->kind == svsl_type_scalar || t->kind == svsl_type_vector) {
				ex = sfmt(e, "select(%s, %s, %s)", val(e, in->args[2]),
				          val(e, in->args[1]), val(e, in->args[0]));
			} else { // WGSL select only takes scalars/vectors
				wln(e, "var _%d : %s;", i, type_name_w(e, in->type, in->loc));
				wln(e, "if (%s) { _%d = %s; } else { _%d = %s; }", val(e, in->args[0]),
				    i, val(e, in->args[1]), i, val(e, in->args[2]));
			}
			break;
		}
		case svsl_ir_convert:
			ex = sfmt(e, "%s(%s)", type_name_w(e, in->type, in->loc), val(e, in->args[0]));
			break;
		case svsl_ir_mat_mul: // swapped operands, mirroring the SPIR-V backend
			ex = sfmt(e, "(%s * %s)", val(e, in->args[1]), val(e, in->args[0]));
			break;

		case svsl_ir_intrinsic:
			ex = intrinsic_expr(e, in); // NULL when it emitted a whole statement
			break;
		case svsl_ir_tex:
			ex = tex_expr(e, in);
			break;
		case svsl_ir_image_load: { // arrayed images split the layer off the coord
			const svsl_type_t *it = svsl_type_get(&e->prog->types,
			                                      e->prog->resources.items[in->args[0]].type);
			const char *c = val(e, in->args[1]);
			ex = shrink4(e, it->arrayed
			   ? sfmt(e, "textureLoad(%s, %s.xy, %s.z)", res_name(e, (int32_t)in->args[0]), c, c)
			   : sfmt(e, "textureLoad(%s, %s)", res_name(e, (int32_t)in->args[0]), c), in->type);
			break;
		}
		case svsl_ir_image_store: {
			const svsl_type_t *it = svsl_type_get(&e->prog->types,
			                                      e->prog->resources.items[in->args[0]].type);
			const char *c = val(e, in->args[1]);
			if (it->arrayed)
				wln(e, "textureStore(%s, %s.xy, %s.z, %s);", res_name(e, (int32_t)in->args[0]),
				    c, c, expand4(e, in->args[2], in->loc));
			else
				wln(e, "textureStore(%s, %s, %s);", res_name(e, (int32_t)in->args[0]),
				    c, expand4(e, in->args[2], in->loc));
			break;
		}

		case svsl_ir_bitfield_extract:
			ex = sfmt(e, "extractBits(%s, u32(%s), u32(%s))",
			          val(e, in->args[0]), val(e, in->args[1]), val(e, in->args[2]));
			break;
		case svsl_ir_bitfield_insert:
			ex = sfmt(e, "insertBits(%s, %s, u32(%s), u32(%s))",
			          val(e, in->args[0]), val(e, in->args[1]), val(e, in->args[2]), val(e, in->args[3]));
			break;

		// -------- structured control flow --------
		case svsl_ir_if:
			wln(e, "if (%s) {", val(e, in->args[0])); e->indent++; break;
		case svsl_ir_else:
			e->indent--; wln(e, "} else {"); e->indent++; break;
		case svsl_ir_end_if:
			e->indent--; wln(e, "}"); break;
		case svsl_ir_loop:
			if (loop_depth >= 32) { skip(e, in->loc, "loops nested deeper than 32"); break; }
			loop_cont[loop_depth++] = false;
			wln(e, "loop {"); e->indent++;
			break;
		case svsl_ir_loop_continue:
			if (loop_depth > 0) loop_cont[loop_depth - 1] = true;
			wln(e, "continuing {"); e->indent++;
			break;
		case svsl_ir_end_loop:
			if (loop_depth > 0 && loop_cont[--loop_depth]) {
				// a do-while's condition rides on end_loop: the continuing block
				// takes a conditional back-edge, which is WGSL's `break if`
				if (in->args[0] != SVSL_IR_NONE)
					wln(e, "break if !(%s);", val(e, in->args[0]));
				e->indent--; wln(e, "}");
			}
			e->indent--; wln(e, "}");
			break;
		case svsl_ir_break:    wln(e, "break;");    break;
		case svsl_ir_continue: wln(e, "continue;"); break;

		case svsl_ir_switch:
			if (sw_depth >= 8) { skip(e, in->loc, "switches nested deeper than 8"); break; }
			sw[sw_depth].inst         = in;
			sw[sw_depth].open_case    = false;
			sw[sw_depth].seen_default = false;
			sw_depth++;
			wln(e, "switch %s {", val(e, in->args[0]));
			e->indent++;
			break;
		case svsl_ir_case: {
			if (sw_depth == 0) break;
			if (sw[sw_depth - 1].open_case) { e->indent--; wln(e, "}"); }
			const svsl_ir_inst_t *sw_inst = sw[sw_depth - 1].inst;
			const svsl_type_t    *st = svsl_type_get(&e->prog->types,
			                                         fn->insts.items[sw_inst->args[0]].type);
			// merge selectors of directly adjacent case markers into one clause
			const char *sel = NULL;
			int32_t     j   = i;
			for (;;) {
				const svsl_ir_inst_t *c = &fn->insts.items[j];
				if (c->args[1]) { sw[sw_depth - 1].seen_default = true; sel = sel ? sfmt(e, "%s, default", sel) : "default"; }
				else {
					uint32_t    lit = fn->aux.items[sw_inst->aux + c->args[0]];
					const char *l   = st->scalar == svsl_scalar_uint32 ? sfmt(e, "%uu", lit)
					                                                   : sfmt(e, "%di", (int32_t)lit);
					sel = sel ? sfmt(e, "%s, %s", sel, l) : l;
				}
				if (j + 1 < fn->insts.count && fn->insts.items[j + 1].op == svsl_ir_case) j++;
				else break;
			}
			i = j; // consumed the merged markers
			wln(e, "case %s: {", sel);
			e->indent++;
			sw[sw_depth - 1].open_case = true;
			break;
		}
		case svsl_ir_end_switch:
			if (sw_depth == 0) break;
			if (sw[sw_depth - 1].open_case) { e->indent--; wln(e, "}"); }
			if (!sw[sw_depth - 1].seen_default) wln(e, "default: {}");
			sw_depth--;
			e->indent--;
			wln(e, "}");
			break;

		case svsl_ir_return:
			emit_return(e, in, out_io, out_struct);
			break;
		case svsl_ir_discard:
		case svsl_ir_demote: // WGSL discard has demote semantics
			wln(e, "discard;");
			break;

		default:
			skip(e, in->loc, "an IR operation the WGSL backend doesn't handle yet (op %d)", op);
			break;
		}

		if (ex) {
			if (e->inline_ok[i]) e->inline_expr[i] = ex;
			else                 wln(e, "let _%d = %s;", i, ex);
		}
	}
}

// ---- module-level declarations ---------------------------------------------------

static void emit_structs(wgsl_t *e) {
	for (int32_t s = 0; s < e->prog->types.structs.count; s++) {
		if (!e->struct_used[s]) continue;
		const svsl_struct_info_t *info = &e->prog->types.structs.items[s];
		wln(e, "struct %s {", struct_name_w(e, s));
		e->indent++;
		for (int32_t m = 0; m < info->members.count; m++)
			wln(e, "%s : %s,", struct_member_name(e, s, m),
			    type_name_w(e, info->members.items[m].type, info->members.items[m].loc));
		e->indent--;
		wln(e, "}");
		wln(e, "");
	}
}

// scalar / array-of-scalar with the leaf wrapped in atomic<T>
static const char *atomic_type_name(wgsl_t *e, svsl_type_id_t id, svsl_loc_t loc) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, id);
	if (t->kind == svsl_type_array)
		return sfmt(e, "array<%s%s>", atomic_type_name(e, t->elem, loc),
		            t->array_count > 0 ? sfmt(e, ", %d", t->array_count) : "");
	return sfmt(e, "atomic<%s>", scalar_name(e, t->scalar, loc));
}

// opt_atomic: per-member marks from the prescan (NULL for uniform buffers)
static void emit_buffer_struct(wgsl_t *e, const svsl_buffer_t *buf, const uint8_t *opt_atomic) {
	wln(e, "struct %s_t {", ident(e, buf->name));
	e->indent++;
	for (int32_t m = 0; m < buf->members.count; m++)
		wln(e, "%s : %s,", ident(e, buf->members.items[m].name),
		    opt_atomic && opt_atomic[m]
		        ? atomic_type_name(e, buf->members.items[m].type, buf->members.items[m].loc)
		        : type_name_w(e, buf->members.items[m].type, buf->members.items[m].loc));
	e->indent--;
	wln(e, "}");
}

// true when this pointer instruction lands on an atomic-retyped leaf, so plain
// accesses must go through atomicLoad/atomicStore
static bool ptr_is_atomic(wgsl_t *e, uint32_t id) {
	while (e->fn->insts.items[id].op == svsl_ir_chain) id = e->fn->insts.items[id].args[0];
	const svsl_ir_inst_t *root = &e->fn->insts.items[id];
	if (root->op != svsl_ir_ptr) return false;
	switch ((svsl_ref_)root->args[0]) {
	case svsl_ref_resource:      return e->res_atomic[root->args[1]] != 0;
	case svsl_ref_buffer_member: return e->buf_atomic[root->args[1]][root->args[2]] != 0;
	case svsl_ref_workgroup:     return e->wg_atomic[root->args[1]] != 0;
	default:                     return false;
	}
}

static void add_sampler(wgsl_t *e, const char *name, uint16_t slot, uint16_t paired) {
	for (int32_t i = 0; i < e->samplers.count; i++)
		if (strcmp(e->samplers.items[i].name.ptr, name) == 0) return;
	svsl_array_push(e->arena, &e->samplers,
	                (svsl_wgsl_sampler_t){ .name = svsl_str(name), .slot = slot, .paired_slot = paired });
}

static void emit_globals(wgsl_t *e) {
	const svsl_program_t *prog = e->prog;

	// specialization constants → pipeline-overridable constants
	for (int32_t i = 0; i < prog->spec_consts.count; i++) {
		const svsl_spec_const_t *sc = &prog->spec_consts.items[i];
		const svsl_type_t       *t  = svsl_type_get(&prog->types, sc->type);
		const char *value =
			t->scalar == svsl_scalar_bool    ? (sc->default_bits ? "true" : "false") :
			t->scalar == svsl_scalar_int32   ? sfmt(e, "%di", (int32_t)sc->default_bits) :
			t->scalar == svsl_scalar_uint32  ? sfmt(e, "%uu", sc->default_bits) :
			                                   sfmt(e, "%.9gf", bits_to_f32(sc->default_bits));
		wln(e, "@id(%u) override %s : %s = %s;", sc->id, ident(e, sc->name),
		    type_name_w(e, sc->type, sc->loc), value);
	}
	if (e->uses_view_index)
		wln(e, "@id(%d) override sk_view_index : u32 = 0u;", WGSL_VIEW_INDEX_SPEC_ID);
	if (prog->spec_consts.count > 0 || e->uses_view_index) wln(e, "");

	// const globals: materialized initializers (prescan), int-folded ones direct
	for (int32_t i = 0; i < prog->const_globals.count; i++) {
		const svsl_global_t *g = &prog->const_globals.items[i];
		if (e->const_texts[i])
			wln(e, "const %s : %s = %s;", ident(e, g->name), type_name_w(e, g->type, g->var->loc),
			    e->const_texts[i]);
		else if (g->has_int)
			wln(e, "const %s : %s = %s(%lld);", ident(e, g->name), type_name_w(e, g->type, g->var->loc),
			    type_name_w(e, g->type, g->var->loc), (long long)g->int_value);
	}

	for (int32_t i = 0; i < prog->workgroup_vars.count; i++)
		wln(e, "var<workgroup> %s : %s;", ident(e, prog->workgroup_vars.items[i].name),
		    e->wg_atomic[i]
		        ? atomic_type_name(e, prog->workgroup_vars.items[i].type, prog->workgroup_vars.items[i].var->loc)
		        : type_name_w(e, prog->workgroup_vars.items[i].type, prog->workgroup_vars.items[i].var->loc));

	// uniform buffers
	for (int32_t b = 0; b < prog->buffers.count; b++) {
		const svsl_buffer_t *buf = &prog->buffers.items[b];
		if (!e->buffer_used[b] || buf->kind != svsl_block_uniform) continue;
		emit_buffer_struct(e, buf, NULL);
		wln(e, "@group(0) @binding(%d) var<uniform> %s : %s_t;",
		    buf->bind.slot, ident(e, buf->name), ident(e, buf->name));
		wln(e, "");
	}

	// resources
	for (int32_t r = 0; r < prog->resources.count; r++) {
		if (!e->res_used[r]) continue;
		const svsl_resource_t *res = &prog->resources.items[r];
		const svsl_type_t     *t   = svsl_type_get(&prog->types, res->type);
		const char            *n   = res_name(e, r);
		switch (res->kind) {
		case svsl_res_texture: {
			const char *sc  = scalar_name(e, svsl_type_get(&prog->types, t->elem)->scalar, res->loc);
			const char *dim = t->dim == svsl_texdim_1d ? "1d" : t->dim == svsl_texdim_3d ? "3d" :
			                  t->dim == svsl_texdim_cube ? "cube" : "2d";
			const char *ty;
			if (tex_is_depth(e, r))
				ty = t->multisampled ? "texture_depth_multisampled_2d"
				   : sfmt(e, "texture_depth_%s%s", dim, t->arrayed ? "_array" : "");
			else if (t->multisampled)
				ty = sfmt(e, "texture_multisampled_2d<%s>", sc);
			else
				ty = sfmt(e, "texture_%s%s<%s>", dim, t->arrayed ? "_array" : "", sc);
			wln(e, "@group(0) @binding(%d) var %s : %s;", SLOT_TEXTURE + res->bind.slot, n, ty);
			if (res->sampler_slot >= 0) { // paired sampler splits off to s+400
				const char *sn = paired_sampler_name(e, r);
				wln(e, "@group(0) @binding(%d) var %s : %s;", SLOT_SAMPLER + res->bind.slot, sn,
				    tex_is_depth(e, r) ? "sampler_comparison" : "sampler");
				add_sampler(e, sn, (uint16_t)(SLOT_SAMPLER + res->bind.slot),
				            (uint16_t)(SLOT_TEXTURE + res->bind.slot));
			}
			break;
		}
		case svsl_res_sampler: {
			if (sampler_is_paired(e, r)) break; // binds through its texture's declaration
			int32_t pair = e->sampler_pair[r];
			wln(e, "@group(0) @binding(%d) var %s : %s;", SLOT_SAMPLER + res->bind.slot, n,
			    t->is_comparison ? "sampler_comparison" : "sampler");
			add_sampler(e, n, (uint16_t)(SLOT_SAMPLER + res->bind.slot),
			            pair >= 0 ? (uint16_t)(SLOT_TEXTURE + prog->resources.items[pair].bind.slot) : 0xFFFF);
			break;
		}
		case svsl_res_structured:
		case svsl_res_rw_structured: {
			bool        rw     = res->kind == svsl_res_rw_structured;
			int32_t     slot   = (rw ? SLOT_READWRITE : SLOT_TEXTURE) + res->bind.slot;
			const char *access = rw ? "read_write" : "read";
			if (res->buffer_index >= 0) { // block form: struct with the block's members
				emit_buffer_struct(e, &prog->buffers.items[res->buffer_index],
				                   e->buf_atomic[res->buffer_index]);
				wln(e, "@group(0) @binding(%d) var<storage, %s> %s : %s_t;", slot, access, n,
				    ident(e, prog->buffers.items[res->buffer_index].name));
			} else if (e->res_atomic[r]) {
				wln(e, "@group(0) @binding(%d) var<storage, %s> %s : array<%s>;", slot, access, n,
				    atomic_type_name(e, t->elem, res->loc));
			} else {
				wln(e, "@group(0) @binding(%d) var<storage, %s> %s : array<%s>;", slot, access, n,
				    type_name_w(e, t->elem, res->loc));
			}
			break;
		}
		case svsl_res_image: {
			uint32_t    fmt    = svsl_image_format_for(&prog->types, t);
			const char *dim    = t->dim == svsl_texdim_1d ? "1d" : t->dim == svsl_texdim_3d ? "3d" : "2d";
			const char *access = e->res_access[r] == 3 ? "read_write" :
			                     e->res_access[r] == 1 ? "read" : "write";
			if (t->arrayed && t->dim != svsl_texdim_2d) {
				skip(e, res->loc, "WGSL storage textures only array in 2D");
				break;
			}
			wln(e, "@group(0) @binding(%d) var %s : texture_storage_%s%s<%s, %s>;",
			    SLOT_READWRITE + res->bind.slot, n, dim, t->arrayed ? "_array" : "",
			    wgsl_texel_format(fmt), access);
			break;
		}
		case svsl_res_subpass: { // lowered to a plain sampled texture (same meta slot)
			const char *sc = scalar_name(e, svsl_type_get(&prog->types, t->elem)->scalar, res->loc);
			wln(e, "@group(0) @binding(%d) var %s : %s;", SLOT_TEXTURE + res->bind.slot, n,
			    t->multisampled ? sfmt(e, "texture_multisampled_2d<%s>", sc)
			                    : sfmt(e, "texture_2d<%s>", sc));
			break;
		}
		default: break; // tileimage already skipped in prescan
		}
	}
	wln(e, "");
}

// ---- public entry ----------------------------------------------------------------

bool svsl_wgsl_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                    const svsl_ir_func_t *fn, svsl_wgsl_blob_t *out_blob,
                    svsl_diag_list_t *ref_diags) {
	*out_blob = (svsl_wgsl_blob_t){0};
	wgsl_t e = { .arena = arena, .prog = prog, .fn = fn, .diags = ref_diags };
	int32_t res_n    = prog->resources.count      > 0 ? prog->resources.count      : 1;
	int32_t buf_n    = prog->buffers.count        > 0 ? prog->buffers.count        : 1;
	int32_t struct_n = prog->types.structs.count  > 0 ? prog->types.structs.count  : 1;
	e.buffer_used  = svsl_arena_alloc(arena, (size_t)buf_n);
	e.res_used     = svsl_arena_alloc(arena, (size_t)res_n);
	e.res_cmp      = svsl_arena_alloc(arena, (size_t)res_n);
	e.res_access   = svsl_arena_alloc(arena, (size_t)res_n);
	e.struct_used  = svsl_arena_alloc(arena, (size_t)struct_n);
	e.sampler_pair = svsl_arena_alloc(arena, (size_t)res_n * sizeof(int32_t));
	for (int32_t i = 0; i < res_n; i++) e.sampler_pair[i] = -1;
	int32_t cg_n  = prog->const_globals.count > 0 ? prog->const_globals.count : 1;
	e.const_texts = svsl_arena_alloc(arena, (size_t)cg_n * sizeof(const char *));
	int32_t wg_n  = prog->workgroup_vars.count > 0 ? prog->workgroup_vars.count : 1;
	e.res_atomic  = svsl_arena_alloc(arena, (size_t)res_n);
	e.wg_atomic   = svsl_arena_alloc(arena, (size_t)wg_n);
	e.buf_atomic  = svsl_arena_alloc(arena, (size_t)buf_n * sizeof(uint8_t *));
	for (int32_t i = 0; i < prog->buffers.count; i++)
		e.buf_atomic[i] = svsl_arena_alloc(arena,
			(size_t)(prog->buffers.items[i].members.count > 0 ? prog->buffers.items[i].members.count : 1));

	int32_t errors_before = ref_diags->error_count;
	prescan(&e);
	if (ref_diags->error_count > errors_before) return false;
	if (e.skipped) return true;

	// entry IO shapes decide the signature and the param copy-ins
	io_list_t in_io = {0}, out_io = {0};
	svsl_stage_ stage = fn->entry->stage;
	io_collect_params(&e, &in_io, stage == svsl_stage_vertex ? svsl_sem_vs_in :
	                              stage == svsl_stage_pixel  ? svsl_sem_ps_in : svsl_sem_cs_in);
	if (stage != svsl_stage_compute)
		io_collect_return(&e, &out_io, stage == svsl_stage_vertex ? svsl_sem_vs_out : svsl_sem_ps_out);
	e.in_io = &in_io;
	if (e.skipped) return true;

	// subpass reads need this fragment's pixel position: reuse the shader's own
	// SV_Position input, or synthesize one (param -2: never copied into a local)
	if (e.uses_subpass) {
		const io_field_t *pos = NULL;
		for (int32_t i = 0; i < in_io.count; i++)
			if (in_io.fields[i].builtin && strcmp(in_io.fields[i].builtin, "position") == 0)
				pos = &in_io.fields[i];
		if (!pos && in_io.count < 64) {
			io_field_t *f = &in_io.fields[in_io.count++];
			*f  = (io_field_t){ .name = svsl_str("sk_frag_pos"), .builtin = "position",
			                    .param = -2, .member = -1, .location = -1 };
			pos = f;
		}
		if (!pos) { skip(&e, fn->entry->func->loc, "no room for the synthesized position input"); return true; }
		e.frag_pos = sfmt(&e, "in.%s", ident(&e, pos->name));
	}

	const char *entry_name = ident(&e, fn->entry->name);
	const char *in_name    = sfmt(&e, "%s_wgsl_in", entry_name);
	const char *out_name   = sfmt(&e, "%s_wgsl_out", entry_name);

	if (e.uses_f16) wln(&e, "enable f16;");
	if (e.uses_derivatives && stage == svsl_stage_pixel)
		wln(&e, "diagnostic(off, derivative_uniformity);");

	emit_globals(&e);
	emit_structs(&e);
	emit_io_struct(&e, in_name, &in_io, stage == svsl_stage_pixel);
	bool struct_out = out_io.count > 1 || (out_io.count == 1 && out_io.fields[0].member >= 0);
	if (struct_out) emit_io_struct(&e, out_name, &out_io, stage == svsl_stage_vertex);

	// signature
	bool has_in = false;
	for (int32_t i = 0; i < in_io.count; i++)
		if (!in_io.fields[i].view_index) has_in = true;
	const char *stage_attr =
		stage == svsl_stage_vertex ? "@vertex" :
		stage == svsl_stage_pixel  ? "@fragment" :
		sfmt(&e, "@compute @workgroup_size(%d, %d, %d)", fn->entry->workgroup[0],
		     fn->entry->workgroup[1], fn->entry->workgroup[2]);
	const char *ret = "";
	if (out_io.count == 1 && !struct_out) {
		const io_field_t *f = &out_io.fields[0];
		ret = f->builtin ? sfmt(&e, " -> @builtin(%s) %s", f->builtin, type_name_w(&e, f->type, (svsl_loc_t){0}))
		                 : sfmt(&e, " -> @location(%d) %s", f->location, type_name_w(&e, f->type, (svsl_loc_t){0}));
	} else if (struct_out) {
		ret = sfmt(&e, " -> %s", out_name);
	}
	wln(&e, "%s fn %s(%s)%s {", stage_attr, entry_name,
	    has_in ? sfmt(&e, "in : %s", in_name) : "", ret);
	e.indent++;
	emit_body(&e, &out_io, out_name);
	e.indent--;
	wln(&e, "}");

	if (ref_diags->error_count > errors_before) return false;
	if (e.skipped) return true;

	svsl_array_push(arena, &e.out, '\0');
	out_blob->text          = e.out.items;
	out_blob->length        = e.out.count - 1;
	out_blob->samplers      = e.samplers.items;
	out_blob->sampler_count = e.samplers.count;
	return true;
}
