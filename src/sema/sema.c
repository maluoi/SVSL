#include "sema.h"
#include "check.h"

#include "../tables/formats.h"
#include "../tables/semantics.h"
#include "../tables/types_builtin.h"
#include "../util/array.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct sema_t {
	svsl_arena_t     *arena;
	svsl_program_t   *prog;
	svsl_diag_list_t *diags;
	const svsl_ast_t *ast;
} sema_t;

static void err(sema_t *s, svsl_loc_t loc, const char *fmt, svsl_str_t arg) {
	svsl_diag_add(s->arena, s->diags, svsl_severity_error, loc, fmt, arg.len, arg.ptr);
}

// --- attribute validation ---------------------------------------------------------

// every attribute the compiler understands, with where it applies. The HLSL
// optimizer hints are accepted and deliberately not acted on (like glslang).
static const struct { const char *name; uint8_t ctx; } attr_table[] = {
	{ "numthreads",                  svsl_attr_ctx_func },
	{ "compute",                     svsl_attr_ctx_func },
	{ "vertex",                      svsl_attr_ctx_func },
	{ "fragment",                    svsl_attr_ctx_func },
	{ "pixel",                       svsl_attr_ctx_func },
	{ "wave_size",                   svsl_attr_ctx_func },
	{ "early_depth_stencil",         svsl_attr_ctx_func },
	{ "earlydepthstencil",           svsl_attr_ctx_func },
	{ "tile_shading_rate_qcom",      svsl_attr_ctx_func },
	{ "non_coherent_tile_reads_qcom",svsl_attr_ctx_func },
	{ "tile_attachment",             svsl_attr_ctx_var },
	{ "specialization",              svsl_attr_ctx_var },
	{ "vk::constant_id",             svsl_attr_ctx_var },
	{ "image_format",                svsl_attr_ctx_var },
	{ "vk::image_format",            svsl_attr_ctx_var },
	{ "input_attachment_index",      svsl_attr_ctx_var },
	{ "vk::input_attachment_index",  svsl_attr_ctx_var },
	{ "vk::binding",                 svsl_attr_ctx_var | svsl_attr_ctx_block },
	{ "vk::push_constant",           svsl_attr_ctx_block },
	{ "location",                    svsl_attr_ctx_member },
	{ "vk::location",                svsl_attr_ctx_member },
	{ "offset",                      svsl_attr_ctx_member },
	{ "vk::offset",                  svsl_attr_ctx_member },
	{ "unroll",  svsl_attr_ctx_stmt }, { "loop",    svsl_attr_ctx_stmt },
	{ "branch",  svsl_attr_ctx_stmt }, { "flatten", svsl_attr_ctx_stmt },
	{ "fastopt", svsl_attr_ctx_stmt },
};

void svsl_attrs_check(svsl_arena_t *arena, svsl_diag_list_t *diags,
                      const svsl_ast_attrs_t *attrs, svsl_attr_ctx_ ctx, bool porting) {
	for (int32_t i = 0; i < attrs->count; i++) {
		const svsl_ast_attr_t *attr  = &attrs->items[i];
		uint8_t                known = 0;
		for (int32_t k = 0; k < (int32_t)(sizeof(attr_table) / sizeof(attr_table[0])); k++)
			if (svsl_str_eq_cstr(attr->name, attr_table[k].name)) { known = attr_table[k].ctx; break; }

		if (known & ctx) {
			// prefer-native hint for the [[vk::*]] escape spellings
			if (porting && svsl_str_starts_with(attr->name, "vk::"))
				svsl_diag_add(arena, diags, svsl_severity_porting, attr->loc,
				              "legacy [[%.*s]] -> spec §6", attr->name.len, attr->name.ptr);
			continue;
		}
		if (known)
			svsl_diag_add(arena, diags, svsl_severity_warning, attr->loc,
			              "attribute '%.*s' does not apply here and is ignored",
			              attr->name.len, attr->name.ptr);
		else if (svsl_str_starts_with(attr->name, "vk::"))
			svsl_diag_add(arena, diags, svsl_severity_error, attr->loc,
			              "unknown attribute '[[%.*s]]'", attr->name.len, attr->name.ptr);
		else
			svsl_diag_add(arena, diags, svsl_severity_warning, attr->loc,
			              "unknown attribute '%.*s' is ignored", attr->name.len, attr->name.ptr);
	}
}

// --- type resolution ------------------------------------------------------------

static int32_t struct_find(const svsl_program_t *prog, svsl_str_t name) {
	for (int32_t i = 0; i < prog->types.structs.count; i++)
		if (svsl_str_eq(prog->types.structs.items[i].name, name)) return i;
	return -1;
}

static svsl_texdim_ texdim_from_name(svsl_str_t name, bool *out_arrayed, bool *out_ms) {
	*out_arrayed = false;
	*out_ms      = false;
	if (svsl_str_find_char(name, 'A') >= 0 && // ...Array suffix
	    name.len > 5 && memcmp(name.ptr + name.len - 5, "Array", 5) == 0) {
		*out_arrayed = true;
		name = svsl_str_slice(name, 0, name.len - 5);
	}
	if (name.len > 2 && memcmp(name.ptr + name.len - 2, "MS", 2) == 0) { // ...2DMS infix
		*out_ms = true;
		name = svsl_str_slice(name, 0, name.len - 2);
	}
	if (name.len >= 4 && memcmp(name.ptr + name.len - 4, "Cube", 4) == 0) return svsl_texdim_cube;
	if (name.len >= 2 && memcmp(name.ptr + name.len - 2, "3D", 2) == 0)   return svsl_texdim_3d;
	if (name.len >= 2 && memcmp(name.ptr + name.len - 2, "1D", 2) == 0)   return svsl_texdim_1d;
	return svsl_texdim_2d;
}

// native SVSL spelling of a legacy HLSL type name, or NULL if the name is not legacy
static const char *legacy_type_native(svsl_str_t name) {
	if (svsl_str_starts_with(name, "min16float"))         return "half";
	if (svsl_str_eq_cstr(name, "SamplerState"))           return "Sampler";
	if (svsl_str_eq_cstr(name, "SamplerComparisonState")) return "SamplerComparison";
	if (svsl_str_eq_cstr(name, "StructuredBuffer"))       return "readonly storagebuffer";
	if (svsl_str_eq_cstr(name, "RWStructuredBuffer"))     return "storagebuffer";
	if (svsl_str_starts_with(name, "RWTexture"))          return "Image";
	return NULL;
}

// A legacy type name resolves on more than one path (a bare global is also a $Global
// member), so dedupe by location+message — one source spelling earns one hint.
static void porting_type_hint(sema_t *s, svsl_loc_t loc, svsl_str_t legacy, const char *native) {
	char msg[128];
	snprintf(msg, sizeof msg, "legacy '%.*s' -> '%s'", (int)legacy.len, legacy.ptr, native);
	for (int32_t i = 0; i < s->diags->count; i++) {
		const svsl_diag_t *d = &s->diags->items[i];
		if (d->severity == svsl_severity_porting && d->loc.line == loc.line &&
		    d->loc.col == loc.col && strcmp(d->message, msg) == 0) return;
	}
	svsl_diag_add(s->arena, s->diags, svsl_severity_porting, loc, "%s", msg);
}

// resolves an AST type reference (without array dims) to a type id
static svsl_type_id_t resolve_base_type(sema_t *s, const svsl_ast_type_t *ref) {
	svsl_types_t *types = &s->prog->types;
	svsl_str_t    name  = ref->name;

	if (s->prog->porting) {
		const char *native = legacy_type_native(name);
		if (native) porting_type_hint(s, ref->loc, name, native);
	}

	// an inline 'enum {...}' resolves to its underlying integer (registered separately
	// by register_enums); recover to int32 on a bad underlying, matching that pass
	if (ref->inline_enum) {
		svsl_type_id_t u = ref->inline_enum->underlying
		                 ? resolve_base_type(s, ref->inline_enum->underlying)
		                 : svsl_type_scalar_id(types, svsl_scalar_int32);
		const svsl_type_t *ut = svsl_type_get(types, u);
		bool ok = ut->kind == svsl_type_scalar &&
		          ut->scalar >= svsl_scalar_int8 && ut->scalar <= svsl_scalar_uint64;
		return ok ? u : svsl_type_scalar_id(types, svsl_scalar_int32);
	}

	if (svsl_str_eq_cstr(name, "void"))
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });

	svsl_scalar_ scalar;
	int32_t      rows, cols;
	if (svsl_scalar_name_parse(name, &scalar, &rows, &cols)) {
		if (cols > 0) return svsl_type_matrix_id(types, scalar, rows, cols);
		if (rows > 0) return svsl_type_vector_id(types, scalar, rows);
		return svsl_type_scalar_id(types, scalar);
	}

	int32_t struct_index = struct_find(s->prog, name);
	if (struct_index >= 0)
		return svsl_type_intern(types, (svsl_type_t){
			.kind = svsl_type_struct, .struct_index = struct_index });

	// a named enum resolves to its underlying integer type (enums are int aliases)
	for (int32_t i = 0; i < s->prog->enums.count; i++)
		if (svsl_str_eq(s->prog->enums.items[i].name, name))
			return s->prog->enums.items[i].underlying;

	// resource object types
	svsl_type_id_t elem = SVSL_TYPE_NONE;
	if (ref->elem) elem = resolve_base_type(s, ref->elem);

	if (svsl_str_starts_with(name, "Texture")) {
		// a MS sample-count template arg is a source annotation only (like HLSL/DXC,
		// the pipeline's sample count is what matters)
		bool         arrayed, ms;
		svsl_texdim_ dim = texdim_from_name(name, &arrayed, &ms);
		if (elem == SVSL_TYPE_NONE) elem = svsl_type_vector_id(types, svsl_scalar_float32, 4);
		return svsl_type_intern(types, (svsl_type_t){
			.kind = svsl_type_texture, .dim = dim, .arrayed = arrayed, .multisampled = ms, .elem = elem });
	}
	if (svsl_str_starts_with(name, "RWTexture") || svsl_str_starts_with(name, "Image")) {
		bool         arrayed, ms;
		svsl_texdim_ dim = texdim_from_name(name, &arrayed, &ms);
		(void)ms; // storage images are never multisampled
		if (elem == SVSL_TYPE_NONE) elem = svsl_type_vector_id(types, svsl_scalar_float32, 4);
		if (ref->format.len > 0 && !svsl_image_format_find(ref->format, NULL))
			err(s, ref->loc, "unrecognized image format '%.*s'", ref->format);
		return svsl_type_intern(types, (svsl_type_t){
			.kind = svsl_type_image, .dim = dim, .arrayed = arrayed, .elem = elem,
			.is_rw = true, .format = ref->format });
	}
	if (svsl_str_eq_cstr(name, "SamplerState") || svsl_str_eq_cstr(name, "Sampler"))
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_sampler });
	if (svsl_str_eq_cstr(name, "SamplerComparisonState") || svsl_str_eq_cstr(name, "SamplerComparison"))
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_sampler, .is_comparison = true });
	if (svsl_str_eq_cstr(name, "StructuredBuffer") || svsl_str_eq_cstr(name, "Buffer"))
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_buffer, .elem = elem });
	if (svsl_str_eq_cstr(name, "RWStructuredBuffer"))
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_buffer, .elem = elem, .is_rw = true });
	if (svsl_str_eq_cstr(name, "SubpassInput") || svsl_str_eq_cstr(name, "SubpassInputMS")) {
		if (elem == SVSL_TYPE_NONE) elem = svsl_type_vector_id(types, svsl_scalar_float32, 4);
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_subpass, .elem = elem,
		                                              .multisampled = name.len > 12 });
	}
	if (svsl_str_eq_cstr(name, "TileImage")) {
		if (elem == SVSL_TYPE_NONE) elem = svsl_type_vector_id(types, svsl_scalar_float32, 4);
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_tileimage, .elem = elem });
	}

	err(s, ref->loc, "unknown type '%.*s'", name);
	return SVSL_TYPE_NONE;
}

// constant expression → int (array sizes); very small on purpose
static bool const_eval_int(sema_t *s, const svsl_ast_expr_t *e, int64_t *out) {
	switch (e->kind) {
	case svsl_expr_int_lit:  *out = (int64_t)e->int_lit.value; return true;
	case svsl_expr_bool_lit: *out = e->bool_lit ? 1 : 0;       return true;
	case svsl_expr_ident: {
		for (int32_t i = 0; i < s->prog->spec_consts.count; i++)
			if (svsl_str_eq(s->prog->spec_consts.items[i].name, e->ident)) {
				*out = (int64_t)s->prog->spec_consts.items[i].default_bits;
				return true;
			}
		for (int32_t i = 0; i < s->prog->const_globals.count; i++)
			if (s->prog->const_globals.items[i].has_int &&
			    svsl_str_eq(s->prog->const_globals.items[i].name, e->ident)) {
				*out = s->prog->const_globals.items[i].int_value;
				return true;
			}
		for (int32_t i = 0; i < s->prog->enum_consts.count; i++)
			if (svsl_str_eq(s->prog->enum_consts.items[i].name, e->ident)) {
				*out = s->prog->enum_consts.items[i].value;
				return true;
			}
		return false;
	}
	case svsl_expr_unary: {
		int64_t v;
		if (!const_eval_int(s, e->unary.operand, &v)) return false;
		if      (e->unary.op == svsl_tok_minus) *out = -v;
		else if (e->unary.op == svsl_tok_plus)  *out = v;
		else if (e->unary.op == svsl_tok_tilde) *out = ~v;
		else return false;
		return true;
	}
	case svsl_expr_binary: {
		int64_t a, b;
		if (!const_eval_int(s, e->binary.lhs, &a) || !const_eval_int(s, e->binary.rhs, &b)) return false;
		switch (e->binary.op) {
		case svsl_tok_plus:    *out = a + b; return true;
		case svsl_tok_minus:   *out = a - b; return true;
		case svsl_tok_star:    *out = a * b; return true;
		case svsl_tok_slash:   if (b == 0 || (a == INT64_MIN && b == -1)) return false; *out = a / b; return true;
		case svsl_tok_percent: if (b == 0 || (a == INT64_MIN && b == -1)) return false; *out = a % b; return true;
		case svsl_tok_shl:     *out = (int64_t)((uint64_t)a << (b & 63)); return true; // unsigned: no signed-overflow UB
		case svsl_tok_shr:     *out = a >> (b & 63); return true;
		default: return false;
		}
	}
	default: return false;
	}
}

// [[attr(N)]] with exactly one constant-integer argument → *out (false otherwise)
static bool attr_int1(sema_t *s, const svsl_ast_attr_t *attr, int64_t *out) {
	return attr->arg_count == 1 && const_eval_int(s, attr->args[0], out);
}

// [[vk::binding(slot[, set])]] → a direct descriptor binding; false if the args
// aren't (constant[, constant]). The caller owns the error (it knows the name).
static bool attr_binding(sema_t *s, const svsl_ast_attr_t *attr, svsl_ast_reg_t *out) {
	int64_t slot = 0, set = 0;
	if (attr->arg_count < 1 || !const_eval_int(s, attr->args[0], &slot)) return false;
	if (attr->arg_count >= 2 && !const_eval_int(s, attr->args[1], &set)) return false;
	*out = (svsl_ast_reg_t){ .present = true, .direct = true,
	                         .slot = (int32_t)slot, .space = (int32_t)set };
	return true;
}

// full type including the declarator's array dimensions (outer-to-inner)
static svsl_type_id_t resolve_type(sema_t *s, const svsl_ast_type_t *ref) {
	svsl_type_id_t id = resolve_base_type(s, ref);
	if (id == SVSL_TYPE_NONE) return id;
	for (int32_t i = ref->array_dim_count - 1; i >= 0; i--) {
		int64_t count = 0;
		if (ref->array_dims[i]) {
			if (!const_eval_int(s, ref->array_dims[i], &count) || count <= 0) {
				err(s, ref->loc, "array size for '%.*s' is not a positive constant", ref->name);
				count = 1;
			}
		}
		id = svsl_type_array_id(&s->prog->types, id, (int32_t)count);
	}
	return id;
}

// Unsized arrays take their length from the initializer: float3 x[] = {...}.
// Only the outermost dimension may be unsized, and only with an init list —
// used for const globals and locals; runtime-sized buffer members keep their
// count-0 spelling and never come through here.
static svsl_type_id_t infer_array_size(sema_t *s, svsl_type_id_t type, const svsl_ast_var_t *var) {
	const svsl_type_t *t = svsl_type_get(&s->prog->types, type);
	if (t->kind != svsl_type_array) return type;

	svsl_type_id_t elem = t->elem;
	if (t->array_count == 0) {
		if (!var->init || var->init->kind != svsl_expr_init_list || var->init->init_list.count == 0) {
			err(s, var->loc, "unsized array '%.*s' needs an initializer list to infer its size", var->name);
			return type;
		}
		type = svsl_type_array_id(&s->prog->types, elem, var->init->init_list.count);
	}
	for (const svsl_type_t *in = svsl_type_get(&s->prog->types, elem);
	     in->kind == svsl_type_array; in = svsl_type_get(&s->prog->types, in->elem))
		if (in->array_count == 0) {
			err(s, var->loc, "only the outermost array dimension of '%.*s' may be unsized", var->name);
			break;
		}
	return type;
}

// --- structs -----------------------------------------------------------------------

// DFS over the containment graph; state: 0 = unvisited, 1 = visiting, 2 = done.
// Returns true when `index` is currently being visited, i.e. a member closes a
// cycle. The offending member is errored and re-typed so layout cannot recurse.
static bool struct_cycle_check(sema_t *s, int32_t index, uint8_t *state) {
	if (state[index] == 2) return false;
	if (state[index] == 1) return true;
	state[index] = 1;
	svsl_struct_info_t *info = &s->prog->types.structs.items[index];
	for (int32_t m = 0; m < info->members.count; m++) {
		svsl_member_t *member = &info->members.items[m];
		if (member->type == SVSL_TYPE_NONE) continue; // unresolved, already errored
		const svsl_type_t *t = svsl_type_get(&s->prog->types, member->type);
		while (t->kind == svsl_type_array) t = svsl_type_get(&s->prog->types, t->elem);
		if (t->kind != svsl_type_struct) continue;
		if (struct_cycle_check(s, t->struct_index, state)) {
			err(s, member->loc, "struct member '%.*s' contains the struct it belongs to", member->name);
			member->type = svsl_type_scalar_id(&s->prog->types, svsl_scalar_float32); // recover
		}
	}
	state[index] = 2;
	return false;
}

// packs a struct whose members carry bit widths into 32-bit backing words
// (dense, LSB-first, C-style padding so no field crosses a word). Fills
// info->fields (logical) and info->members (backing uint32 words).
static void pack_struct(sema_t *s, svsl_struct_info_t *info, const svsl_ast_struct_t *st) {
	info->packed = true;
	svsl_type_id_t uint32_id = svsl_type_scalar_id(&s->prog->types, svsl_scalar_uint32);
	int32_t cur = 0;
	for (int32_t m = 0; m < st->member_count; m++) {
		const svsl_ast_var_t *mv = st->members[m];
		svsl_type_id_t        rt = resolve_type(s, mv->type);
		const svsl_type_t    *t  = svsl_type_get(&s->prog->types, rt);
		if (t->kind != svsl_type_scalar) {
			err(s, mv->loc, "packed field '%.*s' must be a scalar", mv->name);
			continue;
		}
		svsl_scalar_ sc = t->scalar;
		bool is_bool  = sc == svsl_scalar_bool;
		bool is_int   = sc >= svsl_scalar_int8 && sc <= svsl_scalar_uint64;
		bool is_float = (sc >= svsl_scalar_float16 && sc <= svsl_scalar_float64) ||
		                sc == svsl_scalar_half;
		bool is_half  = sc == svsl_scalar_float16 || sc == svsl_scalar_half;
		uint8_t fmt   = mv->bit_format;
		int32_t width = mv->bit_width;

		// a plain member (no ': width') occupies its type's natural bit width, so a
		// packed struct can freely mix bit fields with whole uint8/uint16/half/etc.
		if (width < 0 && !is_bool) {
			int32_t nat = is_half ? 16 : svsl_scalar_size(sc) * 8;
			if (nat > 32) {
				err(s, mv->loc, "packed member '%.*s' is wider than 32 bits; give an explicit width", mv->name);
				continue;
			}
			width = nat;
		}

		if (is_bool) {
			if (width < 0) width = 1;
			if (fmt != svsl_bitfmt_raw || width != 1) {
				err(s, mv->loc, "bool packed field '%.*s' is exactly 1 raw bit", mv->name);
				continue;
			}
		} else if (fmt == svsl_bitfmt_unorm || fmt == svsl_bitfmt_snorm) {
			if (!is_float) { err(s, mv->loc, "unorm/snorm field '%.*s' needs a float/half type", mv->name); continue; }
			if (width < 1 || width > 24) {
				svsl_diag_add(s->arena, s->diags, svsl_severity_error, mv->loc,
				              "normalized field '%.*s' width %d out of range [1,24]", mv->name.len, mv->name.ptr, width);
				continue;
			}
		} else if (is_int) { // raw integer
			int32_t bits = svsl_scalar_size(sc) * 8;
			if (bits > 32) bits = 32;
			if (width < 1 || width > bits) {
				svsl_diag_add(s->arena, s->diags, svsl_severity_error, mv->loc,
				              "raw field '%.*s' width %d exceeds its %d-bit type", mv->name.len, mv->name.ptr, width, bits);
				continue;
			}
		} else if (is_half && width == 16) { // raw 16-bit half bit pattern
			// ok
		} else if (sc == svsl_scalar_float32 && width == 32) { // whole float, bitcast to its word
			// ok
		} else {
			err(s, mv->loc, "field '%.*s': raw float storage must be a whole half (16b) or float (32b); use unN/snN to sub-slice", mv->name);
			continue;
		}

		if ((cur % 32) + width > 32) cur = (cur + 31) & ~31; // C-style: don't cross a word
		svsl_array_push(s->arena, &info->fields, (svsl_field_t){
			.name = mv->name, .type = rt, .bit_offset = (int16_t)cur,
			.bit_width = (uint8_t)width, .bit_format = fmt, .loc = mv->loc });
		cur += width;
	}
	info->backing_words = cur > 0 ? (cur + 31) / 32 : 1;
	for (int32_t w = 0; w < info->backing_words; w++)
		svsl_array_push(s->arena, &info->members, (svsl_member_t){
			.name = svsl_str(""), .type = uint32_id, .explicit_offset = -1, .explicit_location = -1 });
}

// Registers one enum's type alias (if named) and its global integer constants.
static void register_one_enum(sema_t *s, const svsl_ast_enum_t *en) {
	svsl_type_id_t under = en->underlying ? resolve_type(s, en->underlying)
	                                      : svsl_type_scalar_id(&s->prog->types, svsl_scalar_int32);
	const svsl_type_t *ut = svsl_type_get(&s->prog->types, under);
	bool is_int = ut->kind == svsl_type_scalar &&
	              ut->scalar >= svsl_scalar_int8 && ut->scalar <= svsl_scalar_uint64;
	if (!is_int) {
		err(s, en->loc, "enum '%.*s' underlying type must be an integer scalar",
		    en->name.len ? en->name : svsl_str("<anonymous>"));
		under = svsl_type_scalar_id(&s->prog->types, svsl_scalar_int32);
	}

	if (en->name.len) {
		for (int32_t j = 0; j < s->prog->enums.count; j++)
			if (svsl_str_eq(s->prog->enums.items[j].name, en->name))
				err(s, en->loc, "duplicate enum '%.*s'", en->name);
		// struct/enum name clashes are caught in register_structs (structs
		// register after enums)
		svsl_array_push(s->arena, &s->prog->enums,
		                (svsl_enum_t){ .name = en->name, .underlying = under });
	}

	int64_t next = 0;
	for (int32_t k = 0; k < en->item_count; k++) {
		const svsl_ast_enum_item_t *it = &en->items[k];
		if (it->value && !const_eval_int(s, it->value, &next))
			err(s, it->loc, "enum constant '%.*s' is not a constant integer", it->name);
		for (int32_t j = 0; j < s->prog->enum_consts.count; j++)
			if (svsl_str_eq(s->prog->enum_consts.items[j].name, it->name)) {
				err(s, it->loc, "duplicate enum constant '%.*s'", it->name);
				break;
			}
		svsl_array_push(s->arena, &s->prog->enum_consts,
		                (svsl_enum_const_t){ .name = it->name, .type = under, .value = next });
		next++;
	}
}

// An 'enum {...}' can appear in any type position (params, struct members, locals);
// wherever it does, its constants still register globally. These walkers find them.
static void register_type_enums(sema_t *s, const svsl_ast_type_t *t) {
	if (!t) return;
	if (t->inline_enum) register_one_enum(s, t->inline_enum);
	register_type_enums(s, t->elem); // e.g. StructuredBuffer<enum {...}>
}

static void register_stmt_enums(sema_t *s, const svsl_ast_stmt_t *st) {
	if (!st) return;
	switch (st->kind) {
	case svsl_stmt_block:
		for (int32_t i = 0; i < st->block.count; i++) register_stmt_enums(s, st->block.stmts[i]);
		break;
	case svsl_stmt_var_decl:
		for (int32_t i = 0; i < st->var_decl.count; i++) register_type_enums(s, st->var_decl.vars[i]->type);
		break;
	case svsl_stmt_if:
		register_stmt_enums(s, st->if_stmt.then_stmt);
		register_stmt_enums(s, st->if_stmt.else_stmt);
		break;
	case svsl_stmt_for:
		register_stmt_enums(s, st->for_stmt.init);
		register_stmt_enums(s, st->for_stmt.body);
		break;
	case svsl_stmt_while:
	case svsl_stmt_do:
		register_stmt_enums(s, st->while_stmt.body);
		break;
	case svsl_stmt_switch:
		for (int32_t i = 0; i < st->switch_stmt.case_count; i++)
			for (int32_t k = 0; k < st->switch_stmt.cases[i].stmt_count; k++)
				register_stmt_enums(s, st->switch_stmt.cases[i].stmts[k]);
		break;
	default: break;
	}
}

// Registers every enum in the program — top-level and inline — before structs and
// globals, so their types/array-sizes can reference either. Constants are global.
static void register_enums(sema_t *s) {
	for (int32_t i = 0; i < s->ast->decl_count; i++) {
		const svsl_ast_decl_t *d = s->ast->decls[i];
		switch (d->kind) {
		case svsl_decl_enum:
			register_one_enum(s, &d->enum_decl);
			break;
		case svsl_decl_var:
			register_type_enums(s, d->var.type);
			break;
		case svsl_decl_struct:
			for (int32_t m = 0; m < d->struct_decl.member_count; m++)
				register_type_enums(s, d->struct_decl.members[m]->type);
			break;
		case svsl_decl_block:
			for (int32_t m = 0; m < d->block.member_count; m++)
				register_type_enums(s, d->block.members[m]->type);
			break;
		case svsl_decl_func:
			register_type_enums(s, d->func.return_type);
			for (int32_t pi = 0; pi < d->func.param_count; pi++)
				register_type_enums(s, d->func.params[pi]->type);
			register_stmt_enums(s, d->func.body);
			break;
		default: break;
		}
	}
}

static void register_structs(sema_t *s) {
	// pass 1: create entries so member types can reference any struct
	for (int32_t i = 0; i < s->ast->decl_count; i++) {
		if (s->ast->decls[i]->kind != svsl_decl_struct) continue;
		const svsl_ast_struct_t *st = &s->ast->decls[i]->struct_decl;
		if (struct_find(s->prog, st->name) >= 0) {
			err(s, st->loc, "duplicate struct '%.*s'", st->name);
			continue;
		}
		for (int32_t e = 0; e < s->prog->enums.count; e++)
			if (svsl_str_eq(s->prog->enums.items[e].name, st->name)) {
				err(s, st->loc, "'%.*s' is already an enum", st->name);
				break;
			}
		svsl_str_t name = { .ptr = svsl_arena_strndup(s->arena, st->name.ptr, (size_t)st->name.len),
		                    .len = st->name.len };
		svsl_array_push(s->arena, &s->prog->types.structs, (svsl_struct_info_t){ .name = name });
	}
	// pass 2: members
	for (int32_t i = 0; i < s->ast->decl_count; i++) {
		if (s->ast->decls[i]->kind != svsl_decl_struct) continue;
		const svsl_ast_struct_t *st    = &s->ast->decls[i]->struct_decl;
		int32_t                  index = struct_find(s->prog, st->name);
		if (index < 0) continue;
		svsl_struct_info_t *info = &s->prog->types.structs.items[index];
		bool is_packed = false;
		for (int32_t m = 0; m < st->member_count; m++)
			if (st->members[m]->bit_width >= 0 ||
			    (st->members[m]->bit_format != svsl_bitfmt_raw)) is_packed = true;
		if (is_packed) { pack_struct(s, info, st); continue; }
		for (int32_t m = 0; m < st->member_count; m++) {
			const svsl_ast_var_t *mv                = st->members[m];
			int32_t               explicit_offset   = -1;
			int32_t               explicit_location = -1;
			svsl_attrs_check(s->arena, s->diags, &mv->attrs, svsl_attr_ctx_member, s->prog->porting);
			for (int32_t a = 0; a < mv->attrs.count; a++) {
				int64_t v;
				bool has_int = mv->attrs.items[a].arg_count == 1 &&
				               const_eval_int(s, mv->attrs.items[a].args[0], &v);
				if (svsl_str_eq_cstr(mv->attrs.items[a].name, "offset") ||
				    svsl_str_eq_cstr(mv->attrs.items[a].name, "vk::offset")) {
					if (has_int) explicit_offset = (int32_t)v;
				}
				if (svsl_str_eq_cstr(mv->attrs.items[a].name, "location") ||
				    svsl_str_eq_cstr(mv->attrs.items[a].name, "vk::location")) {
					if (has_int && v >= 0) explicit_location = (int32_t)v;
					else err(s, mv->attrs.items[a].loc, "[location] on '%.*s' needs a non-negative constant", mv->name);
				}
			}
			svsl_array_push(s->arena, &info->members, (svsl_member_t){
				.name              = mv->name,
				.type              = resolve_type(s, mv->type),
				.semantic          = mv->semantic,
				.interp            = mv->interp,
				.explicit_offset   = explicit_offset,
				.explicit_location = explicit_location,
				.loc               = mv->loc });
		}
	}
	// pass 3: containment cycles (struct A { A a; }) would recurse forever in layout
	int32_t  total = s->prog->types.structs.count;
	uint8_t *state = total ? svsl_arena_alloc(s->arena, (size_t)total) : NULL;
	for (int32_t i = 0; i < total; i++)
		if (state[i] == 0) struct_cycle_check(s, i, state);
}

// --- initializer folding (buffer defaults) --------------------------------------------

// writes the constant value of `e` into out (member layout: vectors/matrices dense
// by scalar); returns false when not a compile-time constant of a foldable shape
static bool fold_default(sema_t *s, const svsl_ast_expr_t *e, svsl_type_id_t type, uint8_t *out);

static bool fold_scalar(sema_t *s, const svsl_ast_expr_t *e, svsl_scalar_ scalar, uint8_t *out) {
	bool is_float = scalar == svsl_scalar_float32 || scalar == svsl_scalar_float16 ||
	                scalar == svsl_scalar_float64 || scalar == svsl_scalar_half;

	// integer target: const_eval_int handles literals, unary, arithmetic, and named
	// constants (enum members, static-const ints, spec-const defaults)
	if (!is_float) {
		int64_t iv;
		if (!const_eval_int(s, e, &iv)) return false;
		switch (scalar) {
		case svsl_scalar_bool:  { int32_t  v = iv ? 1 : 0; memcpy(out, &v, 4); return true; }
		case svsl_scalar_int8:  case svsl_scalar_uint8:  { uint8_t  v = (uint8_t)iv;  out[0] = v;         return true; }
		case svsl_scalar_int16: case svsl_scalar_uint16: { uint16_t v = (uint16_t)iv; memcpy(out, &v, 2); return true; }
		case svsl_scalar_int32: case svsl_scalar_uint32: { int32_t  v = (int32_t)iv;  memcpy(out, &v, 4); return true; }
		case svsl_scalar_int64: case svsl_scalar_uint64: { int64_t  v = iv;           memcpy(out, &v, 8); return true; }
		default: return false;
		}
	}

	// float target: literal or +/- literal (no float const-folding pass yet)
	double sign = 1;
	if (e->kind == svsl_expr_unary && (e->unary.op == svsl_tok_minus || e->unary.op == svsl_tok_plus)) {
		if (e->unary.op == svsl_tok_minus) sign = -1;
		e = e->unary.operand;
	}
	double f;
	if      (e->kind == svsl_expr_float_lit) f = e->float_lit.value;
	else if (e->kind == svsl_expr_int_lit)   f = (double)(int64_t)e->int_lit.value;
	else if (e->kind == svsl_expr_bool_lit)  f = e->bool_lit ? 1 : 0;
	else return false;
	f *= sign;

	switch (scalar) {
	case svsl_scalar_float32: case svsl_scalar_half: { float v = (float)f; memcpy(out, &v, 4); return true; }
	case svsl_scalar_float16: { uint16_t v = (uint16_t)svsl_f32_to_f16_bits((float)f); memcpy(out, &v, 2); return true; }
	case svsl_scalar_float64: { double   v = f;                                        memcpy(out, &v, 8); return true; }
	default: return false;
	}
}

static bool fold_default(sema_t *s, const svsl_ast_expr_t *e, svsl_type_id_t type, uint8_t *out) {
	const svsl_type_t *t = svsl_type_get(&s->prog->types, type);
	int32_t            n = svsl_scalar_size(t->scalar);

	if (t->kind == svsl_type_scalar)
		return fold_scalar(s, e, t->scalar, out);

	if (t->kind == svsl_type_vector) {
		const svsl_ast_expr_t **items = NULL;
		int32_t                 count = 0;
		if (e->kind == svsl_expr_init_list) {
			items = (const svsl_ast_expr_t **)e->init_list.items;
			count = e->init_list.count;
		} else if (e->kind == svsl_expr_ctor) {
			items = (const svsl_ast_expr_t **)e->ctor.args;
			count = e->ctor.arg_count;
		} else {
			return false;
		}
		if (count == 1) { // broadcast: float4(1)
			if (!fold_scalar(s, items[0], t->scalar, out)) return false;
			for (int32_t k = 1; k < t->count; k++)
				memcpy(out + k * n, out, (size_t)n);
			return true;
		}
		if (count != t->count) return false;
		for (int32_t k = 0; k < count; k++)
			if (!fold_scalar(s, items[k], t->scalar, out + k * n)) return false;
		return true;
	}
	if (t->kind == svsl_type_matrix && e->kind == svsl_expr_init_list &&
	    e->init_list.count == t->rows * t->cols) {
		// flat HLSL initializer fills row by row; storage is column-major with
		// pack16 column stride (defaults only exist in the pack16 $Globals buffer)
		uint32_t col_stride = ((uint32_t)t->rows * (uint32_t)n + 15) & ~15u;
		for (int32_t row = 0; row < t->rows; row++)
			for (int32_t col = 0; col < t->cols; col++) {
				const svsl_ast_expr_t *item = e->init_list.items[row * t->cols + col];
				if (!fold_scalar(s, item, t->scalar, out + col * col_stride + (uint32_t)row * (uint32_t)n))
					return false;
			}
		return true;
	}
	return false; // array defaults not supported (match skshaderc)
}

// --- bindings ------------------------------------------------------------------------

static svsl_binding_t binding_from_reg(const svsl_ast_reg_t *reg) {
	if (!reg->present) return (svsl_binding_t){ .slot = -1 };
	return (svsl_binding_t){
		.cls    = reg->cls,
		.slot   = reg->slot,
		.space  = reg->space,
		.direct = reg->direct };
}

static bool slot_taken(sema_t *s, char cls, int32_t slot, int32_t space) {
	for (int32_t i = 0; i < s->prog->resources.count; i++) {
		const svsl_binding_t *b = &s->prog->resources.items[i].bind;
		if (b->cls == cls && b->slot == slot && b->space == space) return true;
	}
	if (cls == 'b') {
		for (int32_t i = 0; i < s->prog->buffers.count; i++) {
			const svsl_buffer_t *buf = &s->prog->buffers.items[i];
			if (buf->kind != svsl_block_uniform) continue;
			if (buf->bind.slot == slot && buf->bind.space == space) return true;
		}
	}
	return false;
}

// slots never free during assignment, so per (cls, space) the lowest free slot
// is monotone — the hint skips re-probing filled territory (degenerate inputs
// with thousands of resources made the from-zero scan cubic)
typedef struct slot_hint_t {
	char    cls;
	int32_t space;
	int32_t next;
} slot_hint_t;

static int32_t lowest_free_slot(sema_t *s, char cls, int32_t space,
                                slot_hint_t *hints, int32_t *ref_hint_count) {
	slot_hint_t *h = NULL;
	for (int32_t i = 0; i < *ref_hint_count && !h; i++)
		if (hints[i].cls == cls && hints[i].space == space) h = &hints[i];
	if (!h && *ref_hint_count < 16) {
		h  = &hints[(*ref_hint_count)++];
		*h = (slot_hint_t){ .cls = cls, .space = space };
	}
	for (int32_t slot = h ? h->next : 0;; slot++)
		if (!slot_taken(s, cls, slot, space)) {
			if (h) h->next = slot + 1;
			return slot;
		}
}

// --- metadata ---------------------------------------------------------------------------

static const svsl_pp_meta_t *meta_find(const svsl_pp_result_t *pp, const char *name) {
	for (int32_t i = 0; i < pp->meta_count; i++)
		if (svsl_str_eq_cstr(pp->metas[i].name, name)) return &pp->metas[i];
	return NULL;
}

// writes one //-- default token at the member's scalar width and encoding
// (text-metadata counterpart of fold_scalar)
static void write_meta_scalar(const char *buf, svsl_scalar_ scalar, uint8_t *out) {
	switch (scalar) {
	case svsl_scalar_bool:    { int32_t  v = strtol (buf, NULL, 0) ? 1 : 0;                  memcpy(out, &v, 4); break; }
	case svsl_scalar_int8:  case svsl_scalar_uint8:  { uint8_t  v = (uint8_t) strtol (buf, NULL, 0); out[0] = v;         break; }
	case svsl_scalar_int16: case svsl_scalar_uint16: { uint16_t v = (uint16_t)strtol (buf, NULL, 0); memcpy(out, &v, 2); break; }
	case svsl_scalar_int64: case svsl_scalar_uint64: { int64_t  v = (int64_t) strtoll(buf, NULL, 0); memcpy(out, &v, 8); break; }
	case svsl_scalar_float16: { uint16_t v = (uint16_t)svsl_f32_to_f16_bits(strtof(buf, NULL)); memcpy(out, &v, 2); break; }
	case svsl_scalar_float64: { double   v = strtod(buf, NULL);                                 memcpy(out, &v, 8); break; }
	// int32/uint32 and half (RelaxedPrecision float32) both write 4 bytes
	default: { int32_t iv; if (scalar == svsl_scalar_int32 || scalar == svsl_scalar_uint32) { iv = (int32_t)strtol(buf, NULL, 0); memcpy(out, &iv, 4); } else { float v = strtof(buf, NULL); memcpy(out, &v, 4); } break; }
	}
}

// parses "1, 0.5, 0, 1" into member bytes
static bool parse_value_list(svsl_str_t text, const svsl_type_t *t, uint8_t *out) {
	int32_t component = 0;
	int32_t max       = t->kind == svsl_type_vector ? t->count : 1;
	int32_t stride    = svsl_scalar_size(t->scalar);
	int32_t pos       = 0;

	while (pos < text.len && component < max) {
		int32_t next  = pos;
		while (next < text.len && text.ptr[next] != ',') next++;
		svsl_str_t part = svsl_str_trim(svsl_str_slice(text, pos, next));
		if (part.len == 0) return false;

		char buf[64];
		if (part.len >= (int32_t)sizeof(buf)) return false;
		memcpy(buf, part.ptr, (size_t)part.len);
		buf[part.len] = '\0';

		write_meta_scalar(buf, t->scalar, out + component * stride);
		component++;
		pos = next + 1;
	}
	if (component == 1 && max > 1) { // broadcast single value
		for (int32_t k = 1; k < max; k++)
			memcpy(out + k * stride, out, (size_t)stride);
	}
	return component > 0;
}

// --- main pass -----------------------------------------------------------------------------

// a buffer's pack keyword names its memory layout; the fallback is the layout
// used when no keyword was given (differs for uniform blocks vs object buffers)
static svsl_layout_ pack_to_layout(svsl_pack_ pack, svsl_layout_ fallback) {
	switch (pack) {
	case svsl_pack_1:   return svsl_layout_pack1;
	case svsl_pack_8:   return svsl_layout_pack8;
	case svsl_pack_16:  return svsl_layout_pack16;
	case svsl_pack_430: return svsl_layout_std430;
	default:            return fallback;
	}
}

static void add_buffer_block(sema_t *s, const svsl_ast_block_t *block_in) {
	svsl_ast_block_t local = *block_in;
	svsl_ast_block_t *block = &local;
	svsl_attrs_check(s->arena, s->diags, &block->attrs, svsl_attr_ctx_block, s->prog->porting);
	for (int32_t a = 0; a < block->attrs.count; a++) {
		const svsl_ast_attr_t *attr = &block->attrs.items[a];
		// [[vk::push_constant]] re-kinds a cbuffer as a push-constant block
		if (svsl_str_eq_cstr(attr->name, "vk::push_constant"))
			block->kind = svsl_block_pushconstant;
		// [[vk::binding(b, set)]] is a DIRECT descriptor binding (no register class offset)
		if (svsl_str_eq_cstr(attr->name, "vk::binding") && !attr_binding(s, attr, &local.reg))
			err(s, attr->loc, "[[vk::binding]] on '%.*s' needs (binding[, set]) constants", block->name);
	}

	bool         is_uniform = block->kind == svsl_block_uniform;
	svsl_layout_ layout     = pack_to_layout((svsl_pack_)block->pack,
	                                         is_uniform ? svsl_layout_pack16 : svsl_layout_std430);
	if (is_uniform && layout != svsl_layout_pack16) {
		err(s, block->loc, "uniform buffer '%.*s' must use pack16", block->name);
		layout = svsl_layout_pack16;
	}

	// resolve members
	svsl_array_t(svsl_member_t) members = {0};
	for (int32_t i = 0; i < block->member_count; i++) {
		const svsl_ast_var_t *mv              = block->members[i];
		int32_t               explicit_offset = -1;
		svsl_attrs_check(s->arena, s->diags, &mv->attrs, svsl_attr_ctx_member, s->prog->porting);
		for (int32_t a = 0; a < mv->attrs.count; a++) {
			if (svsl_str_eq_cstr(mv->attrs.items[a].name, "offset") ||
			    svsl_str_eq_cstr(mv->attrs.items[a].name, "vk::offset")) {
				int64_t v;
				if (attr_int1(s, &mv->attrs.items[a], &v)) explicit_offset = (int32_t)v;
			}
		}
		svsl_type_id_t type = resolve_type(s, mv->type);
		if (type == SVSL_TYPE_NONE) continue;
		const svsl_type_t *t = svsl_type_get(&s->prog->types, type);
		if (t->kind == svsl_type_array && t->array_count == 0 &&
		    (is_uniform || i != block->member_count - 1))
			err(s, mv->loc, "runtime-sized array '%.*s' must be the last member of a storagebuffer", mv->name);
		svsl_array_push(s->arena, &members, (svsl_member_t){
			.name = mv->name, .type = type, .explicit_offset = explicit_offset,
			.explicit_location = -1, .loc = mv->loc });
	}

	uint32_t *offsets = svsl_arena_alloc(s->arena, (size_t)(members.count > 0 ? members.count : 1) * sizeof(uint32_t));
	int32_t   bad     = -1;
	uint32_t  size    = svsl_layout_members(&s->prog->types, members.items, members.count, layout, offsets, &bad);
	if (bad >= 0)
		err(s, members.items[bad].loc, "[offset] on '%.*s' moves backwards or breaks alignment", members.items[bad].name);

	// explicit pack1/pack8 layouts may break the core relaxed rules; that's legal
	// but requires the scalarBlockLayout device feature — record it
	if (layout == svsl_layout_pack1 || layout == svsl_layout_pack8)
		for (int32_t i = 0; i < members.count; i++)
			if (svsl_layout_needs_scalar(&s->prog->types, members.items[i].type, layout, offsets[i], NULL)) {
				s->prog->needs_scalar_layout = true;
				break;
			}

	svsl_buffer_t buffer = {
		.name   = block->name,
		.kind   = block->kind,
		.layout = layout,
		.bind   = binding_from_reg(&block->reg),
		.size   = size };
	for (int32_t i = 0; i < members.count; i++) {
		svsl_array_push(s->arena, &buffer.members, (svsl_buf_member_t){
			.name   = members.items[i].name,
			.type   = members.items[i].type,
			.offset = offsets[i],
			.size   = svsl_layout_size(&s->prog->types, members.items[i].type, layout),
			.loc    = members.items[i].loc });
	}
	svsl_array_push(s->arena, &s->prog->buffers, buffer);

	// storagebuffer blocks also reflect as buffer resources (like StructuredBuffer),
	// with their members still resolvable by name through the buffer entry
	if (block->kind == svsl_block_storagebuffer) {
		uint32_t elem_size = size;
		if (members.count > 0) {
			const svsl_type_t *last = svsl_type_get(&s->prog->types, members.items[members.count - 1].type);
			if (last->kind == svsl_type_array && last->array_count == 0)
				elem_size = svsl_layout_array_stride(&s->prog->types, last->elem, layout);
		}
		svsl_binding_t bind = binding_from_reg(&block->reg);
		if (!bind.direct && bind.cls == 0)
			bind.cls = (block->flags & svsl_var_flag_readonly) ? 't' : 'u';
		svsl_array_push(s->arena, &s->prog->resources, (svsl_resource_t){
			.name         = block->name,
			.kind         = (block->flags & svsl_var_flag_readonly) ? svsl_res_structured : svsl_res_rw_structured,
			.bind         = bind,
			.sampler_slot = -1,
			.subpass_index= -1,
			.buffer_index = s->prog->buffers.count - 1,
			.element_size = elem_size,
			.layout       = (uint8_t)layout,
			.loc          = block->loc });
	}
}

static void add_spec_const(sema_t *s, const svsl_ast_var_t *var) {
	svsl_type_id_t type = resolve_type(s, var->type);
	if (type == SVSL_TYPE_NONE) return;
	const svsl_type_t *t = svsl_type_get(&s->prog->types, type);
	if (t->kind != svsl_type_scalar || svsl_scalar_size(t->scalar) != 4) {
		err(s, var->loc, "specialization constant '%.*s' must be a 32-bit scalar or bool", var->name);
		return;
	}

	int32_t explicit_id = -1;
	for (int32_t a = 0; a < var->attrs.count; a++) {
		if (svsl_str_eq_cstr(var->attrs.items[a].name, "specialization") ||
		    svsl_str_eq_cstr(var->attrs.items[a].name, "vk::constant_id")) {
			int64_t v;
			if (attr_int1(s, &var->attrs.items[a], &v)) explicit_id = (int32_t)v;
		}
	}

	uint32_t bits = 0;
	if (!var->init || !fold_scalar(s, var->init, t->scalar, (uint8_t *)&bits)) {
		err(s, var->loc, "specialization constant '%.*s' needs a constant default", var->name);
		return;
	}

	uint32_t id = 0;
	if (explicit_id >= 0) {
		id = (uint32_t)explicit_id;
	} else { // lowest free id, never colliding with explicit ones
		bool taken;
		do {
			taken = false;
			for (int32_t i = 0; i < s->prog->spec_consts.count; i++)
				if (s->prog->spec_consts.items[i].id == id) { taken = true; id++; break; }
		} while (taken);
	}
	for (int32_t i = 0; i < s->prog->spec_consts.count; i++)
		if (s->prog->spec_consts.items[i].id == id) { // explicit collision
			svsl_str_t other = s->prog->spec_consts.items[i].name;
			svsl_diag_add(s->arena, s->diags, svsl_severity_error, var->loc,
			              "specialization id %u already used by '%.*s'", id, other.len, other.ptr);
			break;
		}

	svsl_array_push(s->arena, &s->prog->spec_consts, (svsl_spec_const_t){
		.name = var->name, .id = id, .default_bits = bits, .type = type, .loc = var->loc });
}

// True for SV_-prefixed semantics other than SV_Position — the system-value
// spellings. SV_Position on a *vertex input* is a regular attribute (StereoKit's
// pos field — skshaderc reflects it as Position0), so SV_* alone doesn't
// disqualify. Syntactic only: whether the spelling names a real system value is
// the semantics table's call (vs_input_is_attribute).
static bool semantic_is_generated(svsl_str_t semantic) {
	bool sv = semantic.len >= 3 &&
	          (semantic.ptr[0] == 'S' || semantic.ptr[0] == 's') &&
	          (semantic.ptr[1] == 'V' || semantic.ptr[1] == 'v') &&
	          semantic.ptr[2] == '_';
	if (!sv) return false;
	svsl_str_t rest = svsl_str_slice(semantic, 3, semantic.len);
	// case-insensitive compare against "position"
	if (rest.len == 8) {
		static const char *pos = "position";
		bool match = true;
		for (int32_t i = 0; i < 8; i++) {
			char c = rest.ptr[i];
			if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
			if (c != pos[i]) { match = false; break; }
		}
		if (match) return false;
	}
	return true;
}

// A vertex-entry input is a mesh attribute (collected for reflection) or a
// system-generated value (a table-verified vs-input builtin — skipped). An SV_*
// spelling the table doesn't recognize is neither: it can't be fed by a mesh
// and the container can't name it for attribute matching, so it is an error —
// not a silent drop that would leave the metadata disagreeing with the SPIR-V.
// This must classify exactly like the emitter's counted/builtin split; both
// consult the same semantics table, and the emitter's record_vs_input guard
// fails the compile if they ever drift.
static bool vs_input_is_attribute(sema_t *s, svsl_str_t semantic, svsl_loc_t loc) {
	svsl_semantic_info_t info;
	if (svsl_semantic_lookup(semantic, svsl_sem_vs_in, &info) && info.is_builtin)
		return false; // system-generated (SV_VertexID, SV_InstanceID, SV_ViewID...)
	if (semantic_is_generated(semantic)) {
		err(s, loc, "unknown system-value semantic '%.*s' on a vertex input", semantic);
		return false;
	}
	return true;
}

static void collect_vertex_inputs(sema_t *s, const svsl_ast_func_t *func) {
	for (int32_t p = 0; p < func->param_count; p++) {
		const svsl_ast_var_t *param = func->params[p];
		svsl_type_id_t        type  = resolve_type(s, param->type);
		if (type == SVSL_TYPE_NONE) continue;
		const svsl_type_t *t = svsl_type_get(&s->prog->types, type);

		if (t->kind == svsl_type_struct) {
			const svsl_struct_info_t *info = &s->prog->types.structs.items[t->struct_index];
			for (int32_t m = 0; m < info->members.count; m++) {
				const svsl_member_t *member = &info->members.items[m];
				if (!vs_input_is_attribute(s, member->semantic, member->loc)) continue;
				svsl_array_push(s->arena, &s->prog->vertex_inputs, (svsl_vertex_input_t){
					.name = member->name, .type = member->type, .semantic = member->semantic });
			}
		} else if (vs_input_is_attribute(s, param->semantic, param->loc)) {
			svsl_array_push(s->arena, &s->prog->vertex_inputs, (svsl_vertex_input_t){
				.name = param->name, .type = type, .semantic = param->semantic });
		}
	}
}

static void add_entry(sema_t *s, const svsl_ast_func_t *func, svsl_stage_ stage) {
	svsl_entry_t entry = { .name = func->name, .stage = stage, .func = func,
	                       .workgroup = { 1, 1, 1 } };

	for (int32_t a = 0; a < func->attrs.count; a++) {
		const svsl_ast_attr_t *attr = &func->attrs.items[a];
		if (svsl_str_eq_cstr(attr->name, "numthreads") || svsl_str_eq_cstr(attr->name, "compute")) {
			for (int32_t i = 0; i < attr->arg_count && i < 3; i++) {
				int64_t v;
				if (const_eval_int(s, attr->args[i], &v) && v > 0) entry.workgroup[i] = (int32_t)v;
				else err(s, attr->loc, "bad workgroup size on '%.*s'", func->name);
			}
		}
		if (svsl_str_eq_cstr(attr->name, "wave_size")) {
			int64_t v = 0;
			if (attr_int1(s, attr, &v) && v >= 4 && v <= 128 && (v & (v - 1)) == 0) {
				entry.wave_size = (int32_t)v;
				if (s->prog->wave_size == 0) s->prog->wave_size = (int32_t)v;
			} else {
				err(s, attr->loc, "wave_size on '%.*s' must be a power of two in [4,128]", func->name);
			}
		}
		// VK_QCOM_tile_shading execution modes
		if (svsl_str_eq_cstr(attr->name, "tile_shading_rate_qcom")) {
			if (stage != svsl_stage_compute)
				err(s, attr->loc, "[tile_shading_rate_qcom] on '%.*s': compute entries only", func->name);
			// the rate replaces LocalSize — the implementation derives the
			// workgroup shape from it, so an explicit numthreads is an error
			for (int32_t k = 0; k < func->attrs.count; k++)
				if (svsl_str_eq_cstr(func->attrs.items[k].name, "numthreads") ||
				    svsl_str_eq_cstr(func->attrs.items[k].name, "compute"))
					err(s, attr->loc, "[tile_shading_rate_qcom] on '%.*s' replaces [numthreads] — remove one", func->name);
			bool ok = attr->arg_count == 3;
			for (int32_t i = 0; ok && i < 3; i++) {
				int64_t v;
				ok = const_eval_int(s, attr->args[i], &v) && v > 0 &&
				     (i == 2 || (v & (v - 1)) == 0); // x/y rates must be powers of two
				if (ok) entry.tile_rate[i] = (int32_t)v;
			}
			if (!ok)
				err(s, attr->loc, "[tile_shading_rate_qcom] on '%.*s' needs (x, y, z) with power-of-two x/y", func->name);
		}
		if (svsl_str_eq_cstr(attr->name, "non_coherent_tile_reads_qcom")) {
			if (stage != svsl_stage_pixel)
				err(s, attr->loc, "[non_coherent_tile_reads_qcom] on '%.*s': pixel entries only", func->name);
			entry.non_coherent_tile_reads = true;
		}
	}

	for (int32_t i = 0; i < s->prog->entries.count; i++) {
		if (s->prog->entries.items[i].stage == stage) {
			err(s, func->loc, "duplicate entry point for stage ('%.*s')", func->name);
			return;
		}
	}
	svsl_array_push(s->arena, &s->prog->entries, entry);
	if (stage == svsl_stage_vertex) collect_vertex_inputs(s, func);
}

static void discover_entries(sema_t *s, const svsl_sema_options_t *opt) {
	const char *name_vs = opt && opt->entry_vs ? opt->entry_vs : "vs";
	const char *name_ps = opt && opt->entry_ps ? opt->entry_ps : "ps";
	const char *name_cs = opt && opt->entry_cs ? opt->entry_cs : "cs";

	for (int32_t i = 0; i < s->ast->decl_count; i++) {
		if (s->ast->decls[i]->kind != svsl_decl_func) continue;
		const svsl_ast_func_t *func = &s->ast->decls[i]->func;
		svsl_attrs_check(s->arena, s->diags, &func->attrs, svsl_attr_ctx_func, s->prog->porting);
		if (!func->body) continue;

		bool    attr_stage = false;
		int32_t stage      = 0;
		bool    has_numthreads = false;
		for (int32_t a = 0; a < func->attrs.count; a++) {
			svsl_str_t n = func->attrs.items[a].name;
			if      (svsl_str_eq_cstr(n, "vertex"))                                  { stage = svsl_stage_vertex;  attr_stage = true; }
			else if (svsl_str_eq_cstr(n, "fragment") || svsl_str_eq_cstr(n, "pixel")) { stage = svsl_stage_pixel;   attr_stage = true; }
			else if (svsl_str_eq_cstr(n, "compute"))                                 { stage = svsl_stage_compute; attr_stage = true; }
			else if (svsl_str_eq_cstr(n, "numthreads"))                              has_numthreads = true;
			else if (svsl_str_eq_cstr(n, "tile_shading_rate_qcom"))                  has_numthreads = true; // also marks a compute entry
		}
		if (!attr_stage) {
			if      (svsl_str_eq_cstr(func->name, name_vs)) stage = svsl_stage_vertex;
			else if (svsl_str_eq_cstr(func->name, name_ps)) stage = svsl_stage_pixel;
			else if (svsl_str_eq_cstr(func->name, name_cs) || has_numthreads) stage = svsl_stage_compute;
		}
		if (stage != 0) add_entry(s, func, (svsl_stage_)stage);
	}
}

const svsl_func_info_t *svsl_program_func_info(const svsl_program_t *prog,
                                               const svsl_ast_func_t *func) {
	for (int32_t f = 0; f < prog->functions.count; f++)
		if (prog->functions.items[f].func == func) return &prog->functions.items[f];
	return NULL;
}

bool svsl_sema_run(svsl_arena_t *arena, const svsl_ast_t *ast, const svsl_pp_result_t *pp,
                   const char *opt_filename, const svsl_sema_options_t *opt_options,
                   svsl_program_t *out_program, svsl_diag_list_t *ref_diags) {
	int32_t errors_before = ref_diags->error_count;

	*out_program = (svsl_program_t){ .types = {
		.arena         = arena,
		.half_strict16 = opt_options && opt_options->half_strict16,
	}, .porting = opt_options && opt_options->porting_hints, .ast = ast };
	sema_t s = { .arena = arena, .prog = out_program, .diags = ref_diags, .ast = ast };

	// static-const integers first: struct members and arrays may size off them
	for (int32_t i = 0; i < ast->decl_count; i++) {
		if (ast->decls[i]->kind != svsl_decl_var) continue;
		const svsl_ast_var_t *var = &ast->decls[i]->var;
		if (!(var->flags & (svsl_var_flag_static | svsl_var_flag_const))) continue;
		if (var->flags & svsl_var_flag_specialization) continue;
		if (!var->init) continue;
		int64_t v;
		if (const_eval_int(&s, var->init, &v)) {
			svsl_array_push(arena, &s.prog->const_globals, (svsl_global_t){
				.name = var->name, .type = SVSL_TYPE_NONE, .var = var,
				.has_int = true, .int_value = v });
		}
	}

	register_enums(&s);  // enum type aliases + global constants; structs may use both
	register_structs(&s);

	// spec constants next: array sizes may reference them
	for (int32_t i = 0; i < ast->decl_count; i++) {
		if (ast->decls[i]->kind != svsl_decl_var) continue;
		if (ast->decls[i]->var.flags & svsl_var_flag_specialization)
			add_spec_const(&s, &ast->decls[i]->var);
	}

	// buffer blocks
	for (int32_t i = 0; i < ast->decl_count; i++)
		if (ast->decls[i]->kind == svsl_decl_block)
			add_buffer_block(&s, &ast->decls[i]->block);

	// resource globals + $Globals members
	svsl_array_t(const svsl_ast_var_t *) globals = {0};
	for (int32_t i = 0; i < ast->decl_count; i++) {
		if (ast->decls[i]->kind != svsl_decl_var) continue;
		const svsl_ast_var_t *var = &ast->decls[i]->var;
		svsl_attrs_check(arena, s.diags, &var->attrs, svsl_attr_ctx_var, s.prog->porting);
		if (var->pack && (var->flags & (svsl_var_flag_workgroup | svsl_var_flag_static | svsl_var_flag_const | svsl_var_flag_specialization)))
			err(&s, var->loc, "layout keyword on '%.*s': only buffer resources have a memory layout", var->name);
		if (var->flags & svsl_var_flag_specialization) continue;
		if (var->flags & svsl_var_flag_workgroup) {
			svsl_array_push(arena, &s.prog->workgroup_vars, (svsl_global_t){
				.name = var->name, .type = resolve_type(&s, var->type), .var = var });
			continue;
		}
		if (var->flags & (svsl_var_flag_static | svsl_var_flag_const)) {
			// fill in the type on the pre-pass entry, or add a non-integer constant
			svsl_type_id_t type  = infer_array_size(&s, resolve_type(&s, var->type), var);
			bool           found = false;
			for (int32_t k = 0; k < s.prog->const_globals.count; k++) {
				if (svsl_str_eq(s.prog->const_globals.items[k].name, var->name)) {
					s.prog->const_globals.items[k].type = type;
					found = true;
					break;
				}
			}
			if (!found)
				svsl_array_push(arena, &s.prog->const_globals, (svsl_global_t){
					.name = var->name, .type = type, .var = var });
			continue;
		}
		svsl_type_id_t type = resolve_type(&s, var->type);
		if (type == SVSL_TYPE_NONE) continue;
		const svsl_type_t *t = svsl_type_get(&s.prog->types, type);
		const svsl_type_t *base = t->kind == svsl_type_array ? svsl_type_get(&s.prog->types, t->elem) : t;

		if (svsl_type_is_resource(base)) {
			svsl_res_kind_ kind =
				base->kind == svsl_type_texture  ? svsl_res_texture :
				base->kind == svsl_type_sampler  ? svsl_res_sampler :
				base->kind == svsl_type_image    ? svsl_res_image :
				base->kind == svsl_type_subpass  ? svsl_res_subpass :
				base->kind == svsl_type_tileimage? svsl_res_tileimage :
				base->is_rw ? svsl_res_rw_structured : svsl_res_structured;
			// [[vk::image_format("rgba8")]] sets the storage image's format,
			// overriding the texel-type inference (same as the template spelling
			// Image2D<float4, rgba8>)
			for (int32_t a = 0; a < var->attrs.count; a++) {
				const svsl_ast_attr_t *attr = &var->attrs.items[a];
				if (!svsl_str_eq_cstr(attr->name, "vk::image_format") &&
				    !svsl_str_eq_cstr(attr->name, "image_format")) continue;
				if (base->kind != svsl_type_image) {
					err(&s, attr->loc, "[image_format] on '%.*s': only storage images (RWTexture/Image) have a format", var->name);
					continue;
				}
				svsl_str_t fmt_name =
					attr->arg_count == 1 && attr->args[0]->kind == svsl_expr_string_lit
					? attr->args[0]->string_lit : (svsl_str_t){0};
				if (!svsl_image_format_find(fmt_name, NULL)) {
					err(&s, attr->loc, "unrecognized image format '%.*s'", fmt_name);
					continue;
				}
				svsl_type_t with_format = *base;
				with_format.format = fmt_name;
				svsl_type_id_t img = svsl_type_intern(&s.prog->types, with_format);
				if (t->kind == svsl_type_array) {
					svsl_type_t arr = *t;
					arr.elem = img;
					type    = svsl_type_intern(&s.prog->types, arr);
				} else {
					type = img;
				}
				// interning may grow the type table; re-derive the pointers
				t    = svsl_type_get(&s.prog->types, type);
				base = t->kind == svsl_type_array ? svsl_type_get(&s.prog->types, t->elem) : t;
			}
			// Object-form buffer elements default to C layout (pack1 rules), refusing
			// what baseline Vulkan can't express: a straddling vector is an error
			// unless a layout keyword makes the device requirement explicit.
			svsl_layout_ elem_layout  = svsl_layout_pack1;
			uint32_t     element_size = 0;
			if (var->pack && base->kind != svsl_type_buffer)
				err(&s, var->loc, "layout keyword on '%.*s': only buffer resources have a memory layout", var->name);
			if (base->kind == svsl_type_buffer && base->elem != SVSL_TYPE_NONE) {
				elem_layout  = pack_to_layout((svsl_pack_)var->pack, svsl_layout_pack1);
				element_size = svsl_layout_array_stride(&s.prog->types, base->elem, elem_layout);
				// the element sits in an implicit runtime array; probe it as one so
				// the stride-alignment and per-element straddle checks match the block
				// form (needs_scalar's array case walks every distinct stride phase, not
				// just element 0 — a vector can straddle only from element 1 onward)
				svsl_type_id_t elem_array = svsl_type_array_id(&s.prog->types, base->elem, 0);
				bool           violates   = svsl_layout_needs_scalar(&s.prog->types, elem_array, elem_layout, 0, NULL);
				// interning the probe array may have grown the type table; re-derive
				t    = svsl_type_get(&s.prog->types, type);
				base = t->kind == svsl_type_array ? svsl_type_get(&s.prog->types, t->elem) : t;
				if ((elem_layout == svsl_layout_pack1 || elem_layout == svsl_layout_pack8) && violates) {
					if (var->pack == svsl_pack_default)
						svsl_diag_add(arena, s.diags, svsl_severity_error, var->loc,
						              "the C-packed element layout of '%.*s' is not expressible under "
						              "core Vulkan rules — reorder or pad the struct so vectors don't "
						              "cross 16-byte boundaries and the stride is 16-aligned, or "
						              "declare 'pack1' to require the scalarBlockLayout device feature",
						              var->name.len, var->name.ptr);
					else
						s.prog->needs_scalar_layout = true;
				}
			}
			int32_t subpass_index = -1; // attachment index for subpass AND tile image
			if ((base->kind == svsl_type_subpass || base->kind == svsl_type_tileimage) &&
			    var->type->index) {
				int64_t v;
				if (const_eval_int(&s, var->type->index, &v)) subpass_index = (int32_t)v;
			}
			svsl_ast_reg_t reg = var->reg;
			for (int32_t a = 0; a < var->attrs.count; a++) {
				const svsl_ast_attr_t *attr = &var->attrs.items[a];
				int64_t v;
				// [[vk::input_attachment_index(N)]] = SubpassInput<T, N>
				if (svsl_str_eq_cstr(attr->name, "vk::input_attachment_index") ||
				    svsl_str_eq_cstr(attr->name, "input_attachment_index")) {
					if (base->kind != svsl_type_subpass)
						err(&s, attr->loc, "[input_attachment_index] on '%.*s': only SubpassInput has one", var->name);
					else if (attr_int1(&s, attr, &v) && v >= 0)
						subpass_index = (int32_t)v;
					else
						err(&s, attr->loc, "[input_attachment_index] on '%.*s' needs a non-negative constant", var->name);
				}
				// [[vk::binding(b, set)]] — direct descriptor binding, replaces register()
				if (svsl_str_eq_cstr(attr->name, "vk::binding") && !attr_binding(&s, attr, &reg))
					err(&s, attr->loc, "[[vk::binding]] on '%.*s' needs (binding[, set]) constants", var->name);
			}
			// [tile_attachment] — VK_QCOM_tile_shading: the variable lives in the
			// TileAttachmentQCOM storage class (still a set/binding descriptor)
			bool tile_attachment = false;
			for (int32_t a = 0; a < var->attrs.count; a++) {
				if (!svsl_str_eq_cstr(var->attrs.items[a].name, "tile_attachment")) continue;
				bool tex_2d = (base->kind == svsl_type_texture || base->kind == svsl_type_image) &&
				              base->dim == svsl_texdim_2d && !base->arrayed && !base->multisampled;
				if (!tex_2d)
					err(&s, var->attrs.items[a].loc,
					    "[tile_attachment] on '%.*s': only Texture2D and RWTexture2D can be tile attachments", var->name);
				else
					tile_attachment = true;
			}
			svsl_array_push(arena, &s.prog->resources, (svsl_resource_t){
				.name          = var->name,
				.kind          = kind,
				.type          = type,
				.bind          = binding_from_reg(&reg),
				.sampler_slot  = -1,
				.subpass_index = subpass_index,
				.buffer_index  = -1,
				.element_size  = element_size,
				.layout        = (uint8_t)elem_layout,
				.tile_attachment = tile_attachment,
				.loc           = var->loc });
			continue;
		}
		if (var->pack)
			err(&s, var->loc, "layout keyword on '%.*s': only buffer resources have a memory layout", var->name);
		svsl_array_push(arena, &globals, var);
	}

	// $Globals buffer from bare globals (glslang calls it "$Global")
	if (globals.count > 0) {
		svsl_array_t(svsl_member_t) members = {0};
		for (int32_t i = 0; i < globals.count; i++) {
			svsl_array_push(arena, &members, (svsl_member_t){
				.name              = globals.items[i]->name,
				.type              = resolve_type(&s, globals.items[i]->type),
				.explicit_offset   = -1,
				.explicit_location = -1,
				.loc               = globals.items[i]->loc });
		}
		uint32_t *offsets = svsl_arena_alloc(arena, (size_t)members.count * sizeof(uint32_t));
		int32_t   bad;
		uint32_t  size = svsl_layout_members(&s.prog->types, members.items, members.count,
		                                     svsl_layout_pack16, offsets, &bad);

		svsl_buffer_t buffer = {
			.name   = svsl_str("$Global"),
			.kind   = svsl_block_uniform,
			.layout = svsl_layout_pack16,
			.bind   = { .cls = 'b', .slot = -1 },
			.size   = size };
		buffer.defaults = svsl_arena_alloc(arena, size ? size : 1);
		bool any_default = false;
		for (int32_t i = 0; i < members.count; i++) {
			svsl_array_push(arena, &buffer.members, (svsl_buf_member_t){
				.name   = members.items[i].name,
				.type   = members.items[i].type,
				.offset = offsets[i],
				.size   = svsl_layout_size(&s.prog->types, members.items[i].type, svsl_layout_pack16),
				.loc    = members.items[i].loc });
			if (globals.items[i]->init) {
				if (fold_default(&s, globals.items[i]->init, members.items[i].type, buffer.defaults + offsets[i]))
					any_default = true;
				else
					err(&s, globals.items[i]->loc, "initializer for '%.*s' is not a foldable constant", globals.items[i]->name);
			}
		}
		if (!any_default) buffer.defaults = NULL;
		svsl_array_push(arena, &s.prog->buffers, buffer);
	}

	// explicit-slot duplicate check, then auto-assign the rest
	for (int32_t i = 0; i < s.prog->resources.count; i++) {
		svsl_resource_t *res = &s.prog->resources.items[i];
		if (res->bind.slot < 0 || res->bind.direct) continue;
		for (int32_t k = i + 1; k < s.prog->resources.count; k++) {
			const svsl_resource_t *other = &s.prog->resources.items[k];
			if (other->bind.slot == res->bind.slot && other->bind.cls == res->bind.cls &&
			    other->bind.space == res->bind.space && !other->bind.direct)
				err(&s, other->loc, "register %.*s collides with an earlier resource", other->name);
		}
	}
	slot_hint_t hints[16];
	int32_t     hint_count = 0;
	for (int32_t i = 0; i < s.prog->buffers.count; i++) { // $Global auto-slot
		svsl_buffer_t *buf = &s.prog->buffers.items[i];
		if (buf->kind == svsl_block_uniform && buf->bind.slot < 0)
			buf->bind.slot = lowest_free_slot(&s, 'b', buf->bind.space, hints, &hint_count);
	}
	for (int32_t i = 0; i < s.prog->resources.count; i++) {
		svsl_resource_t *res = &s.prog->resources.items[i];
		if (res->kind == svsl_res_tileimage) { // tile memory, not a descriptor
			if (res->bind.slot >= 0 || res->bind.direct)
				err(&s, res->loc, "tile image '%.*s' is not a descriptor and takes no register or binding", res->name);
			if (res->subpass_index < 0) res->subpass_index = 0; // attachment location
			continue;
		}
		if (res->bind.slot >= 0 || res->bind.direct) continue;
		char cls =
			res->kind == svsl_res_texture ? 't' :
			res->kind == svsl_res_sampler ? 's' :
			res->kind == svsl_res_structured ? 't' :
			res->kind == svsl_res_rw_structured || res->kind == svsl_res_image ? 'u' : 't';
		res->bind.cls = cls;

		// a sampler named <texture>_s pairs with that texture's slot
		if (res->kind == svsl_res_sampler && res->name.len > 2 &&
		    memcmp(res->name.ptr + res->name.len - 2, "_s", 2) == 0) {
			svsl_str_t tex_name = svsl_str_slice(res->name, 0, res->name.len - 2);
			bool       paired   = false;
			for (int32_t k = 0; k < s.prog->resources.count; k++) {
				svsl_resource_t *tex = &s.prog->resources.items[k];
				if (tex->kind == svsl_res_texture && svsl_str_eq(tex->name, tex_name) && tex->bind.slot >= 0) {
					res->bind.slot  = tex->bind.slot;
					res->bind.space = tex->bind.space;
					paired          = true;
					break;
				}
			}
			if (paired) continue;
		}
		res->bind.slot = lowest_free_slot(&s, cls, res->bind.space, hints, &hint_count);
	}

	// texture/sampler pairing by slot number (the _s suffix is convention only)
	for (int32_t i = 0; i < s.prog->resources.count; i++) {
		svsl_resource_t *res = &s.prog->resources.items[i];
		if (res->kind != svsl_res_texture) continue;
		for (int32_t k = 0; k < s.prog->resources.count; k++) {
			const svsl_resource_t *smp = &s.prog->resources.items[k];
			if (smp->kind == svsl_res_sampler && smp->bind.slot == res->bind.slot &&
			    smp->bind.space == res->bind.space) {
				res->sampler_slot = smp->bind.slot;
				break;
			}
		}
	}

	// //-- metadata merge
	if (pp) {
		const svsl_pp_meta_t *name_meta = meta_find(pp, "name");
		if (name_meta) {
			s.prog->name           = name_meta->value;
			s.prog->name_from_meta = true;
		}

		const svsl_pp_meta_t *wave = meta_find(pp, "wave_size");
		if (wave) {
			int64_t v = strtol(svsl_arena_strndup(arena, wave->value.ptr, (size_t)wave->value.len), NULL, 10);
			if (v >= 4 && v <= 128 && (v & (v - 1)) == 0) s.prog->wave_size = (int32_t)v;
			else svsl_diag_add(arena, ref_diags, svsl_severity_error, wave->loc,
			                   "//--wave_size must be a power of two in [4,128]");
		}

		// //--apron = W[, H]: VK_QCOM_tile_shading render pass tileApronSize.
		// Purely renderer-facing — the apron has no shader-side representation
		// (shaders only *read* the active size via tile_apron_size_qcom())
		const svsl_pp_meta_t *apron = meta_find(pp, "apron");
		if (apron) {
			char *text = svsl_arena_strndup(arena, apron->value.ptr, (size_t)apron->value.len);
			char *end  = NULL;
			int64_t w = strtol(text, &end, 10), h = w;
			while (end && (*end == ' ' || *end == '\t')) end++;
			if (end && *end == ',') h = strtol(end + 1, &end, 10);
			while (end && (*end == ' ' || *end == '\t')) end++;
			if (apron->value.len == 0 || !end || *end != '\0' || w < 0 || h < 0) {
				svsl_diag_add(arena, ref_diags, svsl_severity_error, apron->loc,
				              "//--apron must be a size in pixels: 'W' or 'W, H'");
			} else {
				s.prog->tile_apron[0] = (int32_t)w;
				s.prog->tile_apron[1] = (int32_t)h;
				bool tiled = false; // warn when nothing tile-shaped consumes it
				for (int32_t r = 0; r < s.prog->resources.count; r++)
					if (s.prog->resources.items[r].tile_attachment) tiled = true;
				for (int32_t en = 0; en < s.prog->entries.count; en++)
					if (s.prog->entries.items[en].tile_rate[0] > 0 ||
					    s.prog->entries.items[en].non_coherent_tile_reads) tiled = true;
				if (!tiled)
					svsl_diag_add(arena, ref_diags, svsl_severity_warning, apron->loc,
					              "//--apron set, but no tile attachment or tile-shading attribute in this shader");
			}
		}

		for (int32_t i = 0; i < pp->meta_count; i++) {
			const svsl_pp_meta_t *meta = &pp->metas[i];
			if (svsl_str_eq_cstr(meta->name, "name") || svsl_str_eq_cstr(meta->name, "wave_size") ||
			    svsl_str_eq_cstr(meta->name, "apron")) continue;

			bool applied = false;
			for (int32_t b = 0; b < s.prog->buffers.count && !applied; b++) {
				svsl_buffer_t *buf = &s.prog->buffers.items[b];
				for (int32_t m = 0; m < buf->members.count; m++) {
					svsl_buf_member_t *member = &buf->members.items[m];
					if (!svsl_str_eq(member->name, meta->name)) continue;
					if (meta->tag.len) member->extra = meta->tag;
					if (meta->value.len) {
						if (!buf->defaults) buf->defaults = svsl_arena_alloc(arena, buf->size ? buf->size : 1);
						const svsl_type_t *mt = svsl_type_get(&s.prog->types, member->type);
						if (!parse_value_list(meta->value, mt, buf->defaults + member->offset))
							svsl_diag_add(arena, ref_diags, svsl_severity_warning, meta->loc,
							              "cannot parse //-- default for '%.*s'", meta->name.len, meta->name.ptr);
					}
					applied = true;
					break;
				}
			}
			for (int32_t r = 0; r < s.prog->resources.count && !applied; r++) {
				svsl_resource_t *res = &s.prog->resources.items[r];
				if (!svsl_str_eq(res->name, meta->name)) continue;
				if (meta->value.len) res->value = meta->value;
				if (meta->tag.len)   res->tags  = meta->tag;
				applied = true;
			}
			if (!applied)
				svsl_diag_add(arena, ref_diags, svsl_severity_warning, meta->loc,
				              "//--%.*s does not match any parameter or resource", meta->name.len, meta->name.ptr);
		}
	}
	if (s.prog->name.len == 0) { // fall back to the filename without directories/extension
		svsl_str_t name = svsl_str(opt_filename ? opt_filename : "shader");
		for (int32_t i = name.len - 1; i >= 0; i--)
			if (name.ptr[i] == '/' || name.ptr[i] == '\\') { name = svsl_str_slice(name, i + 1, name.len); break; }
		int32_t dot = svsl_str_find_char(name, '.');
		if (dot > 0) name = svsl_str_slice(name, 0, dot);
		s.prog->name = name;
	}

	// function table with resolved signatures; forward declarations unify with
	// their definitions so recursion detection sees one node per function
	for (int32_t i = 0; i < ast->decl_count; i++) {
		if (ast->decls[i]->kind != svsl_decl_func) continue;
		const svsl_ast_func_t *func = &ast->decls[i]->func;
		svsl_func_info_t info = {
			.func        = func,
			.return_type = resolve_type(&s, func->return_type),
			.param_types = svsl_arena_alloc(arena, (size_t)(func->param_count > 0 ? func->param_count : 1) * sizeof(svsl_type_id_t)) };
		for (int32_t p = 0; p < func->param_count; p++)
			info.param_types[p] = resolve_type(&s, func->params[p]->type);

		svsl_func_info_t *existing = NULL;
		for (int32_t k = 0; k < s.prog->functions.count; k++) {
			svsl_func_info_t *other = &s.prog->functions.items[k];
			if (!svsl_str_eq(other->func->name, func->name) ||
			    other->func->param_count != func->param_count) continue;
			bool same = other->return_type == info.return_type;
			for (int32_t p = 0; same && p < func->param_count; p++)
				same = other->param_types[p] == info.param_types[p];
			if (same) { existing = other; break; }
		}
		if (existing) {
			if (existing->func->body && func->body)
				err(&s, func->loc, "duplicate definition of '%.*s'", func->name);
			else if (func->body)
				existing->func = func; // definition replaces the forward declaration
			continue;
		}
		svsl_array_push(arena, &s.prog->functions, info);
	}

	discover_entries(&s, opt_options);
	svsl_check_functions(arena, s.prog, ref_diags);
	return ref_diags->error_count == errors_before;
}

// type resolution for the body checker (locals, casts, constructors)
svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *arena, svsl_program_t *prog,
                                      svsl_diag_list_t *diags, const svsl_ast_type_t *ref) {
	sema_t s = { .arena = arena, .prog = prog, .diags = diags, .ast = prog->ast };
	return resolve_type(&s, ref);
}

// array-size inference for local declarations: float3 x[] = {...}
svsl_type_id_t svsl_sema_infer_array_size(svsl_arena_t *arena, svsl_program_t *prog,
                                          svsl_diag_list_t *diags, svsl_type_id_t type,
                                          const svsl_ast_var_t *var) {
	sema_t s = { .arena = arena, .prog = prog, .diags = diags, .ast = prog->ast };
	return infer_array_size(&s, type, var);
}

// Folds a constant integer expression (literals, arithmetic, and named constants:
// enum members, static-const ints, spec-const defaults). For switch case labels.
bool svsl_sema_const_eval_int(svsl_program_t *prog, const svsl_ast_expr_t *e, int64_t *out) {
	sema_t s = { .prog = prog };
	return const_eval_int(&s, e, out);
}
