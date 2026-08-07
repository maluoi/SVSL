// AST → IR lowering with mandatory full inlining. Rvalues that root in storage
// lower as pointer chains + one load (never load-whole-then-extract), which is
// the structural fix for the prototype's over-fetch bug. Opaque parameters
// resolve to their caller's resource index at inline time.

#include "ir.h"
#include "ir_operands.h"

#include "../tables/intrinsics.h"

#include <string.h>

typedef svsl_array_t(uint32_t) u32_list_t;

#define IR_MAX_INLINE_DEPTH 64
#define RES_MARK 0x80000000u // param_vars entry: opaque param → resource index

typedef struct lower_ctx_t lower_ctx_t;
struct lower_ctx_t {
	lower_ctx_t            *parent;
	const svsl_ast_func_t  *func;
	const svsl_func_info_t *info;
	uint32_t               *param_vars;  // var ids, or RES_MARK|resource_index
	uint32_t                result_var;  // SVSL_IR_NONE for void
	uint32_t                returned_var;// bool flag for early returns inside loops/switches; NONE if unused
	bool                    tail_return; // single return as last statement: no wrapper loop
};

typedef struct build_t {
	svsl_arena_t     *arena;
	svsl_program_t   *prog;
	svsl_diag_list_t *diags;
	svsl_ir_func_t   *fn;
	const svsl_func_info_t *entry_info;
	lower_ctx_t      *ctx;   // NULL at entry-function level
	int32_t           depth;
	u32_list_t        local_stack;      // mirrors the checker's scope stack
	bool              failed;
} build_t;

static void berr(build_t *b, svsl_loc_t loc, const char *msg) {
	svsl_diag_add(b->arena, b->diags, svsl_severity_error, loc, "%s", msg);
	b->failed = true;
}

// --- emission helpers -----------------------------------------------------------

static uint32_t emit(build_t *b, svsl_ir_inst_t inst) {
	svsl_array_push(b->arena, &b->fn->insts, inst);
	return (uint32_t)(b->fn->insts.count - 1);
}
static uint32_t emit_op(build_t *b, svsl_ir_op_ op, svsl_type_id_t type,
                        uint32_t a0, uint32_t a1, uint32_t a2, svsl_loc_t loc) {
	return emit(b, (svsl_ir_inst_t){ .op = (uint8_t)op, .type = type,
	                                 .args = { a0, a1, a2, SVSL_IR_NONE }, .loc = loc });
}
static uint32_t aux_push(build_t *b, const uint32_t *values, uint32_t count) {
	uint32_t offset = (uint32_t)b->fn->aux.count;
	for (uint32_t i = 0; i < count; i++)
		svsl_array_push(b->arena, &b->fn->aux, values[i]);
	return offset;
}

static uint32_t emit_const_bits(build_t *b, svsl_type_id_t type, uint64_t bits, svsl_loc_t loc) {
	return emit_op(b, svsl_ir_const, type, (uint32_t)bits, (uint32_t)(bits >> 32), 0, loc);
}

// single-operand call to a table intrinsic by its native name
static uint32_t emit_intr1(build_t *b, const char *name, svsl_type_id_t type,
                           uint32_t arg, svsl_loc_t loc) {
	int32_t  index = svsl_intrinsic_find(svsl_str(name));
	uint32_t aux   = aux_push(b, &arg, 1);
	return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_intrinsic, .type = type,
	                                 .args = { (uint32_t)index, 0, 0, SVSL_IR_NONE },
	                                 .aux = aux, .aux_count = 1, .loc = loc });
}
static uint32_t emit_const_int(build_t *b, svsl_scalar_ scalar, int64_t v, svsl_loc_t loc) {
	svsl_type_id_t type = svsl_type_scalar_id(&b->prog->types, scalar);
	return emit_const_bits(b, type, (uint64_t)v, loc);
}
static uint32_t emit_const_float(build_t *b, svsl_scalar_ scalar, double v, svsl_loc_t loc) {
	svsl_type_id_t type = svsl_type_scalar_id(&b->prog->types, scalar);
	scalar = svsl_type_get(&b->prog->types, type)->scalar; // strict16 re-types half
	if (scalar == svsl_scalar_float64) {
		uint64_t bits;
		memcpy(&bits, &v, 8);
		return emit_const_bits(b, type, bits, loc);
	}
	if (scalar == svsl_scalar_float16)
		return emit_const_bits(b, type, svsl_f32_to_f16_bits((float)v), loc);
	float    f = (float)v;
	uint32_t bits;
	memcpy(&bits, &f, 4);
	return emit_const_bits(b, type, bits, loc);
}

static svsl_type_id_t value_type(const build_t *b, uint32_t id) {
	return b->fn->insts.items[id].type;
}

// zero value of any constructible type ((psIn)0, missing defaults)
static uint32_t emit_zero(build_t *b, svsl_type_id_t type, svsl_loc_t loc) {
	const svsl_type_t *t = svsl_type_get(&b->prog->types, type);
	switch (t->kind) {
	case svsl_type_scalar:
		return emit_const_bits(b, type, 0, loc);
	case svsl_type_vector:
	case svsl_type_matrix:
	case svsl_type_array:
	case svsl_type_struct: {
		const svsl_struct_info_t *info = t->kind == svsl_type_struct ?
			&b->prog->types.structs.items[t->struct_index] : NULL;
		int32_t count = t->kind == svsl_type_vector ? t->count :
		                t->kind == svsl_type_matrix ? t->rows  : // SPIR-V matrix vectors are HLSL rows
		                t->kind == svsl_type_array  ? t->array_count : (int32_t)info->members.count;
		uint32_t *parts = svsl_arena_alloc(b->arena, (size_t)count * sizeof(uint32_t));
		if (t->kind == svsl_type_vector) {
			uint32_t z = emit_zero(b, svsl_type_scalar_id(&b->prog->types, t->scalar), loc);
			for (int32_t i = 0; i < count; i++) parts[i] = z;
		} else if (t->kind == svsl_type_matrix) {
			uint32_t z = emit_zero(b, svsl_type_vector_id(&b->prog->types, t->scalar, t->cols), loc);
			for (int32_t i = 0; i < count; i++) parts[i] = z;
		} else if (t->kind == svsl_type_array) {
			uint32_t z = emit_zero(b, t->elem, loc);
			for (int32_t i = 0; i < count; i++) parts[i] = z;
		} else {
			for (int32_t i = 0; i < count; i++)
				parts[i] = emit_zero(b, info->members.items[i].type, loc);
		}
		uint32_t aux = aux_push(b, parts, (uint32_t)count);
		return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_construct, .type = type,
		                                 .args = { 0, 0, 0, SVSL_IR_NONE },
		                                 .aux = aux, .aux_count = (uint32_t)count, .loc = loc });
	}
	case svsl_type_void:
	case svsl_type_texture:
	case svsl_type_image:
	case svsl_type_sampler:
	case svsl_type_subpass:
	case svsl_type_buffer:
	case svsl_type_tileimage:
		return emit_op(b, svsl_ir_undef, type, 0, 0, 0, loc); // no zero for handles
	}
	return emit_op(b, svsl_ir_undef, type, 0, 0, 0, loc);
}

// --- conversions ------------------------------------------------------------------

static uint32_t convert_value(build_t *b, uint32_t value, svsl_type_id_t to, svsl_loc_t loc) {
	if (value == SVSL_IR_NONE) return value; // void call results
	svsl_type_id_t from = value_type(b, value);
	if (from == to || to == SVSL_TYPE_NONE || from == SVSL_TYPE_NONE) return value;
	svsl_types_t      *types = &b->prog->types;
	const svsl_type_t *f     = svsl_type_get(types, from);
	const svsl_type_t *t     = svsl_type_get(types, to);

	int32_t fc = f->kind == svsl_type_vector ? f->count : f->kind == svsl_type_scalar ? 1 : 0;
	int32_t tc = t->kind == svsl_type_vector ? t->count : t->kind == svsl_type_scalar ? 1 : 0;
	if (fc == 0 || tc == 0) return value; // shape conversions only apply to scalars/vectors

	// scalar component conversion first (on the source shape)
	if (f->scalar != t->scalar) {
		svsl_type_id_t mid = fc == 1 ? svsl_type_scalar_id(types, t->scalar)
		                             : svsl_type_vector_id(types, t->scalar, fc);
		value = emit_op(b, svsl_ir_convert, mid, value, 0, 0, loc);
		from  = mid;
	}
	if (fc == tc) return value;
	if (fc == 1) { // splat
		uint32_t parts[4] = { value, value, value, value };
		uint32_t aux      = aux_push(b, parts, (uint32_t)tc);
		return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_construct, .type = to,
		                                 .args = { 0, 0, 0, SVSL_IR_NONE },
		                                 .aux = aux, .aux_count = (uint32_t)tc, .loc = loc });
	}
	if (tc < fc) { // truncate: leading components
		if (tc == 1)
			return emit_op(b, svsl_ir_extract, to, value, 0, 0, loc);
		return emit_op(b, svsl_ir_shuffle, to, value, 0x3210 & ((1 << (tc * 4)) - 1), (uint32_t)tc, loc);
	}
	return value;
}

// --- expressions -------------------------------------------------------------------

static uint32_t lower_expr(build_t *b, const svsl_ast_expr_t *e);
static void     lower_stmt(build_t *b, const svsl_ast_stmt_t *s);

// resolves an expression naming a resource (texture/sampler/etc.) to its index,
// looking through inlined opaque parameters
static int32_t resolve_resource(build_t *b, const svsl_ast_expr_t *e) {
	if (e->kind == svsl_expr_ident) {
		if (e->sema_ref.kind == svsl_ref_resource) return e->sema_ref.a;
		if (e->sema_ref.kind == svsl_ref_param && b->ctx) {
			uint32_t v = b->ctx->param_vars[e->sema_ref.a];
			if (v & RES_MARK) return (int32_t)(v & ~RES_MARK);
		}
	}
	berr(b, e->loc, "expected a resource here");
	return 0;
}


// natural component/element type of a pointee, from declared types only
static svsl_type_id_t pointee_elem(build_t *b, svsl_type_id_t base, int32_t member_index) {
	const svsl_type_t *t = svsl_type_get(&b->prog->types, base);
	switch (t->kind) {
	case svsl_type_struct:
		return b->prog->types.structs.items[t->struct_index].members.items[member_index].type;
	case svsl_type_array:  return t->elem;
	case svsl_type_buffer: return t->elem;
	case svsl_type_vector: return svsl_type_scalar_id(&b->prog->types, t->scalar);
	case svsl_type_matrix: return svsl_type_vector_id(&b->prog->types, t->scalar, t->cols); // row
	case svsl_type_void:
	case svsl_type_scalar:
	case svsl_type_texture:
	case svsl_type_image:
	case svsl_type_sampler:
	case svsl_type_subpass:
	case svsl_type_tileimage:
		return base; // not element-addressable; chains never descend into these
	}
	return base;
}

// appends indices to a pointer, folding chain-of-chain into one flat chain
// (one OpAccessChain per access path — the over-fetch fix, structurally)
static uint32_t emit_chain(build_t *b, uint32_t base, const uint32_t *indices, uint32_t count,
                           svsl_type_id_t type, svsl_loc_t loc) {
	uint32_t merged[16];
	uint32_t total = 0;
	const svsl_ir_inst_t *bi = &b->fn->insts.items[base];
	if (bi->op == svsl_ir_chain) {
		for (uint32_t i = 0; i < bi->aux_count && total < 16; i++)
			merged[total++] = b->fn->aux.items[bi->aux + i];
		base = bi->args[0];
	}
	for (uint32_t i = 0; i < count && total < 16; i++)
		merged[total++] = indices[i];
	uint32_t aux = aux_push(b, merged, total);
	return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_chain, .type = type,
	                                 .args = { base, 0, 0, SVSL_IR_NONE },
	                                 .aux = aux, .aux_count = total, .loc = loc });
}

// pointer-producing lowering; returns SVSL_IR_NONE when the expression has no
// storage (call results, arithmetic, …).
static uint32_t lower_lvalue(build_t *b, const svsl_ast_expr_t *e) {
	switch (e->kind) {
	case svsl_expr_ident:
		switch (e->sema_ref.kind) {
		case svsl_ref_local:
			return b->local_stack.items[e->sema_ref.a];
		case svsl_ref_param:
			if (b->ctx) {
				uint32_t v = b->ctx->param_vars[e->sema_ref.a];
				return (v & RES_MARK) ? SVSL_IR_NONE : v;
			}
			for (int32_t i = 0; i < b->fn->insts.count; i++)
				if (b->fn->insts.items[i].op == svsl_ir_param &&
				    b->fn->insts.items[i].args[0] == (uint32_t)e->sema_ref.a)
					return (uint32_t)i;
			return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_param,
			                                 .type = b->entry_info ? b->entry_info->param_types[e->sema_ref.a]
			                                                       : e->sema_type,
			                                 .args = { (uint32_t)e->sema_ref.a, 0, 0, SVSL_IR_NONE },
			                                 .loc = e->loc, .name = e->ident });
		case svsl_ref_resource: {
			const svsl_resource_t *res = &b->prog->resources.items[e->sema_ref.a];
			const svsl_type_t     *rt  = svsl_type_get(&b->prog->types, res->type);
			if (rt->kind != svsl_type_buffer) return SVSL_IR_NONE;
			return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_ptr, .type = res->type,
			                                 .args = { e->sema_ref.kind, (uint32_t)e->sema_ref.a,
			                                           0, SVSL_IR_NONE },
			                                 .loc = e->loc, .name = e->ident });
		}
		case svsl_ref_buffer_member: {
			svsl_type_id_t natural =
				b->prog->buffers.items[e->sema_ref.a].members.items[e->sema_ref.b].type;
			return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_ptr, .type = natural,
			                                 .args = { e->sema_ref.kind, (uint32_t)e->sema_ref.a,
			                                           (uint32_t)e->sema_ref.b, SVSL_IR_NONE },
			                                 .loc = e->loc, .name = e->ident });
		}
		case svsl_ref_workgroup:
		case svsl_ref_const_global: {
			svsl_type_id_t natural = e->sema_ref.kind == svsl_ref_workgroup
			                       ? b->prog->workgroup_vars.items[e->sema_ref.a].type
			                       : b->prog->const_globals.items[e->sema_ref.a].type;
			return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_ptr, .type = natural,
			                                 .args = { e->sema_ref.kind, (uint32_t)e->sema_ref.a,
			                                           (uint32_t)e->sema_ref.b, SVSL_IR_NONE },
			                                 .loc = e->loc, .name = e->ident });
		}
		default:
			return SVSL_IR_NONE;
		}
	case svsl_expr_member: {
		if (e->sema_ref.kind == svsl_ref_bitfield)
			return SVSL_IR_NONE; // packed field: no pointer, handled by read/store paths
		if (e->sema_ref.kind == svsl_ref_swizzle && e->sema_ref.b > 1)
			return SVSL_IR_NONE; // multi-component swizzle: handled by assignment
		uint32_t base = lower_lvalue(b, e->member.object);
		if (base == SVSL_IR_NONE) return SVSL_IR_NONE;
		svsl_type_id_t base_type = b->fn->insts.items[base].type;

		if (e->sema_ref.kind == svsl_ref_swizzle) { // single component
			svsl_type_id_t natural = svsl_type_scalar_id(&b->prog->types,
				svsl_type_get(&b->prog->types, base_type)->scalar);
			uint32_t index = emit_const_int(b, svsl_scalar_int32, e->sema_ref.a & 0xF, e->loc);
			return emit_chain(b, base, &index, 1, natural, e->loc);
		}
		if (e->sema_ref.kind == svsl_ref_matrix_elem) {
			svsl_type_id_t natural = svsl_type_scalar_id(&b->prog->types,
				svsl_type_get(&b->prog->types, base_type)->scalar);
			uint32_t indices[2] = { // SPIR-V vector index = HLSL row, then column component
				emit_const_int(b, svsl_scalar_int32, e->sema_ref.a, e->loc),
				emit_const_int(b, svsl_scalar_int32, e->sema_ref.b, e->loc) };
			return emit_chain(b, base, indices, 2, natural, e->loc);
		}
		// struct member: sema_ref.a = member index
		svsl_type_id_t natural = pointee_elem(b, base_type, e->sema_ref.a);
		uint32_t index = emit_const_int(b, svsl_scalar_int32, e->sema_ref.a, e->loc);
		return emit_chain(b, base, &index, 1, natural, e->loc);
	}
	case svsl_expr_index: {
		const svsl_type_t *obj = svsl_type_get(&b->prog->types, e->index.object->sema_type);
		if (obj->kind == svsl_type_image) return SVSL_IR_NONE; // image stores are special-cased
		uint32_t base = lower_lvalue(b, e->index.object);
		if (base == SVSL_IR_NONE) return SVSL_IR_NONE;
		svsl_type_id_t natural = pointee_elem(b, b->fn->insts.items[base].type, 0);
		uint32_t index = lower_expr(b, e->index.index);
		return emit_chain(b, base, &index, 1, natural, e->loc);
	}
	default:
		return SVSL_IR_NONE;
	}
}

// value = raw lowering result; applies the conversion sema recorded on the node
static uint32_t finish(build_t *b, const svsl_ast_expr_t *e, uint32_t value) {
	return convert_value(b, value, e->sema_type, e->loc);
}

// --- packed struct bit fields ------------------------------------------------------
// A field has no pointer of its own; access is arithmetic on the backing uint32
// word. Reads extract + decode; writes encode + insert + read-modify-write.

static const svsl_field_t *bitfield_of(build_t *b, const svsl_ast_expr_t *e) {
	return &b->prog->types.structs.items[e->sema_ref.a].fields.items[e->sema_ref.b];
}

static uint32_t bitfield_word_ptr(build_t *b, uint32_t base, const svsl_field_t *f, svsl_loc_t loc) {
	svsl_type_id_t uint_t = svsl_type_scalar_id(&b->prog->types, svsl_scalar_uint32);
	uint32_t widx = emit_const_int(b, svsl_scalar_int32, f->bit_offset / 32, loc);
	return emit_chain(b, base, &widx, 1, uint_t, loc);
}

// decode extracted bits `raw` (int32 for signed extract, else uint32) into `e`'s
// resolve type, per the field's format
static uint32_t bitfield_decode(build_t *b, const svsl_ast_expr_t *e, const svsl_field_t *f, uint32_t raw) {
	svsl_types_t  *types = &b->prog->types;
	svsl_type_id_t f32_t = svsl_type_scalar_id(types, svsl_scalar_float32);
	svsl_scalar_   rsc   = svsl_type_get(types, f->type)->scalar;

	if (f->bit_format == svsl_bitfmt_unorm || f->bit_format == svsl_bitfmt_snorm) {
		double   scale = f->bit_format == svsl_bitfmt_unorm ? (double)((1u << f->bit_width) - 1)
		                                                    : (double)((1u << (f->bit_width - 1)) - 1);
		uint32_t fv = emit_op(b, svsl_ir_convert, f32_t, raw, 0, 0, e->loc);
		uint32_t k  = emit_const_float(b, svsl_scalar_float32, 1.0 / scale, e->loc);
		return finish(b, e, emit_op(b, svsl_ir_mul, f32_t, fv, k, 0, e->loc));
	}
	if (f->bit_format == svsl_bitfmt_raw && (rsc == svsl_scalar_float16 || rsc == svsl_scalar_half))
		return finish(b, e, emit_intr1(b, "f16tof32", f32_t, raw, e->loc));
	if (f->bit_format == svsl_bitfmt_raw && rsc == svsl_scalar_float32)
		return finish(b, e, emit_intr1(b, "asfloat", f32_t, raw, e->loc)); // reinterpret the word
	if (rsc == svsl_scalar_bool) {
		uint32_t zero = emit_const_int(b, svsl_scalar_uint32, 0, e->loc);
		return finish(b, e, emit_op(b, svsl_ir_ne,
		                            svsl_type_scalar_id(types, svsl_scalar_bool), raw, zero, 0, e->loc));
	}
	return finish(b, e, raw); // raw integer: finish converts to the resolve int type
}

static uint32_t lower_bitfield_read(build_t *b, const svsl_ast_expr_t *e) {
	const svsl_field_t *f    = bitfield_of(b, e);
	uint32_t            base = lower_lvalue(b, e->member.object);
	if (base == SVSL_IR_NONE) { berr(b, e->loc, "cannot read this packed field"); return 0; }
	svsl_types_t  *types  = &b->prog->types;
	svsl_type_id_t uint_t = svsl_type_scalar_id(types, svsl_scalar_uint32);
	svsl_type_id_t int_t  = svsl_type_scalar_id(types, svsl_scalar_int32);

	uint32_t word = emit_op(b, svsl_ir_load, uint_t, bitfield_word_ptr(b, base, f, e->loc), 0, 0, e->loc);
	uint32_t off  = emit_const_int(b, svsl_scalar_uint32, f->bit_offset % 32, e->loc);
	uint32_t wid  = emit_const_int(b, svsl_scalar_uint32, f->bit_width, e->loc);
	svsl_scalar_ rsc = svsl_type_get(types, f->type)->scalar;
	bool sext = f->bit_format == svsl_bitfmt_snorm ||
	            (f->bit_format == svsl_bitfmt_raw && rsc >= svsl_scalar_int8 && rsc <= svsl_scalar_int64);
	// OpBitFieldSExtract requires base and result to share type: bitcast the word to int
	uint32_t src = sext ? emit_op(b, svsl_ir_convert, int_t, word, 0, 0, e->loc) : word;
	uint32_t raw = emit_op(b, svsl_ir_bitfield_extract, sext ? int_t : uint_t, src, off, wid, e->loc);
	return bitfield_decode(b, e, f, raw);
}

// encode a resolve-typed value into the low `width` bits to insert (truncating)
static uint32_t bitfield_encode(build_t *b, const svsl_field_t *f, uint32_t v, svsl_loc_t loc) {
	svsl_types_t  *types  = &b->prog->types;
	svsl_type_id_t uint_t = svsl_type_scalar_id(types, svsl_scalar_uint32);
	svsl_type_id_t int_t  = svsl_type_scalar_id(types, svsl_scalar_int32);
	svsl_type_id_t f32_t  = svsl_type_scalar_id(types, svsl_scalar_float32);
	svsl_scalar_   rsc    = svsl_type_get(types, f->type)->scalar;

	if (f->bit_format == svsl_bitfmt_unorm) {
		double   scale = (double)((1u << f->bit_width) - 1);
		uint32_t sv = emit_op(b, svsl_ir_mul, f32_t, convert_value(b, v, f32_t, loc),
		                      emit_const_float(b, svsl_scalar_float32, scale, loc), 0, loc);
		return emit_op(b, svsl_ir_convert, uint_t, sv, 0, 0, loc); // FToU truncates
	}
	if (f->bit_format == svsl_bitfmt_snorm) {
		double   scale = (double)((1u << (f->bit_width - 1)) - 1);
		uint32_t sv = emit_op(b, svsl_ir_mul, f32_t, convert_value(b, v, f32_t, loc),
		                      emit_const_float(b, svsl_scalar_float32, scale, loc), 0, loc);
		uint32_t iv = emit_op(b, svsl_ir_convert, int_t, sv, 0, 0, loc); // FToS truncates toward 0
		return emit_op(b, svsl_ir_convert, uint_t, iv, 0, 0, loc);       // bitcast to uint bits
	}
	if (f->bit_format == svsl_bitfmt_raw && (rsc == svsl_scalar_float16 || rsc == svsl_scalar_half))
		return emit_intr1(b, "f32tof16", uint_t, convert_value(b, v, f32_t, loc), loc);
	if (f->bit_format == svsl_bitfmt_raw && rsc == svsl_scalar_float32)
		return emit_intr1(b, "asuint", uint_t, convert_value(b, v, f32_t, loc), loc); // float bits → word
	if (rsc == svsl_scalar_bool) {
		uint32_t one  = emit_const_int(b, svsl_scalar_uint32, 1, loc);
		uint32_t zero = emit_const_int(b, svsl_scalar_uint32, 0, loc);
		return emit_op(b, svsl_ir_select, uint_t, v, one, zero, loc);
	}
	return convert_value(b, v, uint_t, loc); // raw integer: low bits preserved, insert masks
}

// read-modify-write a packed field: *word = insert(*word, encode(value))
static void bitfield_store(build_t *b, const svsl_ast_expr_t *e, uint32_t value, svsl_loc_t loc) {
	const svsl_field_t *f    = bitfield_of(b, e);
	uint32_t            base = lower_lvalue(b, e->member.object);
	if (base == SVSL_IR_NONE) { berr(b, loc, "cannot store this packed field"); return; }
	svsl_type_id_t uint_t = svsl_type_scalar_id(&b->prog->types, svsl_scalar_uint32);
	uint32_t wptr = bitfield_word_ptr(b, base, f, loc);
	uint32_t word = emit_op(b, svsl_ir_load, uint_t, wptr, 0, 0, loc);
	uint32_t raw  = bitfield_encode(b, f, value, loc);
	uint32_t off  = emit_const_int(b, svsl_scalar_uint32, f->bit_offset % 32, loc);
	uint32_t wid  = emit_const_int(b, svsl_scalar_uint32, f->bit_width, loc);
	uint32_t nw   = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_bitfield_insert, .type = uint_t,
	                                          .args = { word, raw, off, wid }, .loc = loc });
	emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, wptr, nw, 0, loc);
}

static uint32_t lower_binary(build_t *b, const svsl_ast_expr_t *e);
static uint32_t lower_call(build_t *b, const svsl_ast_expr_t *e);
static uint32_t lower_ctor(build_t *b, const svsl_ast_expr_t *e);
static uint32_t lower_init(build_t *b, const svsl_ast_expr_t *e, svsl_type_id_t type);

static svsl_ir_op_ binop_ir(svsl_tok_ op) {
	switch (op) {
	case svsl_tok_plus:  case svsl_tok_plus_assign:    return svsl_ir_add;
	case svsl_tok_minus: case svsl_tok_minus_assign:   return svsl_ir_sub;
	case svsl_tok_star:  case svsl_tok_star_assign:    return svsl_ir_mul;
	case svsl_tok_slash: case svsl_tok_slash_assign:   return svsl_ir_div;
	case svsl_tok_percent: case svsl_tok_percent_assign: return svsl_ir_rem;
	case svsl_tok_amp:   case svsl_tok_and_assign:     return svsl_ir_bit_and;
	case svsl_tok_pipe:  case svsl_tok_or_assign:      return svsl_ir_bit_or;
	case svsl_tok_caret: case svsl_tok_xor_assign:     return svsl_ir_bit_xor;
	case svsl_tok_shl:   case svsl_tok_shl_assign:     return svsl_ir_shl;
	case svsl_tok_shr:   case svsl_tok_shr_assign:     return svsl_ir_shr;
	case svsl_tok_eq:     return svsl_ir_eq;
	case svsl_tok_neq:    return svsl_ir_ne;
	case svsl_tok_lt:     return svsl_ir_lt;
	case svsl_tok_le:     return svsl_ir_le;
	case svsl_tok_gt:     return svsl_ir_gt;
	case svsl_tok_ge:     return svsl_ir_ge;
	case svsl_tok_andand: return svsl_ir_log_and;
	case svsl_tok_oror:   return svsl_ir_log_or;
	default:              return svsl_ir_nop;
	}
}

// An assignment target with its lvalue chain evaluated exactly once. Compound
// assignment and increment load AND store through the same resolved target, so
// side effects in the chain run once: coeffs[idx++] *= s must not re-run idx++
// between its load and its store.
typedef struct target_t {
	uint32_t               ptr;     // pointer chain, or SVSL_IR_NONE for images
	int32_t                res;     // image resource when ptr is none
	uint32_t               coord;   // image coordinate when ptr is none
	const svsl_ast_expr_t *swizzle;  // multi-component swizzle applied on *ptr, or NULL
	const svsl_ast_expr_t *bitfield; // packed struct field: read-modify-write, or NULL
} target_t;

static bool target_resolve(build_t *b, const svsl_ast_expr_t *target, target_t *out) {
	*out = (target_t){ .ptr = SVSL_IR_NONE, .res = -1 };
	// packed struct bit field: no pointer, read-modify-write on the backing word
	if (target->kind == svsl_expr_member && target->sema_ref.kind == svsl_ref_bitfield) {
		out->bitfield = target;
		return true;
	}
	// image[coord]
	if (target->kind == svsl_expr_index) {
		const svsl_type_t *obj = svsl_type_get(&b->prog->types, target->index.object->sema_type);
		if (obj->kind == svsl_type_image) {
			out->res   = resolve_resource(b, target->index.object);
			out->coord = lower_expr(b, target->index.index);
			return true;
		}
	}
	// multi-component swizzle: address the whole vector, remember the swizzle
	if (target->kind == svsl_expr_member && target->sema_ref.kind == svsl_ref_swizzle &&
	    target->sema_ref.b > 1) {
		out->ptr     = lower_lvalue(b, target->member.object);
		out->swizzle = target;
		return out->ptr != SVSL_IR_NONE;
	}
	out->ptr = lower_lvalue(b, target);
	return out->ptr != SVSL_IR_NONE;
}

// current value of a resolved target, for compound assignment and increment
static uint32_t target_load(build_t *b, const target_t *t, svsl_type_id_t type, svsl_loc_t loc) {
	if (t->bitfield)
		return lower_bitfield_read(b, t->bitfield);
	if (t->ptr == SVSL_IR_NONE)
		return emit_op(b, svsl_ir_image_load, type, (uint32_t)t->res, t->coord, 0, loc);
	if (t->swizzle) {
		const svsl_ast_expr_t *sw       = t->swizzle;
		svsl_type_id_t         vec_type = sw->member.object->sema_type;
		uint32_t whole = emit_op(b, svsl_ir_load, vec_type, t->ptr, 0, 0, loc);
		return emit_op(b, svsl_ir_shuffle,
		               svsl_type_vector_id(&b->prog->types,
		                                   svsl_type_get(&b->prog->types, vec_type)->scalar,
		                                   sw->sema_ref.b),
		               whole, (uint32_t)sw->sema_ref.a, (uint32_t)sw->sema_ref.b, loc);
	}
	return emit_op(b, svsl_ir_load, b->fn->insts.items[t->ptr].type, t->ptr, 0, 0, loc);
}

static void target_store(build_t *b, const target_t *t, uint32_t value, svsl_loc_t loc) {
	if (t->bitfield) {
		bitfield_store(b, t->bitfield, value, loc);
		return;
	}
	if (t->ptr == SVSL_IR_NONE) {
		emit_op(b, svsl_ir_image_store, SVSL_TYPE_NONE, (uint32_t)t->res, t->coord, value, loc);
		return;
	}
	// multi-component swizzle write: load vector, insert components, store back
	if (t->swizzle) {
		const svsl_ast_expr_t *sw       = t->swizzle;
		svsl_type_id_t         vec_type = sw->member.object->sema_type;
		uint32_t whole = emit_op(b, svsl_ir_load, vec_type, t->ptr, 0, 0, loc);
		for (int32_t i = 0; i < sw->sema_ref.b; i++) {
			int32_t  component = (sw->sema_ref.a >> (i * 4)) & 0xF;
			uint32_t part      = emit_op(b, svsl_ir_extract,
			                             svsl_type_scalar_id(&b->prog->types,
			                                                 svsl_type_get(&b->prog->types, vec_type)->scalar),
			                             value, (uint32_t)i, 0, loc);
			whole = emit_op(b, svsl_ir_insert, vec_type, whole, (uint32_t)component, part, loc);
		}
		emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, t->ptr, whole, 0, loc);
		return;
	}
	emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, t->ptr, value, 0, loc);
}

// stores `value` through a (possibly swizzled) assignment target
static void store_target(build_t *b, const svsl_ast_expr_t *target, uint32_t value) {
	target_t t;
	if (!target_resolve(b, target, &t)) {
		berr(b, target->loc, "cannot store through this expression");
		return;
	}
	target_store(b, &t, value, target->loc);
}

static uint32_t lower_binary(build_t *b, const svsl_ast_expr_t *e) {
	svsl_tok_ op = e->binary.op;

	if (op == svsl_tok_comma) {
		lower_expr(b, e->binary.lhs);
		return lower_expr(b, e->binary.rhs);
	}
	if (op == svsl_tok_assign) {
		uint32_t value = lower_expr(b, e->binary.rhs);
		store_target(b, e->binary.lhs, value);
		return value;
	}
	if (binop_ir(op) != svsl_ir_nop &&
	    (op == svsl_tok_plus_assign || op == svsl_tok_minus_assign || op == svsl_tok_star_assign ||
	     op == svsl_tok_slash_assign || op == svsl_tok_percent_assign || op == svsl_tok_and_assign ||
	     op == svsl_tok_or_assign || op == svsl_tok_xor_assign || op == svsl_tok_shl_assign ||
	     op == svsl_tok_shr_assign)) {
		target_t t;
		if (!target_resolve(b, e->binary.lhs, &t)) {
			berr(b, e->binary.lhs->loc, "cannot store through this expression");
			return SVSL_IR_NONE;
		}
		uint32_t old   = target_load(b, &t, e->binary.lhs->sema_type, e->loc);
		uint32_t rhs   = lower_expr(b, e->binary.rhs);
		rhs = convert_value(b, rhs, value_type(b, old), e->loc);
		uint32_t value = emit_op(b, binop_ir(op), value_type(b, old), old, rhs, 0, e->loc);
		target_store(b, &t, value, e->loc);
		return value;
	}

	uint32_t lhs = lower_expr(b, e->binary.lhs);
	uint32_t rhs = lower_expr(b, e->binary.rhs);

	// the natural result type comes from the operands (sema converted both to a
	// common type); e->sema_type may additionally carry an outer conversion that
	// finish() applies afterwards
	svsl_type_id_t operand_type = value_type(b, lhs);
	rhs = convert_value(b, rhs, operand_type, e->loc);

	svsl_type_id_t result = operand_type;
	if (op == svsl_tok_eq || op == svsl_tok_neq || op == svsl_tok_lt ||
	    op == svsl_tok_gt || op == svsl_tok_le || op == svsl_tok_ge) {
		const svsl_type_t *ot = svsl_type_get(&b->prog->types, operand_type);
		int32_t count = ot->kind == svsl_type_vector ? ot->count : 1;
		result = count == 1 ? svsl_type_scalar_id(&b->prog->types, svsl_scalar_bool)
		                    : svsl_type_vector_id(&b->prog->types, svsl_scalar_bool, count);
	}
	return emit_op(b, binop_ir(op), result, lhs, rhs, 0, e->loc);
}

static uint32_t lower_intrinsic(build_t *b, const svsl_ast_expr_t *e, int32_t index) {
	const svsl_intrinsic_t *intr = svsl_intrinsic_get(index);

	svsl_intr_tag_ tag = (svsl_intr_tag_)intr->tag;

	// mul: shape-directed
	if (tag == svsl_intr_mul) {
		uint32_t a = lower_expr(b, e->call.args[0]);
		uint32_t v = lower_expr(b, e->call.args[1]);
		const svsl_type_t *ta = svsl_type_get(&b->prog->types, value_type(b, a));
		const svsl_type_t *tb = svsl_type_get(&b->prog->types, value_type(b, v));
		svsl_types_t *types = &b->prog->types;
		if (ta->kind == svsl_type_scalar || tb->kind == svsl_type_scalar)
			return emit_op(b, svsl_ir_mul,
			               ta->kind == svsl_type_scalar ? value_type(b, v) : value_type(b, a),
			               a, v, 0, e->loc);
		if (ta->kind == svsl_type_vector && tb->kind == svsl_type_vector) {
			int32_t dot = svsl_intrinsic_find(svsl_str("dot"));
			uint32_t parts[2] = { a, v };
			uint32_t aux = aux_push(b, parts, 2);
			return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_intrinsic,
			                                 .type = svsl_type_scalar_id(types, ta->scalar),
			                                 .args = { (uint32_t)dot, 0, 0, SVSL_IR_NONE },
			                                 .aux = aux, .aux_count = 2, .loc = e->loc });
		}
		svsl_type_id_t result =
			ta->kind == svsl_type_vector ? svsl_type_vector_id(types, tb->scalar, tb->cols) :
			tb->kind == svsl_type_vector ? svsl_type_vector_id(types, ta->scalar, ta->rows) :
			svsl_type_matrix_id(types, ta->scalar, ta->rows, tb->cols);
		return emit_op(b, svsl_ir_mat_mul, result, a, v, 0, e->loc);
	}
	// select(cond, a, b): the same OpSelect the ternary lowers to (sema coerced
	// cond to a bool of the operands' shape; emit splats a scalar cond as needed)
	if (tag == svsl_intr_select) {
		uint32_t cond = lower_expr(b, e->call.args[0]);
		uint32_t a    = lower_expr(b, e->call.args[1]);
		uint32_t v    = lower_expr(b, e->call.args[2]);
		return emit_op(b, svsl_ir_select, value_type(b, a), cond, a, v, e->loc);
	}
	// out-parameter math lowers to existing intrinsics plus stores
	if (tag == svsl_intr_sincos || tag == svsl_intr_modf || tag == svsl_intr_frexp) {
		uint32_t       x = lower_expr(b, e->call.args[0]);
		svsl_type_id_t g = value_type(b, x);
		if (tag == svsl_intr_sincos) {
			store_target(b, e->call.args[1], emit_intr1(b, "sin", g, x, e->loc));
			uint32_t cos_v = emit_intr1(b, "cos", g, x, e->loc);
			store_target(b, e->call.args[2], cos_v);
			return cos_v;
		}
		if (tag == svsl_intr_modf) { // ip = trunc(x), frac = x - ip
			uint32_t ip = emit_intr1(b, "trunc", g, x, e->loc);
			store_target(b, e->call.args[1], ip);
			return emit_op(b, svsl_ir_sub, g, x, ip, 0, e->loc);
		}
		// frexp: mantissa result, exponent out (as x's float type)
		store_target(b, e->call.args[1], emit_intr1(b, "frexp_exp", g, x, e->loc));
		return emit_intr1(b, "frexp_mant", g, x, e->loc);
	}

	// atomics on pointers: the table tag doubles as the IR op code
	if (SVSL_INTR_IS_ATOMIC(tag)) {
		uint32_t atomic_op = SVSL_INTR_ATOMIC_OP(tag);
		bool     cmpxchg   = tag == svsl_intr_atomic_cmpxchg;
		// HLSL's subscript spelling on a storage image, InterlockedAdd(img[c], v):
		// an image has no pointer, so this takes the same image-atomic path as the
		// img.InterlockedAdd(c, v) method form. Sema rejected cmpxchg and an
		// agnostic format here, and the two forms share one atomic op-code space.
		const svsl_ast_expr_t *dst = e->call.args[0];
		if (dst->kind == svsl_expr_index &&
		    svsl_type_get(&b->prog->types, dst->index.object->sema_type)->kind == svsl_type_image) {
			int32_t  res   = resolve_resource(b, dst->index.object);
			uint32_t coord = lower_expr(b, dst->index.index);
			uint32_t value = lower_expr(b, e->call.args[1]);
			uint32_t prior = emit_op(b, svsl_ir_image_atomic, value_type(b, value),
			                         (uint32_t)res, coord, value, e->loc);
			// the third argument means different things per spelling, exactly as on
			// the pointer path below: an out-param on the HLSL alias, a memory-order
			// name on the native one. Sema rejected cmpxchg, so the value is args[1].
			uint32_t order = 0;
			if (!intr->opt_native && e->call.arg_count == 3) {
				int32_t o = svsl_atomic_order(e->call.args[2]->ident);
				if (o > 0) order = (uint32_t)o;
			}
			b->fn->insts.items[prior].args[3] = atomic_op | (order << 8);
			if (intr->opt_native && e->call.arg_count == 3)
				store_target(b, e->call.args[2], prior); // out-param form
			return prior;
		}
		uint32_t ptr     = lower_lvalue(b, e->call.args[0]);
		if (ptr == SVSL_IR_NONE) { berr(b, e->loc, "bad atomic destination"); return SVSL_IR_NONE; }
		uint32_t v1 = lower_expr(b, e->call.args[1]);
		uint32_t v2 = cmpxchg ? lower_expr(b, e->call.args[2]) : 0;
		svsl_type_id_t scalar = value_type(b, v1);
		uint32_t prior = emit_op(b, svsl_ir_atomic, scalar, ptr, v1, v2, e->loc);
		int32_t base_args = cmpxchg ? 3 : 2;
		// a trailing memory-order name (native form only) packs into args[3]'s high
		// byte; the op code stays in the low byte. Scope is inferred at emit time.
		uint32_t order = 0;
		if (!intr->opt_native && e->call.arg_count == base_args + 1) {
			int32_t o = svsl_atomic_order(e->call.args[base_args]->ident);
			if (o > 0) order = (uint32_t)o;
		}
		b->fn->insts.items[prior].args[3] = atomic_op | (order << 8);
		// Interlocked out-param form: write the prior value back
		if (intr->opt_native && e->call.arg_count == base_args + 1)
			store_target(b, e->call.args[base_args], prior);
		return prior;
	}

	// generic: operands into aux; the natural result type follows the table's
	// result rule over the first operand (e->sema_type may carry an outer
	// conversion, applied by finish())
	uint32_t parts[8];
	uint32_t count = 0;
	for (int32_t i = 0; i < e->call.arg_count && count < 8; i++)
		parts[count++] = lower_expr(b, e->call.args[i]);

	svsl_types_t  *types  = &b->prog->types;
	svsl_type_id_t result = e->sema_type;
	if (count > 0) {
		svsl_type_id_t     gen = value_type(b, parts[0]);
		const svsl_type_t *gt  = svsl_type_get(types, gen);
		int32_t gn = gt->kind == svsl_type_vector ? gt->count : 1;
		switch ((svsl_ires_)intr->result) {
		case svsl_ires_gen:         result = gen; break;
		case svsl_ires_scalar:      result = svsl_type_scalar_id(types, gt->scalar); break;
		case svsl_ires_bool_shape:  result = gn == 1 ? svsl_type_scalar_id(types, svsl_scalar_bool)
		                                             : svsl_type_vector_id(types, svsl_scalar_bool, gn); break;
		case svsl_ires_uint_shape:  result = gn == 1 ? svsl_type_scalar_id(types, svsl_scalar_uint32)
		                                             : svsl_type_vector_id(types, svsl_scalar_uint32, gn); break;
		case svsl_ires_int_shape:   result = gn == 1 ? svsl_type_scalar_id(types, svsl_scalar_int32)
		                                             : svsl_type_vector_id(types, svsl_scalar_int32, gn); break;
		case svsl_ires_float_shape: result = gn == 1 ? svsl_type_scalar_id(types, svsl_scalar_float32)
		                                             : svsl_type_vector_id(types, svsl_scalar_float32, gn); break;
		default: break; // fixed result types are always exact already
		}
	}
	uint32_t aux = aux_push(b, parts, count);
	return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_intrinsic, .type = result,
	                                 .args = { (uint32_t)index, 0, 0, SVSL_IR_NONE },
	                                 .aux = aux, .aux_count = count, .loc = e->loc });
}

static uint32_t lower_method(build_t *b, const svsl_ast_expr_t *e) {
	const svsl_ast_expr_t *callee = e->call.callee;
	int32_t method  = callee->sema_ref.a;
	int32_t channel = callee->sema_ref.b;
	int32_t res     = resolve_resource(b, callee->member.object);
	const svsl_type_t *obj = svsl_type_get(&b->prog->types, callee->member.object->sema_type);

	if (obj->kind == svsl_type_image) {
		if (method == svsl_method_load) {
			uint32_t coord = lower_expr(b, e->call.args[0]);
			return emit_op(b, svsl_ir_image_load, e->sema_type, (uint32_t)res, coord, 0, e->loc);
		}
		if (method == svsl_method_store) {
			uint32_t coord = lower_expr(b, e->call.args[0]);
			uint32_t value = lower_expr(b, e->call.args[1]);
			return emit_op(b, svsl_ir_image_store, SVSL_TYPE_NONE, (uint32_t)res, coord, value, e->loc);
		}
		if (method == svsl_method_atomic) {
			uint32_t coord = lower_expr(b, e->call.args[0]);
			uint32_t value = lower_expr(b, e->call.args[1]);
			uint32_t prior = emit_op(b, svsl_ir_image_atomic,
			                         value_type(b, value), (uint32_t)res, coord, value, e->loc);
			b->fn->insts.items[prior].args[3] = (uint32_t)channel;
			if (e->call.arg_count == 3) store_target(b, e->call.args[2], prior);
			return prior;
		}
	}

	// GetDimensions out-parameter forms: query, then write each component out
	if (method == svsl_method_get_dimensions && e->call.arg_count > 0) {
		svsl_types_t  *types = &b->prog->types;
		svsl_type_id_t u32   = svsl_type_scalar_id(types, svsl_scalar_uint32);
		if (obj->kind == svsl_type_buffer) { // (out count[, out stride])
			uint32_t count = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = u32,
			                                           .args = { (uint32_t)res, SVSL_IR_NONE,
			                                                     (uint32_t)method, SVSL_IR_NONE },
			                                           .loc = e->loc });
			store_target(b, e->call.args[0], convert_value(b, count, e->call.args[0]->sema_type, e->loc));
			if (e->call.arg_count == 2) { // stride is a compile-time layout fact
				uint32_t stride = emit_const_bits(b, u32,
					(uint64_t)b->prog->resources.items[res].element_size, e->loc);
				store_target(b, e->call.args[1], convert_value(b, stride, e->call.args[1]->sema_type, e->loc));
			}
			return count;
		}
		// textures/images: (out dims...), (mip, out dims..., out levels), or
		// for multisampled textures (out dims..., out samples)
		int32_t dims = (obj->dim == svsl_texdim_1d ? 1 : obj->dim == svsl_texdim_3d ? 3 : 2) +
		               (obj->arrayed ? 1 : 0);
		bool     mip_form   = e->call.arg_count == dims + 2;
		bool     count_form = mip_form || (obj->multisampled && e->call.arg_count == dims + 1);
		uint32_t mip        = mip_form ? lower_expr(b, e->call.args[0]) : SVSL_IR_NONE;
		svsl_type_id_t qtype = dims == 1 ? u32 : svsl_type_vector_id(types, svsl_scalar_uint32, dims);
		uint32_t query = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = qtype,
		                                           .args = { (uint32_t)res, SVSL_IR_NONE,
		                                                     (uint32_t)method, SVSL_IR_NONE },
		                                           .aux = mip_form ? aux_push(b, &mip, 1) : 0,
		                                           .aux_count = mip_form ? 1u : 0u, .loc = e->loc });
		int32_t first = mip_form ? 1 : 0;
		for (int32_t i = 0; i < dims; i++) {
			uint32_t comp = dims == 1 ? query
			              : emit_op(b, svsl_ir_extract, u32, query, (uint32_t)i, 0, e->loc);
			comp = convert_value(b, comp, e->call.args[first + i]->sema_type, e->loc);
			store_target(b, e->call.args[first + i], comp);
		}
		if (count_form) { // trailing level/sample count; channel 1 selects the query
			uint32_t levels = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = u32,
			                                            .args = { (uint32_t)res, SVSL_IR_NONE,
			                                                      (uint32_t)method | (1u << 8),
			                                                      SVSL_IR_NONE },
			                                            .loc = e->loc });
			const svsl_ast_expr_t *last = e->call.args[e->call.arg_count - 1];
			store_target(b, last, convert_value(b, levels, last->sema_type, e->loc));
		}
		return query;
	}

	// QCOM image processing with a second texture operand: the weight/reference
	// texture is a resource riding args[3]; one sampler serves both sampled images
	if (method == svsl_method_sample_weighted || method == svsl_method_block_match) {
		uint32_t sampler = (uint32_t)resolve_resource(b, e->call.args[0]);
		uint32_t res2    = (uint32_t)resolve_resource(b, e->call.args[2]);
		uint32_t parts[3];
		uint32_t count = 0;
		parts[count++] = lower_expr(b, e->call.args[1]);
		if (method == svsl_method_block_match) {
			parts[count++] = lower_expr(b, e->call.args[3]);
			parts[count++] = lower_expr(b, e->call.args[4]);
		}
		uint32_t aux = aux_push(b, parts, count);
		return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = e->sema_type,
		                                 .args = { (uint32_t)res, sampler,
		                                           (uint32_t)method | ((uint32_t)(channel & 0xFF) << 8),
		                                           res2 },
		                                 .aux = aux, .aux_count = count, .loc = e->loc });
	}

	// sampler argument (Sample family) becomes the sampler resource index
	uint32_t sampler = SVSL_IR_NONE;
	int32_t  first   = 0;
	if (((method >= svsl_method_sample && method <= svsl_method_gather) ||
	     method == svsl_method_query_lod || method == svsl_method_box_filter) &&
	    e->call.arg_count > 0 &&
	    svsl_type_get(&b->prog->types, e->call.args[0]->sema_type)->kind == svsl_type_sampler) {
		sampler = (uint32_t)resolve_resource(b, e->call.args[0]);
		first   = 1;
	}
	uint32_t parts[8];
	uint32_t count = 0;
	for (int32_t i = first; i < e->call.arg_count && count < 8; i++)
		parts[count++] = lower_expr(b, e->call.args[i]);
	uint32_t aux = aux_push(b, parts, count);
	return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = e->sema_type,
	                                 .args = { (uint32_t)res, sampler,
	                                           (uint32_t)method | ((uint32_t)(channel & 0xFF) << 8),
	                                           SVSL_IR_NONE },
	                                 .aux = aux, .aux_count = count, .loc = e->loc });
}

// counts return statements; used to decide whether an inlined body needs the
// single-trip loop wrapper (return = break)
static void count_returns(const svsl_ast_stmt_t *s, int32_t *count, bool *last_is_return) {
	if (!s) return;
	switch (s->kind) {
	case svsl_stmt_return: (*count)++; break;
	case svsl_stmt_block:
		for (int32_t i = 0; i < s->block.count; i++)
			count_returns(s->block.stmts[i], count, last_is_return);
		break;
	case svsl_stmt_if:
		count_returns(s->if_stmt.then_stmt, count, last_is_return);
		count_returns(s->if_stmt.else_stmt, count, last_is_return);
		break;
	case svsl_stmt_for:   count_returns(s->for_stmt.body, count, last_is_return); break;
	case svsl_stmt_while:
	case svsl_stmt_do:    count_returns(s->while_stmt.body, count, last_is_return); break;
	case svsl_stmt_switch:
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++)
			for (int32_t k = 0; k < s->switch_stmt.cases[i].stmt_count; k++)
				count_returns(s->switch_stmt.cases[i].stmts[k], count, last_is_return);
		break;
	default: break;
	}
}

static bool body_is_tail_return_only(const svsl_ast_stmt_t *body) {
	int32_t count = 0;
	bool    last  = false;
	count_returns(body, &count, &last);
	if (count == 0) return true; // void function without return
	if (count > 1) return false;
	// the single return must be the last top-level statement
	if (body->kind != svsl_stmt_block || body->block.count == 0) return false;
	return body->block.stmts[body->block.count - 1]->kind == svsl_stmt_return;
}

// True if any return sits inside a loop or switch. Such a return can't break
// straight to the function wrapper — structured CF forbids a multi-level break —
// so the wrapper needs a `returned` flag whose break cascades outward one level
// per enclosing construct (matches glslang's early-return lowering).
static bool return_inside_loop(const svsl_ast_stmt_t *s, bool in_loop) {
	if (!s) return false;
	switch (s->kind) {
	case svsl_stmt_return: return in_loop;
	case svsl_stmt_block:
		for (int32_t i = 0; i < s->block.count; i++)
			if (return_inside_loop(s->block.stmts[i], in_loop)) return true;
		return false;
	case svsl_stmt_if:
		return return_inside_loop(s->if_stmt.then_stmt, in_loop) ||
		       return_inside_loop(s->if_stmt.else_stmt, in_loop);
	case svsl_stmt_for:   return return_inside_loop(s->for_stmt.body,   true);
	case svsl_stmt_while:
	case svsl_stmt_do:    return return_inside_loop(s->while_stmt.body, true);
	case svsl_stmt_switch:
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++)
			for (int32_t k = 0; k < s->switch_stmt.cases[i].stmt_count; k++)
				if (return_inside_loop(s->switch_stmt.cases[i].stmts[k], true)) return true;
		return false;
	default: return false;
	}
}

static bool subtree_has_return(const svsl_ast_stmt_t *s) {
	int32_t count = 0;
	bool    last  = false;
	count_returns(s, &count, &last);
	return count > 0;
}

// `if (returned) break;` after a loop/switch body: propagates an early return one
// structured level outward (to the enclosing loop/switch, else the function wrapper).
static void emit_return_guard(build_t *b, svsl_loc_t loc) {
	uint32_t flag = emit_op(b, svsl_ir_load, svsl_type_scalar_id(&b->prog->types, svsl_scalar_bool),
	                        b->ctx->returned_var, 0, 0, loc);
	emit_op(b, svsl_ir_if,     SVSL_TYPE_NONE, flag, 0, 0, loc);
	emit_op(b, svsl_ir_break,  SVSL_TYPE_NONE, 0, 0, 0, loc);
	emit_op(b, svsl_ir_end_if, SVSL_TYPE_NONE, 0, 0, 0, loc);
}

static uint32_t lower_user_call(build_t *b, const svsl_ast_expr_t *e) {
	if (b->depth >= IR_MAX_INLINE_DEPTH) { berr(b, e->loc, "inlining too deep"); return 0; }
	const svsl_func_info_t *info = &b->prog->functions.items[e->call.callee->sema_ref.a];
	const svsl_ast_func_t  *func = info->func;
	if (!func->body) { berr(b, e->loc, "call to undefined function"); return 0; }

	uint32_t *param_vars = svsl_arena_alloc(b->arena,
		(size_t)(func->param_count > 0 ? func->param_count : 1) * sizeof(uint32_t));
	uint32_t *out_ptrs = svsl_arena_alloc(b->arena,
		(size_t)(func->param_count > 0 ? func->param_count : 1) * sizeof(uint32_t));

	for (int32_t i = 0; i < func->param_count; i++) {
		out_ptrs[i] = SVSL_IR_NONE;
		const svsl_type_t *pt = svsl_type_get(&b->prog->types, info->param_types[i]);
		bool opaque = pt->kind == svsl_type_texture || pt->kind == svsl_type_sampler ||
		              pt->kind == svsl_type_image || pt->kind == svsl_type_buffer ||
		              pt->kind == svsl_type_subpass;
		if (opaque) { // resolves to the caller's resource — the prototype-killer fix
			param_vars[i] = RES_MARK | (uint32_t)resolve_resource(b, e->call.args[i]);
			continue;
		}
		uint32_t var = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_var, .type = info->param_types[i],
		                                         .args = { 0, 0, 0, SVSL_IR_NONE },
		                                         .loc = e->loc, .name = func->params[i]->name });
		param_vars[i] = var;
		uint8_t dir = func->params[i]->dir;
		if (dir == svsl_dir_out || dir == svsl_dir_inout) {
			uint32_t ptr = lower_lvalue(b, e->call.args[i]);
			if (ptr == SVSL_IR_NONE) { berr(b, e->call.args[i]->loc, "out argument needs storage"); return 0; }
			out_ptrs[i] = ptr;
			if (dir == svsl_dir_inout) {
				uint32_t init = emit_op(b, svsl_ir_load, info->param_types[i], ptr, 0, 0, e->loc);
				emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, var, init, 0, e->loc);
			}
		} else {
			uint32_t value = lower_expr(b, e->call.args[i]);
			emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, var, value, 0, e->loc);
		}
	}

	const svsl_type_t *rt = svsl_type_get(&b->prog->types, info->return_type);
	uint32_t result_var = SVSL_IR_NONE;
	if (rt->kind != svsl_type_void)
		result_var = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_var, .type = info->return_type,
		                                       .args = { 0, 0, 0, SVSL_IR_NONE },
		                                       .loc = e->loc, .name = func->name });

	lower_ctx_t ctx = {
		.parent       = b->ctx,
		.func         = func,
		.info         = info,
		.param_vars   = param_vars,
		.result_var   = result_var,
		.returned_var = SVSL_IR_NONE,
		.tail_return  = body_is_tail_return_only(func->body) };

	// save the local stack: the callee gets a fresh one
	u32_list_t saved_stack = b->local_stack;
	b->local_stack = (u32_list_t){0};
	b->ctx = &ctx;
	b->depth++;

	if (!ctx.tail_return) {
		if (return_inside_loop(func->body, false)) { // flag + break cascade (see return_inside_loop)
			ctx.returned_var = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_var,
			    .type = svsl_type_scalar_id(&b->prog->types, svsl_scalar_bool),
			    .args = { 0, 0, 0, SVSL_IR_NONE }, .loc = e->loc });
			emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, ctx.returned_var,
			        emit_const_int(b, svsl_scalar_bool, 0, e->loc), 0, e->loc);
		}
		emit_op(b, svsl_ir_loop, SVSL_TYPE_NONE, 0, 0, 0, e->loc); // return = break to here
	}
	lower_stmt(b, func->body);
	if (!ctx.tail_return) {
		emit_op(b, svsl_ir_break, SVSL_TYPE_NONE, 0, 0, 0, e->loc);
		emit_op(b, svsl_ir_end_loop, SVSL_TYPE_NONE, SVSL_IR_NONE, 0, 0, e->loc);
	}

	b->depth--;
	b->ctx         = ctx.parent;
	b->local_stack = saved_stack;

	for (int32_t i = 0; i < func->param_count; i++) // out/inout copy-back
		if (out_ptrs[i] != SVSL_IR_NONE) {
			uint32_t final = emit_op(b, svsl_ir_load, info->param_types[i], param_vars[i], 0, 0, e->loc);
			emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, out_ptrs[i], final, 0, e->loc);
		}

	if (result_var == SVSL_IR_NONE) return SVSL_IR_NONE;
	return emit_op(b, svsl_ir_load, info->return_type, result_var, 0, 0, e->loc);
}

static uint32_t lower_call(build_t *b, const svsl_ast_expr_t *e) {
	const svsl_ast_expr_t *callee = e->call.callee;
	if (callee->kind == svsl_expr_member)             return lower_method(b, e);
	if (callee->sema_ref.kind == svsl_ref_function)   return lower_user_call(b, e);
	if (callee->sema_ref.kind == svsl_ref_intrinsic)  return lower_intrinsic(b, e, callee->sema_ref.a);
	berr(b, e->loc, "unresolved call");
	return 0;
}

// flattens ctor args into scalar components of the target's component type
static int32_t flatten_components(build_t *b, const svsl_ast_expr_t *e, svsl_scalar_ scalar,
                                  uint32_t *out, int32_t max) {
	svsl_types_t *types = &b->prog->types;
	int32_t       total = 0;
	for (int32_t i = 0; i < e->ctor.arg_count; i++) {
		uint32_t v = lower_expr(b, e->ctor.args[i]);
		const svsl_type_t *t = svsl_type_get(types, value_type(b, v));
		if (t->kind == svsl_type_scalar) {
			if (total < max)
				out[total] = convert_value(b, v, svsl_type_scalar_id(types, scalar), e->loc);
			total++;
		} else if (t->kind == svsl_type_vector) {
			for (int32_t k = 0; k < t->count; k++) {
				uint32_t part = emit_op(b, svsl_ir_extract,
				                        svsl_type_scalar_id(types, t->scalar), v, (uint32_t)k, 0, e->loc);
				if (total < max)
					out[total] = convert_value(b, part, svsl_type_scalar_id(types, scalar), e->loc);
				total++;
			}
		}
	}
	return total;
}

static uint32_t construct(build_t *b, svsl_type_id_t type, const uint32_t *parts, uint32_t count,
                          svsl_loc_t loc) {
	uint32_t aux = aux_push(b, parts, count);
	return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_construct, .type = type,
	                                 .args = { 0, 0, 0, SVSL_IR_NONE },
	                                 .aux = aux, .aux_count = count, .loc = loc });
}

static uint32_t lower_ctor(build_t *b, const svsl_ast_expr_t *e) {
	svsl_types_t      *types = &b->prog->types;
	const svsl_type_t *t     = svsl_type_get(types, e->sema_type);

	if (t->kind == svsl_type_scalar) {
		uint32_t v = lower_expr(b, e->ctor.args[0]);
		return convert_value(b, v, e->sema_type, e->loc);
	}
	if (t->kind == svsl_type_vector) {
		uint32_t parts[4];
		int32_t  total = flatten_components(b, e, t->scalar, parts, 4);
		if (total == 1) { // broadcast
			for (int32_t i = 1; i < t->count; i++) parts[i] = parts[0];
			total = t->count;
		}
		return construct(b, e->sema_type, parts, (uint32_t)total, e->loc);
	}
	if (t->kind == svsl_type_matrix) {
		// SPIR-V matrix vectors are HLSL rows, so row-major arguments compose directly
		svsl_type_id_t row_type = svsl_type_vector_id(types, t->scalar, t->cols);
		uint32_t scalars[16];
		int32_t  total = flatten_components(b, e, t->scalar, scalars, 16);
		uint32_t rows[4];
		if (total == 1) { // float4x4(s): s on the diagonal
			uint32_t zero = emit_const_float(b, t->scalar, 0.0, e->loc);
			for (int32_t row = 0; row < t->rows; row++) {
				uint32_t parts[4];
				for (int32_t col = 0; col < t->cols; col++)
					parts[col] = col == row ? scalars[0] : zero;
				rows[row] = construct(b, row_type, parts, (uint32_t)t->cols, e->loc);
			}
		} else {
			for (int32_t row = 0; row < t->rows; row++) {
				uint32_t parts[4];
				for (int32_t col = 0; col < t->cols; col++)
					parts[col] = scalars[row * t->cols + col];
				rows[row] = construct(b, row_type, parts, (uint32_t)t->cols, e->loc);
			}
		}
		return construct(b, e->sema_type, rows, (uint32_t)t->rows, e->loc);
	}
	berr(b, e->loc, "cannot construct this type");
	return 0;
}

// initializer lists build composites recursively toward the declared type
static uint32_t lower_init(build_t *b, const svsl_ast_expr_t *e, svsl_type_id_t type) {
	if (e->kind != svsl_expr_init_list) {
		uint32_t v = lower_expr(b, e);
		return convert_value(b, v, type, e->loc);
	}
	svsl_types_t      *types = &b->prog->types;
	const svsl_type_t *t     = svsl_type_get(types, type);

	if (t->kind == svsl_type_vector && e->init_list.count == t->count) {
		uint32_t parts[4];
		for (int32_t i = 0; i < t->count; i++)
			parts[i] = lower_init(b, e->init_list.items[i], svsl_type_scalar_id(types, t->scalar));
		return construct(b, type, parts, (uint32_t)t->count, e->loc);
	}
	if (t->kind == svsl_type_matrix && e->init_list.count == t->rows * t->cols) {
		svsl_type_id_t st = svsl_type_scalar_id(types, t->scalar);
		uint32_t scalars[16];
		for (int32_t i = 0; i < t->rows * t->cols && i < 16; i++)
			scalars[i] = lower_init(b, e->init_list.items[i], st);
		uint32_t rows[4]; // SPIR-V matrix vectors are HLSL rows
		for (int32_t row = 0; row < t->rows; row++) {
			uint32_t parts[4];
			for (int32_t col = 0; col < t->cols; col++)
				parts[col] = scalars[row * t->cols + col];
			rows[row] = construct(b, svsl_type_vector_id(types, t->scalar, t->cols),
			                      parts, (uint32_t)t->cols, e->loc);
		}
		return construct(b, type, rows, (uint32_t)t->rows, e->loc);
	}
	if (t->kind == svsl_type_array && e->init_list.count == t->array_count) {
		int32_t   count = t->array_count;
		uint32_t *parts = svsl_arena_alloc(b->arena, (size_t)count * sizeof(uint32_t));
		for (int32_t i = 0; i < count; i++)
			parts[i] = lower_init(b, e->init_list.items[i], t->elem);
		return construct(b, type, parts, (uint32_t)count, e->loc);
	}
	if (t->kind == svsl_type_struct) {
		const svsl_struct_info_t *info = &types->structs.items[t->struct_index];
		if (e->init_list.count == info->members.count) {
			int32_t   count = info->members.count;
			uint32_t *parts = svsl_arena_alloc(b->arena, (size_t)count * sizeof(uint32_t));
			for (int32_t i = 0; i < count; i++)
				parts[i] = lower_init(b, e->init_list.items[i], info->members.items[i].type);
			return construct(b, type, parts, (uint32_t)count, e->loc);
		}
	}
	berr(b, e->loc, "initializer shape does not match the declared type");
	return emit_zero(b, type, e->loc);
}

static uint32_t lower_expr(build_t *b, const svsl_ast_expr_t *e) {
	switch (e->kind) {
	case svsl_expr_int_lit: {
		svsl_scalar_ s = e->int_lit.suffix == svsl_suffix_u  ? svsl_scalar_uint32 :
		                 e->int_lit.suffix == svsl_suffix_l  ? svsl_scalar_int64 :
		                 e->int_lit.suffix == svsl_suffix_ul ? svsl_scalar_uint64 : svsl_scalar_int32;
		return finish(b, e, emit_const_int(b, s, (int64_t)e->int_lit.value, e->loc));
	}
	case svsl_expr_float_lit: {
		svsl_scalar_ s = e->float_lit.suffix == svsl_suffix_h  ? svsl_scalar_half :
		                 e->float_lit.suffix == svsl_suffix_lf ? svsl_scalar_float64 : svsl_scalar_float32;
		return finish(b, e, emit_const_float(b, s, e->float_lit.value, e->loc));
	}
	case svsl_expr_bool_lit:
		return finish(b, e, emit_const_int(b, svsl_scalar_bool, e->bool_lit ? 1 : 0, e->loc));
	case svsl_expr_ident: {
		switch (e->sema_ref.kind) {
		case svsl_ref_spec_const:
			return finish(b, e, emit_op(b, svsl_ir_spec_const, e->sema_type,
			                            (uint32_t)e->sema_ref.a, 0, 0, e->loc));
		case svsl_ref_enum_const: {
			const svsl_enum_const_t *ec = &b->prog->enum_consts.items[e->sema_ref.a];
			svsl_scalar_ sc = svsl_type_get(&b->prog->types, ec->type)->scalar;
			return finish(b, e, emit_const_int(b, sc, ec->value, e->loc)); // folds to a literal
		}
		case svsl_ref_builtin_var: {
			// read via a dedicated intrinsic-like op: reuse ir_ptr with builtin kind + load
			uint32_t ptr = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_ptr, .type = e->sema_type,
			                                         .args = { svsl_ref_builtin_var,
			                                                   (uint32_t)e->sema_ref.a, 0, SVSL_IR_NONE },
			                                         .loc = e->loc, .name = e->ident });
			return finish(b, e, emit_op(b, svsl_ir_load, e->sema_type, ptr, 0, 0, e->loc));
		}
		default: {
			uint32_t ptr = lower_lvalue(b, e);
			if (ptr == SVSL_IR_NONE) { berr(b, e->loc, "cannot read this identifier"); return 0; }
			// pointer type annotations carry the pointee type
			svsl_type_id_t pointee = b->fn->insts.items[ptr].type;
			return finish(b, e, emit_op(b, svsl_ir_load, pointee, ptr, 0, 0, e->loc));
		}
		}
	}
	case svsl_expr_member: {
		if (e->sema_ref.kind == svsl_ref_bitfield)
			return lower_bitfield_read(b, e);
		if (e->sema_ref.kind == svsl_ref_swizzle) {
			// prefer chains for single components of addressable objects
			if (e->sema_ref.b == 1) {
				uint32_t ptr = lower_lvalue(b, e);
				if (ptr != SVSL_IR_NONE)
					return finish(b, e, emit_op(b, svsl_ir_load,
					                            b->fn->insts.items[ptr].type, ptr, 0, 0, e->loc));
			}
			uint32_t vec = lower_expr(b, e->member.object);
			// shuffles/extracts keep the object's component type; finish() converts
			const svsl_type_t *ot   = svsl_type_get(&b->prog->types, value_type(b, vec));
			svsl_scalar_       scal = ot->scalar;
			if (ot->kind == svsl_type_scalar) { // f.xxx broadcast on a scalar
				if (e->sema_ref.b == 1) return finish(b, e, vec);
				uint32_t parts[4] = { vec, vec, vec, vec };
				return finish(b, e, construct(b,
					svsl_type_vector_id(&b->prog->types, scal, e->sema_ref.b),
					parts, (uint32_t)e->sema_ref.b, e->loc));
			}
			if (e->sema_ref.b == 1)
				return finish(b, e, emit_op(b, svsl_ir_extract,
				                            svsl_type_scalar_id(&b->prog->types, scal), vec,
				                            (uint32_t)(e->sema_ref.a & 0xF), 0, e->loc));
			return finish(b, e, emit_op(b, svsl_ir_shuffle,
			                            svsl_type_vector_id(&b->prog->types, scal, e->sema_ref.b), vec,
			                            (uint32_t)e->sema_ref.a, (uint32_t)e->sema_ref.b, e->loc));
		}
		uint32_t ptr = lower_lvalue(b, e);
		if (ptr != SVSL_IR_NONE)
			return finish(b, e, emit_op(b, svsl_ir_load,
			                            b->fn->insts.items[ptr].type, ptr, 0, 0, e->loc));
		// member of a non-addressable value (call result): extract
		uint32_t obj = lower_expr(b, e->member.object);
		if (e->sema_ref.kind == svsl_ref_matrix_elem) {
			uint32_t row = emit_op(b, svsl_ir_extract,
			                       svsl_type_vector_id(&b->prog->types,
			                                           svsl_type_get(&b->prog->types, value_type(b, obj))->scalar,
			                                           svsl_type_get(&b->prog->types, value_type(b, obj))->cols),
			                       obj, (uint32_t)e->sema_ref.a, 0, e->loc);
			return finish(b, e, emit_op(b, svsl_ir_extract, e->sema_type, row,
			                            (uint32_t)e->sema_ref.b, 0, e->loc));
		}
		return finish(b, e, emit_op(b, svsl_ir_extract, e->sema_type, obj,
		                            (uint32_t)e->sema_ref.a, 0, e->loc));
	}
	case svsl_expr_index: {
		const svsl_type_t *obj = svsl_type_get(&b->prog->types, e->index.object->sema_type);
		if (obj->kind == svsl_type_image) {
			int32_t  res   = resolve_resource(b, e->index.object);
			uint32_t coord = lower_expr(b, e->index.index);
			return finish(b, e, emit_op(b, svsl_ir_image_load, e->sema_type,
			                            (uint32_t)res, coord, 0, e->loc));
		}
		if (obj->kind == svsl_type_texture) {
			// tex[coord] == tex.Load(int(dim+1)(coord, 0)): build the mip-0 fetch
			// coordinate and reuse the Load path (svsl_ir_tex).
			int32_t  res   = resolve_resource(b, e->index.object);
			uint32_t coord = lower_expr(b, e->index.index);
			int32_t  n     = obj->dim == svsl_texdim_1d ? 1 : obj->dim == svsl_texdim_3d ? 3 : 2;
			n += obj->arrayed ? 1 : 0;
			uint32_t parts[2] = { coord, emit_const_int(b, svsl_scalar_int32, 0, e->loc) };
			uint32_t full     = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_construct,
			                          .type = svsl_type_vector_id(&b->prog->types, svsl_scalar_int32, n + 1),
			                          .args = { 0, 0, 0, SVSL_IR_NONE },
			                          .aux = aux_push(b, parts, 2), .aux_count = 2, .loc = e->loc });
			return finish(b, e, emit(b, (svsl_ir_inst_t){ .op = svsl_ir_tex, .type = e->sema_type,
			                          .args = { (uint32_t)res, SVSL_IR_NONE, (uint32_t)svsl_method_load, SVSL_IR_NONE },
			                          .aux = aux_push(b, &full, 1), .aux_count = 1, .loc = e->loc }));
		}
		uint32_t ptr = lower_lvalue(b, e);
		if (ptr != SVSL_IR_NONE)
			return finish(b, e, emit_op(b, svsl_ir_load,
			                            b->fn->insts.items[ptr].type, ptr, 0, 0, e->loc));
		// indexing a non-addressable value (call result, constructor):
		// constant indices extract the element directly; dynamic indices spill
		// the value into a temporary so it becomes addressable
		uint32_t base  = lower_expr(b, e->index.object);
		uint32_t index = lower_expr(b, e->index.index);
		svsl_type_id_t elem = pointee_elem(b, value_type(b, base), 0);
		const svsl_ir_inst_t *ii = &b->fn->insts.items[index];
		if (ii->op == svsl_ir_const)
			return finish(b, e, emit_op(b, svsl_ir_extract, elem, base, ii->args[0], 0, e->loc));
		uint32_t tmp = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_var, .type = value_type(b, base),
		                                         .args = { 0, 0, 0, SVSL_IR_NONE },
		                                         .loc = e->loc, .name = svsl_str("indexed") });
		emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, tmp, base, 0, e->loc);
		uint32_t chain = emit_chain(b, tmp, &index, 1, elem, e->loc);
		return finish(b, e, emit_op(b, svsl_ir_load, elem, chain, 0, 0, e->loc));
	}
	case svsl_expr_unary: {
		if (e->unary.op == svsl_tok_plusplus || e->unary.op == svsl_tok_minusminus) {
			target_t t;
			if (!target_resolve(b, e->unary.operand, &t)) {
				berr(b, e->unary.operand->loc, "cannot store through this expression");
				return SVSL_IR_NONE;
			}
			uint32_t old = target_load(b, &t, e->unary.operand->sema_type, e->loc);
			uint32_t one = emit_const_int(b, svsl_scalar_int32, 1, e->loc);
			one = convert_value(b, one, value_type(b, old), e->loc);
			uint32_t neu = emit_op(b, e->unary.op == svsl_tok_plusplus ? svsl_ir_add : svsl_ir_sub,
			                       value_type(b, old), old, one, 0, e->loc);
			target_store(b, &t, neu, e->loc);
			return finish(b, e, neu); // pre-increment yields the new value
		}
		uint32_t v = lower_expr(b, e->unary.operand);
		if (e->unary.op == svsl_tok_plus) return finish(b, e, v);
		svsl_ir_op_ op = e->unary.op == svsl_tok_minus ? svsl_ir_neg :
		                 e->unary.op == svsl_tok_tilde ? svsl_ir_bit_not : svsl_ir_log_not;
		return finish(b, e, emit_op(b, op, value_type(b, v), v, 0, 0, e->loc));
	}
	case svsl_expr_post: {
		target_t t;
		if (!target_resolve(b, e->post.operand, &t)) {
			berr(b, e->post.operand->loc, "cannot store through this expression");
			return SVSL_IR_NONE;
		}
		uint32_t old = target_load(b, &t, e->post.operand->sema_type, e->loc);
		uint32_t one = emit_const_int(b, svsl_scalar_int32, 1, e->loc);
		one = convert_value(b, one, value_type(b, old), e->loc);
		uint32_t neu = emit_op(b, e->post.op == svsl_tok_plusplus ? svsl_ir_add : svsl_ir_sub,
		                       value_type(b, old), old, one, 0, e->loc);
		target_store(b, &t, neu, e->loc);
		return finish(b, e, old); // post-increment yields the old value
	}
	case svsl_expr_ternary: { // HLSL ?: evaluates both sides (select semantics)
		uint32_t cond = lower_expr(b, e->ternary.cond);
		uint32_t a    = lower_expr(b, e->ternary.then_expr);
		uint32_t v    = lower_expr(b, e->ternary.else_expr);
		return finish(b, e, emit_op(b, svsl_ir_select, value_type(b, a), cond, a, v, e->loc));
	}
	case svsl_expr_binary:
		return finish(b, e, lower_binary(b, e));
	case svsl_expr_cast: {
		const svsl_type_t *t = svsl_type_get(&b->prog->types, e->sema_type);
		uint32_t v = lower_expr(b, e->cast.operand);
		const svsl_type_t *f = svsl_type_get(&b->prog->types, value_type(b, v));
		if (t->kind == svsl_type_struct) // (psIn)0
			return emit_zero(b, e->sema_type, e->loc);
		if (t->kind == svsl_type_matrix && f->kind == svsl_type_matrix) {
			// shrink: leading rows, each truncated to the new column count
			uint32_t rows[4];
			svsl_type_id_t row_type = svsl_type_vector_id(&b->prog->types, t->scalar, t->cols);
			for (int32_t row = 0; row < t->rows; row++) {
				uint32_t big = emit_op(b, svsl_ir_extract,
				                       svsl_type_vector_id(&b->prog->types, f->scalar, f->cols),
				                       v, (uint32_t)row, 0, e->loc);
				uint32_t packed = 0;
				for (int32_t c = 0; c < t->cols; c++) packed |= (uint32_t)c << (c * 4);
				rows[row] = emit_op(b, svsl_ir_shuffle, row_type, big, packed, (uint32_t)t->cols, e->loc);
			}
			return construct(b, e->sema_type, rows, (uint32_t)t->rows, e->loc);
		}
		if (t->kind == svsl_type_matrix && f->kind == svsl_type_scalar) { // (float3x3)0
			return emit_zero(b, e->sema_type, e->loc);
		}
		return convert_value(b, v, e->sema_type, e->loc);
	}
	case svsl_expr_ctor:
		return finish(b, e, lower_ctor(b, e));
	case svsl_expr_call:
		return finish(b, e, lower_call(b, e));
	case svsl_expr_init_list:
		return lower_init(b, e, e->sema_type);
	case svsl_expr_spirv_asm: {
		// lower the $value operands first (forward refs), in binary order; their
		// ids become the aux dataflow. The opcode/operand structure stays in the
		// AST node, recovered at emit via asm_nodes.
		u32_list_t vals = {0};
		for (int32_t k = 0; k < e->spirv_asm.inst_count; k++) {
			const svsl_ast_spv_inst_t *inst = &e->spirv_asm.insts[k];
			for (int32_t o = 0; o < inst->operand_count; o++)
				if (inst->operands[o].kind == svsl_spv_operand_value)
					svsl_array_push(b->arena, &vals, lower_expr(b, inst->operands[o].value));
		}
		uint32_t aux  = aux_push(b, vals.items, (uint32_t)vals.count);
		uint32_t node = (uint32_t)b->fn->asm_nodes.count;
		svsl_array_push(b->arena, &b->fn->asm_nodes, e);
		return emit(b, (svsl_ir_inst_t){ .op = svsl_ir_spirv_asm, .type = e->sema_type,
		                                 .args = { node, 0, 0, SVSL_IR_NONE },
		                                 .aux = aux, .aux_count = (uint32_t)vals.count, .loc = e->loc });
	}
	default:
		berr(b, e->loc, "cannot lower this expression");
		return 0;
	}
}

// --- statements ----------------------------------------------------------------------

static void lower_block(build_t *b, svsl_ast_stmt_t *const *stmts, int32_t count) {
	int32_t mark = b->local_stack.count;
	for (int32_t i = 0; i < count; i++)
		lower_stmt(b, stmts[i]);
	b->local_stack.count = mark;
}

// `precise`: forbid fma contraction in a marked variable's computation so the
// result matches the written expression (numerical stability, crack-free
// geometry between adjacent triangles). Walks the assigned value's contributing
// float-arithmetic ops and flags them; emit turns each flag into an OpDecorate
// NoContraction. This covers the idiomatic single-expression form
// `precise float x = a*b + c;`; a contribution routed through a separate,
// non-precise local is not tracked across the memory round-trip (mark it too).
static bool is_precise_arith(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul:
	case svsl_ir_div: case svsl_ir_neg: case svsl_ir_mat_mul:
		return true;
	default:
		return false;
	}
}
static void mark_precise(build_t *b, uint32_t val, uint8_t *seen) {
	if (val >= (uint32_t)b->fn->insts.count || seen[val]) return;
	seen[val] = 1;
	svsl_ir_inst_t *in = &b->fn->insts.items[val];
	if (is_precise_arith((svsl_ir_op_)in->op)) in->flags |= svsl_ir_flag_precise;
	uint32_t mask = svsl_ir_value_arg_mask(in);
	for (int32_t a = 0; a < 4; a++)
		if (mask & (1u << a)) mark_precise(b, in->args[a], seen);
	if (svsl_ir_aux_holds_values(in))
		for (uint32_t k = 0; k < in->aux_count; k++)
			mark_precise(b, b->fn->aux.items[in->aux + k], seen);
}

// Control-flow hints from statement attributes. Encoded to match SPIR-V's
// SelectionControl / LoopControl masks (Flatten/Unroll = 1, DontFlatten/
// DontUnroll = 2 in both), so emit forwards the value straight to OpSelectionMerge
// / OpLoopMerge. These carry author intent the backend cannot infer: whether a
// branch is worth predicating (`[flatten]`) or keeping (`[branch]`), and whether
// a loop should unroll (`[unroll]`) or stay rolled (`[loop]`).
static uint32_t selection_control(const svsl_ast_attrs_t *a) {
	uint32_t m = 0;
	for (int32_t i = 0; i < a->count; i++) {
		if (svsl_str_eq_cstr(a->items[i].name, "flatten")) m |= 1;
		if (svsl_str_eq_cstr(a->items[i].name, "branch"))  m |= 2;
	}
	return m;
}
static uint32_t loop_control(const svsl_ast_attrs_t *a) {
	uint32_t m = 0;
	for (int32_t i = 0; i < a->count; i++) {
		if (svsl_str_eq_cstr(a->items[i].name, "unroll")) m |= 1;
		if (svsl_str_eq_cstr(a->items[i].name, "loop"))   m |= 2;
	}
	return m;
}

static void lower_stmt(build_t *b, const svsl_ast_stmt_t *s) {
	if (!s) return;
	switch (s->kind) {
	case svsl_stmt_block:
		lower_block(b, s->block.stmts, s->block.count);
		break;
	case svsl_stmt_expr:
		if (s->expr) lower_expr(b, s->expr);
		break;
	case svsl_stmt_var_decl:
		for (int32_t i = 0; i < s->var_decl.count; i++) {
			const svsl_ast_var_t *var = s->var_decl.vars[i];
			// the declared type: recover from the init annotation or default 0
			svsl_type_id_t type = var->init ? var->init->sema_type : SVSL_TYPE_NONE;
			if (type == SVSL_TYPE_NONE) {
				// no initializer: sema stored the local's type on the checker stack in
				// declaration order — replay the resolution
				extern svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *, svsl_program_t *,
				                                             svsl_diag_list_t *, const svsl_ast_type_t *);
				type = svsl_sema_resolve_type(b->arena, b->prog, b->diags, var->type);
			}
			uint32_t v = emit(b, (svsl_ir_inst_t){ .op = svsl_ir_var, .type = type,
			                                       .args = { 0, 0, 0, SVSL_IR_NONE },
			                                       .loc = var->loc, .name = var->name });
			svsl_array_push(b->arena, &b->local_stack, v);
			if (var->init) {
				uint32_t init = lower_init(b, var->init, type);
				emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, v, init, 0, var->loc);
				if (var->flags & svsl_var_flag_precise)
					mark_precise(b, init, svsl_arena_alloc(b->arena, (size_t)b->fn->insts.count));
			}
		}
		break;
	case svsl_stmt_if: {
		uint32_t cond = lower_expr(b, s->if_stmt.cond);
		emit_op(b, svsl_ir_if, SVSL_TYPE_NONE, cond, selection_control(&s->attrs), 0, s->loc);
		lower_stmt(b, s->if_stmt.then_stmt);
		if (s->if_stmt.else_stmt) {
			emit_op(b, svsl_ir_else, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
			lower_stmt(b, s->if_stmt.else_stmt);
		}
		emit_op(b, svsl_ir_end_if, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		break;
	}
	case svsl_stmt_for: {
		int32_t mark = b->local_stack.count;
		lower_stmt(b, s->for_stmt.init);
		emit_op(b, svsl_ir_loop, SVSL_TYPE_NONE, loop_control(&s->attrs), 0, 0, s->loc);
		if (s->for_stmt.cond) {
			uint32_t cond = lower_expr(b, s->for_stmt.cond);
			uint32_t not_ = emit_op(b, svsl_ir_log_not, value_type(b, cond), cond, 0, 0, s->loc);
			emit_op(b, svsl_ir_if, SVSL_TYPE_NONE, not_, 0, 0, s->loc);
			emit_op(b, svsl_ir_break, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
			emit_op(b, svsl_ir_end_if, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		}
		lower_stmt(b, s->for_stmt.body);
		emit_op(b, svsl_ir_loop_continue, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		if (s->for_stmt.inc) lower_expr(b, s->for_stmt.inc);
		emit_op(b, svsl_ir_end_loop, SVSL_TYPE_NONE, SVSL_IR_NONE, 0, 0, s->loc);
		if (b->ctx && b->ctx->returned_var != SVSL_IR_NONE && subtree_has_return(s->for_stmt.body))
			emit_return_guard(b, s->loc);
		b->local_stack.count = mark;
		break;
	}
	case svsl_stmt_while: {
		emit_op(b, svsl_ir_loop, SVSL_TYPE_NONE, loop_control(&s->attrs), 0, 0, s->loc);
		uint32_t cond = lower_expr(b, s->while_stmt.cond);
		uint32_t not_ = emit_op(b, svsl_ir_log_not, value_type(b, cond), cond, 0, 0, s->loc);
		emit_op(b, svsl_ir_if, SVSL_TYPE_NONE, not_, 0, 0, s->loc);
		emit_op(b, svsl_ir_break, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		emit_op(b, svsl_ir_end_if, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		lower_stmt(b, s->while_stmt.body);
		emit_op(b, svsl_ir_loop_continue, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		emit_op(b, svsl_ir_end_loop, SVSL_TYPE_NONE, SVSL_IR_NONE, 0, 0, s->loc);
		if (b->ctx && b->ctx->returned_var != SVSL_IR_NONE && subtree_has_return(s->while_stmt.body))
			emit_return_guard(b, s->loc);
		break;
	}
	case svsl_stmt_do: {
		emit_op(b, svsl_ir_loop, SVSL_TYPE_NONE, loop_control(&s->attrs), 0, 0, s->loc);
		lower_stmt(b, s->while_stmt.body);
		emit_op(b, svsl_ir_loop_continue, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		// the continue construct ends with a conditional back-edge (structured CF
		// forbids breaking out of the continue construct)
		uint32_t cond = lower_expr(b, s->while_stmt.cond);
		emit_op(b, svsl_ir_end_loop, SVSL_TYPE_NONE, cond, 0, 0, s->loc);
		if (b->ctx && b->ctx->returned_var != SVSL_IR_NONE && subtree_has_return(s->while_stmt.body))
			emit_return_guard(b, s->loc);
		break;
	}
	case svsl_stmt_switch: {
		uint32_t value = lower_expr(b, s->switch_stmt.value);
		uint32_t *lits = svsl_arena_alloc(b->arena, (size_t)s->switch_stmt.case_count * sizeof(uint32_t));
		uint32_t lit_count = 0;
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++) {
			const svsl_ast_case_t *cs = &s->switch_stmt.cases[i];
			// literal value (default = marker 0), matched positionally with ir_case ops.
			// Fold named constants (enum members, static-const ints), not just literals.
			extern bool svsl_sema_const_eval_int(svsl_program_t *, const svsl_ast_expr_t *, int64_t *);
			uint32_t lit = 0;
			int64_t  v;
			if (cs->value && svsl_sema_const_eval_int(b->prog, cs->value, &v))
				lit = (uint32_t)v;
			lits[lit_count++] = lit;
		}
		uint32_t aux = aux_push(b, lits, lit_count);
		emit(b, (svsl_ir_inst_t){ .op = svsl_ir_switch, .type = SVSL_TYPE_NONE,
		                          .args = { value, 0, 0, SVSL_IR_NONE },
		                          .aux = aux, .aux_count = lit_count, .loc = s->loc });
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++) {
			const svsl_ast_case_t *cs = &s->switch_stmt.cases[i];
			emit_op(b, svsl_ir_case, SVSL_TYPE_NONE, (uint32_t)i, cs->value ? 0u : 1u, 0, cs->loc);
			lower_block(b, cs->stmts, cs->stmt_count);
		}
		emit_op(b, svsl_ir_end_switch, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		if (b->ctx && b->ctx->returned_var != SVSL_IR_NONE && subtree_has_return(s))
			emit_return_guard(b, s->loc);
		break;
	}
	case svsl_stmt_break:
		emit_op(b, svsl_ir_break, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		break;
	case svsl_stmt_continue:
		emit_op(b, svsl_ir_continue, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		break;
	case svsl_stmt_discard:
		emit_op(b, svsl_ir_discard, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		break;
	case svsl_stmt_demote:
		emit_op(b, svsl_ir_demote, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
		break;
	case svsl_stmt_return: {
		uint32_t value = SVSL_IR_NONE;
		if (s->return_value) value = lower_expr(b, s->return_value);
		if (b->ctx) { // inlined: store the result; break out unless it's the tail return
			if (b->ctx->result_var != SVSL_IR_NONE && value != SVSL_IR_NONE)
				emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, b->ctx->result_var, value, 0, s->loc);
			if (!b->ctx->tail_return) {
				if (b->ctx->returned_var != SVSL_IR_NONE)
					emit_op(b, svsl_ir_store, SVSL_TYPE_NONE, b->ctx->returned_var,
					        emit_const_int(b, svsl_scalar_bool, 1, s->loc), 0, s->loc);
				emit_op(b, svsl_ir_break, SVSL_TYPE_NONE, 0, 0, 0, s->loc);
			}
		} else {
			emit_op(b, svsl_ir_return, SVSL_TYPE_NONE, value, 0, 0, s->loc);
		}
		break;
	}
	}
}

// --- entry ---------------------------------------------------------------------------------

bool svsl_ir_build(svsl_arena_t *arena, svsl_program_t *prog, svsl_opt_level_ opt_level,
                   svsl_ir_module_t *out_module, svsl_diag_list_t *ref_diags) {
	int32_t errors_before = ref_diags->error_count;

	out_module->func_count = prog->entries.count;
	out_module->funcs      = svsl_arena_alloc(arena,
		(size_t)(prog->entries.count > 0 ? prog->entries.count : 1) * sizeof(svsl_ir_func_t));

	for (int32_t i = 0; i < prog->entries.count; i++) {
		svsl_ir_func_t *fn = &out_module->funcs[i];
		*fn = (svsl_ir_func_t){ .entry = &prog->entries.items[i] };

		build_t b = { .arena = arena, .prog = prog, .diags = ref_diags, .fn = fn };
		for (int32_t f = 0; f < prog->functions.count; f++)
			if (prog->functions.items[f].func == prog->entries.items[i].func)
				b.entry_info = &prog->functions.items[f];
		lower_stmt(&b, prog->entries.items[i].func->body);

		// ensure a trailing return (void entries often have no explicit one)
		if (fn->insts.count == 0 ||
		    fn->insts.items[fn->insts.count - 1].op != svsl_ir_return)
			emit_op(&b, svsl_ir_return, SVSL_TYPE_NONE, SVSL_IR_NONE, 0, 0,
			        prog->entries.items[i].func->loc);

		svsl_ir_optimize(arena, fn, prog, opt_level);
	}
	return ref_diags->error_count == errors_before;
}
