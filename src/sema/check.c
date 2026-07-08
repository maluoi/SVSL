#include "check.h"

#include "../tables/intrinsics.h"
#include "../tables/spirv_opcodes.h"
#include "../util/array.h"

#include <stdio.h>
#include <string.h>

typedef struct check_local_t {
	svsl_str_t     name;
	svsl_type_id_t type;
	int32_t        depth;
} check_local_t;

typedef struct check_t {
	svsl_arena_t     *arena;
	svsl_program_t   *prog;
	svsl_diag_list_t *diags;

	const svsl_ast_func_t *func;       // function being checked
	int32_t                func_index;
	svsl_type_id_t         return_type;

	svsl_array_t(check_local_t) locals;
	int32_t                     depth;

	svsl_array_t(int32_t) call_edges; // pairs (caller, callee)
} check_t;

static void cerr(check_t *c, svsl_loc_t loc, const char *fmt, svsl_str_t arg) {
	svsl_diag_add(c->arena, c->diags, svsl_severity_error, loc, fmt, arg.len, arg.ptr);
}

// type name for diagnostics; arena-owned, safe to format directly
static const char *tname(check_t *c, svsl_type_id_t id) {
	return svsl_type_name(&c->prog->types, id);
}

static svsl_type_id_t check_expr(check_t *c, svsl_ast_expr_t *e);

// --- shape helpers ----------------------------------------------------------------

static bool scalar_is_float(svsl_scalar_ s) {
	return s == svsl_scalar_half || s == svsl_scalar_float16 ||
	       s == svsl_scalar_float32 || s == svsl_scalar_float64;
}
static bool scalar_is_int(svsl_scalar_ s) {
	return s >= svsl_scalar_int8 && s <= svsl_scalar_uint64;
}
static int32_t scalar_rank(svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_bool:    return 0;
	case svsl_scalar_int8:    return 1;
	case svsl_scalar_uint8:   return 2;
	case svsl_scalar_int16:   return 3;
	case svsl_scalar_uint16:  return 4;
	case svsl_scalar_int32:   return 5;
	case svsl_scalar_uint32:  return 6;
	case svsl_scalar_int64:   return 7;
	case svsl_scalar_uint64:  return 8;
	case svsl_scalar_float16: return 9;
	case svsl_scalar_half:    return 10;
	case svsl_scalar_float32: return 11;
	case svsl_scalar_float64: return 12;
	default:                  return 0;
	}
}

// scalar count for scalar/vector types; 0 for anything else
static int32_t shape_count(const svsl_types_t *types, svsl_type_id_t id) {
	const svsl_type_t *t = svsl_type_get(types, id);
	if (t->kind == svsl_type_scalar) return 1;
	if (t->kind == svsl_type_vector) return t->count;
	return 0;
}
static svsl_scalar_ shape_scalar(const svsl_types_t *types, svsl_type_id_t id) {
	return svsl_type_get(types, id)->scalar;
}
static svsl_type_id_t make_shape(svsl_types_t *types, svsl_scalar_ scalar, int32_t count) {
	return count <= 1 ? svsl_type_scalar_id(types, scalar)
	                  : svsl_type_vector_id(types, scalar, count);
}

// --- implicit conversions ----------------------------------------------------------------

// -1 = impossible; 0 = exact; larger = worse (used to rank user-function overloads)
static int32_t conv_cost(svsl_types_t *types, svsl_type_id_t from, svsl_type_id_t to) {
	if (from == to) return 0;
	if (from == SVSL_TYPE_NONE || to == SVSL_TYPE_NONE) return -1;
	const svsl_type_t *f = svsl_type_get(types, from);
	const svsl_type_t *t = svsl_type_get(types, to);

	bool f_shaped = f->kind == svsl_type_scalar || f->kind == svsl_type_vector;
	bool t_shaped = t->kind == svsl_type_scalar || t->kind == svsl_type_vector;
	if (!f_shaped || !t_shaped) {
		// arrays/structs/resources convert only exactly (handled above)
		return -1;
	}

	int32_t fc = f->kind == svsl_type_vector ? f->count : 1;
	int32_t tc = t->kind == svsl_type_vector ? t->count : 1;
	int32_t component = f->scalar == t->scalar ? 0
	                  : 1 + (scalar_rank(t->scalar) < scalar_rank(f->scalar) ? 4 : 0);

	if (fc == tc) return component;           // same shape
	if (fc == 1)  return component + 8;       // scalar → vector splat
	if (tc < fc)  return component + 16;      // truncation (warned at apply time)
	return -1;                                 // vector widening never implicit
}

// converts `e` (already checked) to `to`, warning on truncation; false = error emitted
static bool convert_to(check_t *c, svsl_ast_expr_t *e, svsl_type_id_t to) {
	svsl_type_id_t from = e->sema_type;
	int32_t        cost = conv_cost(&c->prog->types, from, to);
	if (cost < 0) {
		svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
		              "cannot convert '%s' to '%s'",
		              svsl_type_name(&c->prog->types, from), svsl_type_name(&c->prog->types, to));
		return false;
	}
	if (cost >= 16)
		svsl_diag_add(c->arena, c->diags, svsl_severity_warning, e->loc,
		              "implicit truncation of '%s' to '%s'",
		              svsl_type_name(&c->prog->types, from), svsl_type_name(&c->prog->types, to));
	e->sema_type = to; // IR reads the annotation and inserts the conversion
	return true;
}

// usual arithmetic conversions: common scalar = higher rank, common count from shapes
static svsl_type_id_t common_shape(check_t *c, svsl_ast_expr_t *a, svsl_ast_expr_t *b) {
	svsl_types_t *types = &c->prog->types;
	int32_t ac = shape_count(types, a->sema_type);
	int32_t bc = shape_count(types, b->sema_type);
	if (ac == 0 || bc == 0) return SVSL_TYPE_NONE;

	svsl_scalar_ as     = shape_scalar(types, a->sema_type);
	svsl_scalar_ bs     = shape_scalar(types, b->sema_type);
	svsl_scalar_ scalar = scalar_rank(as) >= scalar_rank(bs) ? as : bs;

	int32_t count;
	if      (ac == bc)          count = ac;
	else if (ac == 1)           count = bc;
	else if (bc == 1)           count = ac;
	else { // HLSL truncates the wider operand (with a warning at convert time)
		count = ac < bc ? ac : bc;
	}
	return make_shape(types, scalar, count);
}

// --- lvalues ---------------------------------------------------------------------------------

static bool expr_is_lvalue(check_t *c, const svsl_ast_expr_t *e) {
	switch (e->kind) {
	case svsl_expr_ident:
		switch (e->sema_ref.kind) {
		case svsl_ref_local:
		case svsl_ref_workgroup:
			return true;
		case svsl_ref_param: {
			return true; // params are mutable locals in HLSL (in = copy)
		}
		case svsl_ref_buffer_member: {
			const svsl_buffer_t *buf = &c->prog->buffers.items[e->sema_ref.a];
			return buf->kind != svsl_block_uniform; // cbuffers are read-only
		}
		case svsl_ref_resource: {
			const svsl_resource_t *res = &c->prog->resources.items[e->sema_ref.a];
			return res->kind == svsl_res_rw_structured;
		}
		default:
			return false;
		}
	case svsl_expr_member:
		return expr_is_lvalue(c, e->member.object);
	case svsl_expr_index: {
		// indexing an rw image yields a writable texel
		const svsl_type_t *obj = svsl_type_get(&c->prog->types, e->index.object->sema_type);
		if (obj->kind == svsl_type_image) return true;
		return expr_is_lvalue(c, e->index.object);
	}
	default:
		return false;
	}
}

// duplicate components make a swizzle unwritable (c.xx = ...)
static bool swizzle_write_has_dup(const svsl_ast_expr_t *e) {
	if (e->kind != svsl_expr_member || e->sema_ref.kind != svsl_ref_swizzle) return false;
	int32_t seen = 0;
	for (int32_t i = 0; i < e->sema_ref.b; i++) {
		int32_t index = (e->sema_ref.a >> (i * 4)) & 0xF;
		if (seen & (1 << index)) return true;
		seen |= 1 << index;
	}
	return false;
}

// --- identifier resolution ----------------------------------------------------------------------

static svsl_type_id_t resolve_ident(check_t *c, svsl_ast_expr_t *e) {
	svsl_str_t name = e->ident;

	for (int32_t i = c->locals.count - 1; i >= 0; i--) {
		if (svsl_str_eq(c->locals.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_local, .a = i };
			return c->locals.items[i].type;
		}
	}
	for (int32_t i = 0; i < c->func->param_count; i++) {
		if (svsl_str_eq(c->func->params[i]->name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_param, .a = i };
			return c->prog->functions.items[c->func_index].param_types[i];
		}
	}
	for (int32_t b = 0; b < c->prog->buffers.count; b++) {
		const svsl_buffer_t *buf = &c->prog->buffers.items[b];
		for (int32_t m = 0; m < buf->members.count; m++) {
			if (svsl_str_eq(buf->members.items[m].name, name)) {
				e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_buffer_member, .a = b, .b = m };
				return buf->members.items[m].type;
			}
		}
	}
	for (int32_t i = 0; i < c->prog->resources.count; i++) {
		if (svsl_str_eq(c->prog->resources.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_resource, .a = i };
			return c->prog->resources.items[i].type;
		}
	}
	for (int32_t i = 0; i < c->prog->const_globals.count; i++) {
		if (svsl_str_eq(c->prog->const_globals.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_const_global, .a = i };
			return c->prog->const_globals.items[i].type;
		}
	}
	for (int32_t i = 0; i < c->prog->enum_consts.count; i++) {
		if (svsl_str_eq(c->prog->enum_consts.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_enum_const, .a = i };
			return c->prog->enum_consts.items[i].type;
		}
	}
	for (int32_t i = 0; i < c->prog->workgroup_vars.count; i++) {
		if (svsl_str_eq(c->prog->workgroup_vars.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_workgroup, .a = i };
			return c->prog->workgroup_vars.items[i].type;
		}
	}
	for (int32_t i = 0; i < c->prog->spec_consts.count; i++) {
		if (svsl_str_eq(c->prog->spec_consts.items[i].name, name)) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_spec_const, .a = i };
			return c->prog->spec_consts.items[i].type;
		}
	}
	int32_t builtin = svsl_builtin_var_find(name);
	if (builtin >= 0) {
		e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_builtin_var, .a = builtin };
		return svsl_type_scalar_id(&c->prog->types, svsl_scalar_uint32);
	}

	cerr(c, e->loc, "unknown identifier '%.*s'", name);
	return SVSL_TYPE_NONE;
}

// --- swizzles / member access ----------------------------------------------------------------------

static int32_t swizzle_index(char ch) {
	switch (ch) {
	case 'x': case 'r': return 0;
	case 'y': case 'g': return 1;
	case 'z': case 'b': return 2;
	case 'w': case 'a': return 3;
	default:            return -1;
	}
}
static bool swizzle_is_rgba(char ch) { return ch == 'r' || ch == 'g' || ch == 'b' || ch == 'a'; }

static svsl_type_id_t check_swizzle(check_t *c, svsl_ast_expr_t *e, svsl_type_id_t obj_type) {
	svsl_types_t      *types = &c->prog->types;
	const svsl_type_t *t     = svsl_type_get(types, obj_type);
	svsl_str_t         sw    = e->member.name;
	int32_t            count = t->kind == svsl_type_vector ? t->count : 1;

	if (sw.len < 1 || sw.len > 4) {
		cerr(c, e->loc, "bad swizzle '.%.*s'", sw);
		return SVSL_TYPE_NONE;
	}
	bool    rgba   = swizzle_is_rgba(sw.ptr[0]);
	int32_t packed = 0;
	for (int32_t i = 0; i < sw.len; i++) {
		int32_t index = swizzle_index(sw.ptr[i]);
		if (index < 0 || swizzle_is_rgba(sw.ptr[i]) != rgba) {
			cerr(c, e->loc, "bad swizzle '.%.*s' (mixed or unknown components)", sw);
			return SVSL_TYPE_NONE;
		}
		if (index >= count) {
			cerr(c, e->loc, "swizzle '.%.*s' out of range", sw);
			return SVSL_TYPE_NONE;
		}
		packed |= index << (i * 4);
	}
	e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_swizzle, .a = packed, .b = sw.len };
	return make_shape(types, t->scalar, sw.len);
}

// _m00.._m33 (0-based) and _11.._44 (1-based)
static bool matrix_elem_parse(svsl_str_t name, int32_t *out_row, int32_t *out_col) {
	if (name.len == 4 && name.ptr[0] == '_' && name.ptr[1] == 'm' &&
	    name.ptr[2] >= '0' && name.ptr[2] <= '3' && name.ptr[3] >= '0' && name.ptr[3] <= '3') {
		*out_row = name.ptr[2] - '0';
		*out_col = name.ptr[3] - '0';
		return true;
	}
	if (name.len == 3 && name.ptr[0] == '_' &&
	    name.ptr[1] >= '1' && name.ptr[1] <= '4' && name.ptr[2] >= '1' && name.ptr[2] <= '4') {
		*out_row = name.ptr[1] - '1';
		*out_col = name.ptr[2] - '1';
		return true;
	}
	return false;
}

static svsl_type_id_t check_member(check_t *c, svsl_ast_expr_t *e) {
	svsl_type_id_t obj_type = check_expr(c, e->member.object);
	if (obj_type == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
	svsl_types_t      *types = &c->prog->types;
	const svsl_type_t *t     = svsl_type_get(types, obj_type);

	if (t->kind == svsl_type_struct) {
		const svsl_struct_info_t *info = &types->structs.items[t->struct_index];
		if (info->packed) {
			for (int32_t f = 0; f < info->fields.count; f++)
				if (svsl_str_eq(info->fields.items[f].name, e->member.name)) {
					e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_bitfield,
					                                 .a = t->struct_index, .b = f };
					return info->fields.items[f].type;
				}
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%s' has no field '%.*s'", tname(c, obj_type),
			              e->member.name.len, e->member.name.ptr);
			return SVSL_TYPE_NONE;
		}
		for (int32_t m = 0; m < info->members.count; m++) {
			if (svsl_str_eq(info->members.items[m].name, e->member.name)) {
				e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_none, .a = m }; // struct member index
				return info->members.items[m].type;
			}
		}
		svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
		              "'%s' has no member '%.*s'", tname(c, obj_type),
		              e->member.name.len, e->member.name.ptr);
		return SVSL_TYPE_NONE;
	}
	if (t->kind == svsl_type_vector || t->kind == svsl_type_scalar)
		return check_swizzle(c, e, obj_type);
	if (t->kind == svsl_type_matrix) {
		int32_t row, col;
		if (matrix_elem_parse(e->member.name, &row, &col) && row < t->rows && col < t->cols) {
			e->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_matrix_elem, .a = row, .b = col };
			return svsl_type_scalar_id(types, t->scalar);
		}
		cerr(c, e->loc, "bad matrix element '.%.*s'", e->member.name);
		return SVSL_TYPE_NONE;
	}
	svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
	              "cannot use '.%.*s' on a value of type '%s'",
	              e->member.name.len, e->member.name.ptr, tname(c, obj_type));
	return SVSL_TYPE_NONE;
}

// --- calls: methods -----------------------------------------------------------------------------------

static int32_t texdim_coord_count(const svsl_type_t *t) {
	int32_t n = t->dim == svsl_texdim_1d ? 1 : t->dim == svsl_texdim_2d ? 2 : 3;
	return n + (t->arrayed ? 1 : 0);
}

static svsl_type_id_t check_method_call(check_t *c, svsl_ast_expr_t *e) {
	svsl_ast_expr_t *callee   = e->call.callee;
	svsl_type_id_t   obj_type = check_expr(c, callee->member.object);
	if (obj_type == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;

	svsl_types_t      *types = &c->prog->types;
	const svsl_type_t *obj   = svsl_type_get(types, obj_type);
	int32_t            aux   = 0;
	int32_t            method = svsl_method_find(callee->member.name, &aux);
	if (method < 0 ||
	    (obj->kind != svsl_type_texture && obj->kind != svsl_type_image &&
	     obj->kind != svsl_type_buffer && obj->kind != svsl_type_subpass &&
	     obj->kind != svsl_type_tileimage)) {
		cerr(c, callee->loc, "unknown method '%.*s'", callee->member.name);
		return SVSL_TYPE_NONE;
	}
	callee->sema_ref  = (svsl_sema_ref_t){ .kind = svsl_ref_method, .a = method, .b = aux };
	callee->sema_type = obj_type;

	svsl_ast_expr_t **args      = e->call.args;
	int32_t           arg_count = e->call.arg_count;
	for (int32_t i = 0; i < arg_count; i++)
		if (check_expr(c, args[i]) == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;

	svsl_type_id_t elem       = obj->elem;
	svsl_scalar_   elem_scal  = svsl_type_get(types, elem)->scalar;
	int32_t        coords     = obj->kind == svsl_type_texture || obj->kind == svsl_type_image
	                          ? texdim_coord_count(obj) : 1;
	svsl_type_id_t coord_f    = make_shape(types, svsl_scalar_float32, coords);
	svsl_type_id_t coord_i    = make_shape(types, svsl_scalar_int32, coords);
	svsl_type_id_t f32        = svsl_type_scalar_id(types, svsl_scalar_float32);

	switch ((svsl_method_)method) {
	case svsl_method_sample:
	case svsl_method_sample_level:
	case svsl_method_sample_bias:
	case svsl_method_sample_grad:
	case svsl_method_sample_cmp:
	case svsl_method_sample_cmp_level_zero: {
		if (obj->kind != svsl_type_texture) { cerr(c, e->loc, "'%.*s' needs a sampled texture", callee->member.name); return SVSL_TYPE_NONE; }
		if (obj->multisampled) { cerr(c, e->loc, "multisampled textures cannot be sampled; use Load(coord, sample)%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
		int32_t expect = method == svsl_method_sample ? 2 :
		                 method == svsl_method_sample_grad ? 4 : 3;
		if (arg_count != expect) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%.*s' takes %d arguments (sampler, coords, ...), got %d",
			              callee->member.name.len, callee->member.name.ptr, expect, arg_count);
			return SVSL_TYPE_NONE;
		}
		if (svsl_type_get(types, args[0]->sema_type)->kind != svsl_type_sampler) {
			cerr(c, args[0]->loc, "first argument of '%.*s' must be a sampler", callee->member.name);
			return SVSL_TYPE_NONE;
		}
		if (!convert_to(c, args[1], coord_f)) return SVSL_TYPE_NONE;
		for (int32_t i = 2; i < arg_count; i++) {
			svsl_type_id_t want = method == svsl_method_sample_grad ? coord_f : f32;
			if (!convert_to(c, args[i], want)) return SVSL_TYPE_NONE;
		}
		bool cmp = method == svsl_method_sample_cmp || method == svsl_method_sample_cmp_level_zero;
		return cmp ? f32 : elem;
	}
	case svsl_method_gather: {
		if (obj->kind != svsl_type_texture) { cerr(c, e->loc, "'%.*s' needs a sampled texture", callee->member.name); return SVSL_TYPE_NONE; }
		if (arg_count < 2) { cerr(c, e->loc, "'%.*s' takes a sampler and coordinates", callee->member.name); return SVSL_TYPE_NONE; }
		if (!convert_to(c, args[1], coord_f)) return SVSL_TYPE_NONE;
		return svsl_type_vector_id(types, elem_scal, 4);
	}
	case svsl_method_load: {
		if (obj->kind == svsl_type_subpass) {
			if (obj->multisampled) { // SubpassInputMS.SubpassLoad(sample)
				if (arg_count != 1) { cerr(c, e->loc, "SubpassInputMS.Load takes a sample index%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
				if (!convert_to(c, args[0], svsl_type_scalar_id(types, svsl_scalar_int32))) return SVSL_TYPE_NONE;
				return elem;
			}
			if (arg_count != 0) { cerr(c, e->loc, "SubpassInput.Load takes no arguments%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
			return elem;
		}
		if (obj->kind == svsl_type_tileimage) {
			if (arg_count != 0) { cerr(c, e->loc, "TileImage.Load takes no arguments%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
			return elem;
		}
		if (obj->kind == svsl_type_texture && obj->multisampled) { // Load(coord, sample)
			if (arg_count != 2) { cerr(c, e->loc, "Texture2DMS.Load takes (coord, sample)%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
			if (!convert_to(c, args[0], make_shape(types, svsl_scalar_int32, 2)))     return SVSL_TYPE_NONE;
			if (!convert_to(c, args[1], svsl_type_scalar_id(types, svsl_scalar_int32))) return SVSL_TYPE_NONE;
			return elem;
		}
		if (arg_count < 1) { cerr(c, e->loc, "Load needs coordinates%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
		// sampled textures take int(dim+1) with mip; images and buffers take int(dim)
		int32_t n = obj->kind == svsl_type_texture ? coords + 1 :
		            obj->kind == svsl_type_buffer  ? 1 : coords;
		if (!convert_to(c, args[0], make_shape(types, svsl_scalar_int32, n))) return SVSL_TYPE_NONE;
		return elem;
	}
	case svsl_method_store: {
		if (obj->kind != svsl_type_image || arg_count != 2) { cerr(c, e->loc, "Store needs an image, coords, value%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
		if (!convert_to(c, args[0], coord_i)) return SVSL_TYPE_NONE;
		if (!convert_to(c, args[1], elem))    return SVSL_TYPE_NONE;
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });
	}
	case svsl_method_get_dimensions: {
		if (obj->kind == svsl_type_subpass || obj->kind == svsl_type_tileimage) {
			// attachments cover the framebuffer; there is no size to query
			cerr(c, e->loc, "'%.*s' has no dimensions to query", callee->member.name);
			return SVSL_TYPE_NONE;
		}
		int32_t n = obj->kind == svsl_type_buffer ? 1 :
		            (obj->dim == svsl_texdim_1d ? 1 : obj->dim == svsl_texdim_3d ? 3 : 2) +
		            (obj->arrayed ? 1 : 0);
		if (arg_count == 0) // value form: element count / dimensions of mip 0
			return make_shape(types, svsl_scalar_uint32, n);
		// buffers: (out count[, out stride]); multisampled textures:
		// (out w, out h[, out samples]); other textures also take the HLSL
		// mip-level overload (mip, out dims..., out levels)
		bool mip_form = obj->kind == svsl_type_texture && !obj->multisampled &&
		                arg_count == n + 2;
		bool ok_count = arg_count == n || mip_form ||
		                (obj->kind == svsl_type_buffer && arg_count == 2) ||
		                (obj->kind == svsl_type_texture && obj->multisampled && arg_count == n + 1);
		if (!ok_count) {
			if (obj->kind == svsl_type_buffer)
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
				              "GetDimensions on '%s' takes (out count) or (out count, out stride)",
				              svsl_type_name(types, obj_type));
			else
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
				              "GetDimensions on '%s' takes %d output arguments%s",
				              svsl_type_name(types, obj_type), n,
				              obj->kind == svsl_type_texture
				              ? ", or a leading mip level plus a trailing level count" : "");
			return SVSL_TYPE_NONE;
		}
		if (mip_form && !convert_to(c, args[0], svsl_type_scalar_id(types, svsl_scalar_uint32)))
			return SVSL_TYPE_NONE;
		for (int32_t i = mip_form ? 1 : 0; i < arg_count; i++) {
			if (!expr_is_lvalue(c, args[i]) ||
			    svsl_type_get(types, args[i]->sema_type)->kind != svsl_type_scalar) {
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, args[i]->loc,
				              "GetDimensions argument %d must be a writable scalar variable", i + 1);
				return SVSL_TYPE_NONE;
			}
		}
		return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });
	}
	case svsl_method_query_lod: {
		if (obj->kind != svsl_type_texture || obj->multisampled) { cerr(c, e->loc, "'%.*s' needs a sampled texture", callee->member.name); return SVSL_TYPE_NONE; }
		if (arg_count != 2) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%.*s' takes 2 arguments (sampler, coords), got %d",
			              callee->member.name.len, callee->member.name.ptr, arg_count);
			return SVSL_TYPE_NONE;
		}
		if (svsl_type_get(types, args[0]->sema_type)->kind != svsl_type_sampler) {
			cerr(c, args[0]->loc, "first argument of '%.*s' must be a sampler", callee->member.name);
			return SVSL_TYPE_NONE;
		}
		// the coordinate excludes any array layer
		int32_t qn = obj->dim == svsl_texdim_1d ? 1 : obj->dim == svsl_texdim_2d ? 2 : 3;
		if (!convert_to(c, args[1], make_shape(types, svsl_scalar_float32, qn))) return SVSL_TYPE_NONE;
		return f32;
	}
	case svsl_method_atomic: {
		if (obj->kind != svsl_type_image || arg_count < 2) { cerr(c, e->loc, "image atomics need coords and a value%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
		if (!convert_to(c, args[0], coord_i)) return SVSL_TYPE_NONE;
		svsl_type_id_t scalar = svsl_type_scalar_id(types, elem_scal);
		if (!convert_to(c, args[1], scalar)) return SVSL_TYPE_NONE;
		if (arg_count == 3) { // out-param form returns void
			if (!expr_is_lvalue(c, args[2]))
				cerr(c, args[2]->loc, "atomic out-argument must be writable%.*s", (svsl_str_t){0});
			return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });
		}
		return scalar;
	}
	}
	return SVSL_TYPE_NONE;
}

// --- calls: intrinsics -----------------------------------------------------------------------------------

static void mul_err(check_t *c, const svsl_ast_expr_t *e, const char *why) {
	svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
	              "cannot mul('%s', '%s'): %s",
	              tname(c, e->call.args[0]->sema_type), tname(c, e->call.args[1]->sema_type), why);
}

static svsl_type_id_t check_mul(check_t *c, svsl_ast_expr_t *e) {
	svsl_types_t *types = &c->prog->types;
	if (e->call.arg_count != 2) { cerr(c, e->loc, "mul takes two arguments%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
	svsl_ast_expr_t *a = e->call.args[0];
	svsl_ast_expr_t *b = e->call.args[1];
	const svsl_type_t *ta = svsl_type_get(types, a->sema_type);
	const svsl_type_t *tb = svsl_type_get(types, b->sema_type);

	// scalar * anything → componentwise
	if (ta->kind == svsl_type_scalar) return b->sema_type;
	if (tb->kind == svsl_type_scalar) return a->sema_type;

	if (ta->kind == svsl_type_vector && tb->kind == svsl_type_vector) {
		if (ta->count != tb->count) { mul_err(c, e, "vector sizes must match"); return SVSL_TYPE_NONE; }
		return svsl_type_scalar_id(types, ta->scalar); // dot
	}
	if (ta->kind == svsl_type_vector && tb->kind == svsl_type_matrix) {
		if (ta->count != tb->rows) { mul_err(c, e, "vector size must match matrix rows"); return SVSL_TYPE_NONE; }
		return svsl_type_vector_id(types, tb->scalar, tb->cols);
	}
	if (ta->kind == svsl_type_matrix && tb->kind == svsl_type_vector) {
		if (ta->cols != tb->count) { mul_err(c, e, "matrix columns must match vector size"); return SVSL_TYPE_NONE; }
		return svsl_type_vector_id(types, ta->scalar, ta->rows);
	}
	if (ta->kind == svsl_type_matrix && tb->kind == svsl_type_matrix) {
		if (ta->cols != tb->rows) { mul_err(c, e, "left columns must match right rows"); return SVSL_TYPE_NONE; }
		return svsl_type_matrix_id(types, ta->scalar, ta->rows, tb->cols);
	}
	mul_err(c, e, "operands must be scalars, vectors, or matrices");
	return SVSL_TYPE_NONE;
}

static svsl_type_id_t check_special_intrinsic(check_t *c, svsl_ast_expr_t *e, const svsl_intrinsic_t *intr) {
	svsl_types_t     *types     = &c->prog->types;
	svsl_ast_expr_t **args      = e->call.args;
	int32_t           arg_count = e->call.arg_count;

	svsl_intr_tag_ tag = (svsl_intr_tag_)intr->tag;

	if (tag == svsl_intr_mul) return check_mul(c, e);

	if (tag == svsl_intr_transpose && arg_count == 1) {
		const svsl_type_t *t = svsl_type_get(types, args[0]->sema_type);
		if (t->kind != svsl_type_matrix) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "transpose needs a matrix, got '%s'", tname(c, args[0]->sema_type));
			return SVSL_TYPE_NONE;
		}
		return svsl_type_matrix_id(types, t->scalar, t->cols, t->rows);
	}
	if ((tag == svsl_intr_determinant || tag == svsl_intr_inverse) && arg_count == 1) {
		const svsl_type_t *t = svsl_type_get(types, args[0]->sema_type);
		if (t->kind != svsl_type_matrix || t->rows != t->cols) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "%s needs a square matrix, got '%s'",
			              intr->name, tname(c, args[0]->sema_type));
			return SVSL_TYPE_NONE;
		}
		return tag == svsl_intr_inverse ? args[0]->sema_type
		                                          : svsl_type_scalar_id(types, t->scalar);
	}
	if (tag == svsl_intr_select && arg_count == 3) {
		svsl_type_id_t common = common_shape(c, args[1], args[2]);
		if (common == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
		if (!convert_to(c, args[1], common) || !convert_to(c, args[2], common)) return SVSL_TYPE_NONE;
		int32_t n = shape_count(types, common);
		svsl_type_id_t cond = make_shape(types, svsl_scalar_bool,
		                                 shape_count(types, args[0]->sema_type) == 1 ? 1 : n);
		if (!convert_to(c, args[0], cond)) return SVSL_TYPE_NONE;
		return common;
	}
	if (tag == svsl_intr_pack2 || tag == svsl_intr_pack4) {
		int32_t n = tag == svsl_intr_pack2 ? 2 : 4;
		if (arg_count != 1 || !convert_to(c, args[0], svsl_type_vector_id(types, svsl_scalar_float32, n)))
			return SVSL_TYPE_NONE;
		return svsl_type_scalar_id(types, svsl_scalar_uint32);
	}

	// legacy D3D9 intrinsics: reject with direction instead of approximating
	if (tag == svsl_intr_legacy) {
		svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
		              "'%s' is a legacy D3D intrinsic with no SPIR-V equivalent; compute the terms directly",
		              intr->name);
		return SVSL_TYPE_NONE;
	}

	// out-parameter math: sincos(x, out s, out c), modf(x, out ip), frexp(x, out exp)
	if (tag == svsl_intr_sincos || tag == svsl_intr_modf || tag == svsl_intr_frexp) {
		bool    is_sincos = tag == svsl_intr_sincos;
		int32_t want      = is_sincos ? 3 : 2;
		if (arg_count != want) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%s' takes %d arguments, got %d", intr->name, want, arg_count);
			return SVSL_TYPE_NONE;
		}
		int32_t nshape = shape_count(types, args[0]->sema_type);
		if (nshape == 0) { cerr(c, args[0]->loc, "'%.*s' needs a scalar or vector", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
		if (!scalar_is_float(shape_scalar(types, args[0]->sema_type)) &&
		    !convert_to(c, args[0], make_shape(types, svsl_scalar_float32, nshape)))
			return SVSL_TYPE_NONE;
		svsl_type_id_t g = args[0]->sema_type;
		for (int32_t i = 1; i < want; i++) {
			if (!expr_is_lvalue(c, args[i]) || args[i]->sema_type != g) {
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, args[i]->loc,
				              "'%s' output argument must be a writable '%s'",
				              intr->name, svsl_type_name(types, g));
				return SVSL_TYPE_NONE;
			}
		}
		return is_sincos ? svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void }) : g;
	}

	// atomics: (dest lvalue int/uint scalar, value...) — native returns the prior value
	if (SVSL_INTR_IS_ATOMIC(tag)) {
		bool    is_alias = intr->opt_native != NULL;
		bool    is_cmpxchg = tag == svsl_intr_atomic_cmpxchg;
		int32_t base_args  = is_cmpxchg ? 3 : 2;
		// a native atomic may take one extra trailing arg: a memory-order name
		bool    has_order  = !is_alias && arg_count == base_args + 1 &&
		                     args[arg_count - 1]->kind == svsl_expr_ident &&
		                     svsl_atomic_order(args[arg_count - 1]->ident) >= 0;
		int32_t max_extra  = is_alias ? 1 : (has_order ? 1 : 0);
		if (arg_count < base_args || arg_count > base_args + max_extra) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%s' takes %d arguments (destination, %svalue%s), got %d",
			              intr->name, base_args, is_cmpxchg ? "compare, " : "",
			              is_alias ? "[, out original]" : " [, memory order]", arg_count);
			return SVSL_TYPE_NONE;
		}
		// Vulkan's memory model has no sequential consistency — reject it up front
		// rather than emit SPIR-V that spirv-val refuses (the strongest is acq_rel)
		if (has_order && svsl_atomic_order(args[arg_count - 1]->ident) == svsl_mem_order_seq_cst)
			cerr(c, args[arg_count - 1]->loc,
			     "sequentially-consistent order isn't available on Vulkan; the strongest is acq_rel%.*s",
			     (svsl_str_t){0});
		svsl_type_id_t     dest = args[0]->sema_type;
		const svsl_type_t *td   = svsl_type_get(types, dest);
		// float32 destinations: add/min/max/exchange only (SPV_EXT_shader_atomic_float
		// has no float sub, bitwise, or compare-exchange)
		bool float_ok = td->kind == svsl_type_scalar && td->scalar == svsl_scalar_float32 &&
		                (tag == svsl_intr_atomic_add || tag == svsl_intr_atomic_min ||
		                 tag == svsl_intr_atomic_max || tag == svsl_intr_atomic_exchange);
		if (!float_ok &&
		    (td->kind != svsl_type_scalar || !scalar_is_int(td->scalar) || svsl_scalar_size(td->scalar) != 4)) {
			if (td->kind == svsl_type_scalar && td->scalar == svsl_scalar_float32)
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, args[0]->loc,
				              "'%s' has no float form; float atomics support add, min, max, and exchange",
				              intr->name);
			else
				svsl_diag_add(c->arena, c->diags, svsl_severity_error, args[0]->loc,
				              "atomic destination must be a 32-bit integer (or float for add/min/max/exchange), got '%s'",
				              tname(c, dest));
			return SVSL_TYPE_NONE;
		}
		if (!expr_is_lvalue(c, args[0]))
			cerr(c, args[0]->loc, "atomic destination must be writable%.*s", (svsl_str_t){0});
		for (int32_t i = 1; i < base_args; i++)
			if (!convert_to(c, args[i], dest)) return SVSL_TYPE_NONE;
		if (is_alias && arg_count == base_args + 1) { // Interlocked* out-param form
			if (!expr_is_lvalue(c, args[base_args]))
				cerr(c, args[base_args]->loc, "atomic out-argument must be writable%.*s", (svsl_str_t){0});
			return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });
		}
		return is_alias ? svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void }) : dest;
	}

	cerr(c, e->loc, "no overload of '%.*s' matches these arguments", svsl_str(intr->name));
	return SVSL_TYPE_NONE;
}

static svsl_type_id_t check_intrinsic_call(check_t *c, svsl_ast_expr_t *e, int32_t index) {
	const svsl_intrinsic_t *intr  = svsl_intrinsic_get(index);
	svsl_types_t           *types = &c->prog->types;
	svsl_ast_expr_t       **args  = e->call.args;

	e->call.callee->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_intrinsic, .a = index };

	// opt_native names an intrinsic's SVSL-native spelling — i.e. this row is a legacy alias
	if (c->prog->porting && intr->opt_native)
		svsl_diag_add(c->arena, c->diags, svsl_severity_porting, e->loc,
		              "legacy '%s' -> '%s'", intr->name, intr->opt_native);
	for (int32_t i = 0; i < e->call.arg_count; i++) {
		// a trailing memory-order name on a native atomic (atomic_add(p, v, acquire))
		// is not a value operand — it names the ordering, resolved in the atomic path
		if (SVSL_INTR_IS_ATOMIC(intr->tag) && intr->opt_native == NULL &&
		    i == e->call.arg_count - 1 && args[i]->kind == svsl_expr_ident &&
		    svsl_atomic_order(args[i]->ident) >= 0)
			continue;
		if (check_expr(c, args[i]) == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
	}

	if (intr->result == svsl_ires_special)
		return check_special_intrinsic(c, e, intr);

	// count declared params
	int32_t param_count = 0;
	while (param_count < 5 && intr->params[param_count] != svsl_iparam_end) param_count++;
	if (e->call.arg_count != param_count) {
		svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
		              "'%s' takes %d argument%s, got %d",
		              intr->name, param_count, param_count == 1 ? "" : "s", e->call.arg_count);
		return SVSL_TYPE_NONE;
	}

	// HLSL intrinsics promote arguments: the generic binds to the widest shape
	// and highest-rank scalar across all generic/same parameters, so that
	// lerp(0.04h, albedo.rgb, metal) is a float3 lerp, not a truncation
	svsl_type_id_t gen = SVSL_TYPE_NONE;
	{
		svsl_scalar_ scal      = svsl_scalar_bool;
		int32_t      count_max = 0;
		bool         any       = false;
		for (int32_t i = 0; i < param_count; i++) {
			svsl_iparam_ p = (svsl_iparam_)intr->params[i];
			if (p != svsl_iparam_genf && p != svsl_iparam_gen && p != svsl_iparam_geni &&
			    p != svsl_iparam_genb && p != svsl_iparam_same && p != svsl_iparam_vec3f)
				continue;
			int32_t n = shape_count(types, args[i]->sema_type);
			if (n == 0) continue;
			svsl_scalar_ as = shape_scalar(types, args[i]->sema_type);
			if (!any || scalar_rank(as) > scalar_rank(scal)) scal = as;
			if (n > count_max) count_max = n;
			any = true;
		}
		if (any) gen = make_shape(types, scal, count_max);
	}

	for (int32_t i = 0; i < param_count; i++) {
		svsl_type_id_t at = args[i]->sema_type;
		switch ((svsl_iparam_)intr->params[i]) {
		case svsl_iparam_genf:
		case svsl_iparam_gen:
		case svsl_iparam_geni:
		case svsl_iparam_genb:
		case svsl_iparam_vec3f:
		case svsl_iparam_any32: {
			int32_t n = shape_count(types, at);
			if (n == 0) { cerr(c, args[i]->loc, "'%.*s' needs a scalar or vector", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
			svsl_scalar_ scal = shape_scalar(types, at);
			svsl_iparam_ p    = (svsl_iparam_)intr->params[i];
			if (p == svsl_iparam_genf && !scalar_is_float(scal)) {
				// ints promote to float implicitly: sqrt(2) etc.
				at   = make_shape(types, svsl_scalar_float32, n);
				if (!convert_to(c, args[i], at)) return SVSL_TYPE_NONE;
				scal = svsl_scalar_float32;
			}
			if (p == svsl_iparam_geni && !scalar_is_int(scal)) { cerr(c, args[i]->loc, "'%.*s' needs integers", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
			if (p == svsl_iparam_genb && scal != svsl_scalar_bool) {
				at = make_shape(types, svsl_scalar_bool, n);
				if (!convert_to(c, args[i], at)) return SVSL_TYPE_NONE;
			}
			if (p == svsl_iparam_gen && scal == svsl_scalar_bool) { cerr(c, args[i]->loc, "'%.*s' needs numeric input", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
			if (p == svsl_iparam_vec3f && (n != 3 || !scalar_is_float(scal))) { cerr(c, args[i]->loc, "'%.*s' needs float3", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
			if (p == svsl_iparam_any32 && svsl_scalar_size(scal) != 4) { cerr(c, args[i]->loc, "'%.*s' needs 32-bit input", svsl_str(intr->name)); return SVSL_TYPE_NONE; }
			if (gen == SVSL_TYPE_NONE) gen = args[i]->sema_type;
			// promote to the common generic shape (any32 keeps its own type: bit casts)
			if (p != svsl_iparam_any32 && p != svsl_iparam_vec3f && at != gen) {
				// genf may have re-typed the arg to float; re-fetch before converting
				if (!convert_to(c, args[i], p == svsl_iparam_genb
				                            ? make_shape(types, svsl_scalar_bool, shape_count(types, gen))
				                            : gen)) return SVSL_TYPE_NONE;
			}
			break;
		}
		case svsl_iparam_same:
			if (!convert_to(c, args[i], gen)) return SVSL_TYPE_NONE;
			break;
		case svsl_iparam_scalar:
			if (!convert_to(c, args[i], svsl_type_scalar_id(types, shape_scalar(types, gen)))) return SVSL_TYPE_NONE;
			break;
		case svsl_iparam_uint:
			if (!convert_to(c, args[i], svsl_type_scalar_id(types, svsl_scalar_uint32))) return SVSL_TYPE_NONE;
			break;
		case svsl_iparam_bool:
			if (!convert_to(c, args[i], svsl_type_scalar_id(types, svsl_scalar_bool))) return SVSL_TYPE_NONE;
			break;
		default:
			break;
		}
	}

	int32_t gn = gen != SVSL_TYPE_NONE ? shape_count(types, gen) : 1;
	switch ((svsl_ires_)intr->result) {
	case svsl_ires_gen:         return gen;
	case svsl_ires_scalar:      return svsl_type_scalar_id(types, shape_scalar(types, gen));
	case svsl_ires_bool_shape:  return make_shape(types, svsl_scalar_bool, gn);
	case svsl_ires_bool:        return svsl_type_scalar_id(types, svsl_scalar_bool);
	case svsl_ires_uint_shape:  return make_shape(types, svsl_scalar_uint32, gn);
	case svsl_ires_int_shape:   return make_shape(types, svsl_scalar_int32, gn);
	case svsl_ires_float_shape: return make_shape(types, svsl_scalar_float32, gn);
	case svsl_ires_uint:        return svsl_type_scalar_id(types, svsl_scalar_uint32);
	case svsl_ires_uint4:       return svsl_type_vector_id(types, svsl_scalar_uint32, 4);
	case svsl_ires_float2:      return svsl_type_vector_id(types, svsl_scalar_float32, 2);
	case svsl_ires_float4:      return svsl_type_vector_id(types, svsl_scalar_float32, 4);
	case svsl_ires_void:        return svsl_type_intern(types, (svsl_type_t){ .kind = svsl_type_void });
	default:                    return SVSL_TYPE_NONE;
	}
}

// --- calls: user functions ---------------------------------------------------------------------------------

static svsl_type_id_t check_user_call(check_t *c, svsl_ast_expr_t *e) {
	svsl_str_t name = e->call.callee->ident;

	int32_t best      = -1;
	int32_t best_cost = 0x7fffffff;
	for (int32_t f = 0; f < c->prog->functions.count; f++) {
		const svsl_func_info_t *info = &c->prog->functions.items[f];
		if (!svsl_str_eq(info->func->name, name)) continue;
		if (info->func->param_count != e->call.arg_count) continue;
		int32_t cost = 0;
		bool    ok   = true;
		for (int32_t i = 0; i < e->call.arg_count && ok; i++) {
			int32_t k = conv_cost(&c->prog->types, e->call.args[i]->sema_type, info->param_types[i]);
			if (k < 0) ok = false;
			else       cost += k;
		}
		if (ok && cost < best_cost) { best = f; best_cost = cost; }
	}
	if (best < 0) {
		char    sig[256] = "";
		int32_t len      = 0;
		for (int32_t i = 0; i < e->call.arg_count && len < (int32_t)sizeof(sig) - 8; i++)
			len += snprintf(sig + len, sizeof(sig) - (size_t)len, "%s%s",
			                i ? ", " : "", tname(c, e->call.args[i]->sema_type));
		svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
		              "no matching function for '%.*s(%s)'", name.len, name.ptr, sig);
		return SVSL_TYPE_NONE;
	}

	const svsl_func_info_t *info = &c->prog->functions.items[best];
	for (int32_t i = 0; i < e->call.arg_count; i++) {
		uint8_t dir = info->func->params[i]->dir;
		if (dir == svsl_dir_out || dir == svsl_dir_inout) {
			if (!expr_is_lvalue(c, e->call.args[i]) || swizzle_write_has_dup(e->call.args[i]))
				cerr(c, e->call.args[i]->loc, "argument for out parameter '%.*s' must be writable",
				     info->func->params[i]->name);
			// out args keep their own type; a copy-back conversion happens at inline time
			if (conv_cost(&c->prog->types, e->call.args[i]->sema_type, info->param_types[i]) != 0)
				cerr(c, e->call.args[i]->loc, "out argument type must match parameter '%.*s' exactly",
				     info->func->params[i]->name);
		} else {
			if (!convert_to(c, e->call.args[i], info->param_types[i])) return SVSL_TYPE_NONE;
		}
	}
	e->call.callee->sema_ref = (svsl_sema_ref_t){ .kind = svsl_ref_function, .a = best };

	svsl_array_push(c->arena, &c->call_edges, c->func_index);
	svsl_array_push(c->arena, &c->call_edges, best);
	return info->return_type;
}

// --- expressions --------------------------------------------------------------------------------------------

static bool binary_is_assign(svsl_tok_ op) {
	return op == svsl_tok_assign || op == svsl_tok_plus_assign || op == svsl_tok_minus_assign ||
	       op == svsl_tok_star_assign || op == svsl_tok_slash_assign || op == svsl_tok_percent_assign ||
	       op == svsl_tok_and_assign || op == svsl_tok_or_assign || op == svsl_tok_xor_assign ||
	       op == svsl_tok_shl_assign || op == svsl_tok_shr_assign;
}

static svsl_type_id_t check_binary(check_t *c, svsl_ast_expr_t *e) {
	svsl_types_t *types = &c->prog->types;
	svsl_tok_     op    = e->binary.op;

	svsl_type_id_t lt = check_expr(c, e->binary.lhs);
	svsl_type_id_t rt = check_expr(c, e->binary.rhs);
	if (lt == SVSL_TYPE_NONE || rt == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;

	if (binary_is_assign(op)) {
		if (!expr_is_lvalue(c, e->binary.lhs))
			cerr(c, e->binary.lhs->loc, "cannot assign to this expression%.*s", (svsl_str_t){0});
		else if (swizzle_write_has_dup(e->binary.lhs))
			cerr(c, e->binary.lhs->loc, "swizzle '.%.*s' writes a component more than once",
			     e->binary.lhs->member.name);
		if (!convert_to(c, e->binary.rhs, lt)) return SVSL_TYPE_NONE;
		return lt;
	}
	if (op == svsl_tok_comma) return rt;

	if (op == svsl_tok_andand || op == svsl_tok_oror) {
		svsl_type_id_t common = common_shape(c, e->binary.lhs, e->binary.rhs);
		if (common == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
		svsl_type_id_t b = make_shape(types, svsl_scalar_bool, shape_count(types, common));
		if (!convert_to(c, e->binary.lhs, b) || !convert_to(c, e->binary.rhs, b)) return SVSL_TYPE_NONE;
		return b;
	}

	bool comparison = op == svsl_tok_eq || op == svsl_tok_neq || op == svsl_tok_lt ||
	                  op == svsl_tok_gt || op == svsl_tok_le || op == svsl_tok_ge;
	bool bitwise    = op == svsl_tok_amp || op == svsl_tok_pipe || op == svsl_tok_caret ||
	                  op == svsl_tok_shl || op == svsl_tok_shr;

	// matrix arithmetic: componentwise with a matching matrix or a scalar
	const svsl_type_t *ltt = svsl_type_get(types, lt);
	const svsl_type_t *rtt = svsl_type_get(types, rt);
	if (!comparison && (ltt->kind == svsl_type_matrix || rtt->kind == svsl_type_matrix)) {
		if (ltt->kind == svsl_type_matrix && rtt->kind == svsl_type_matrix) {
			if (lt != rt) { cerr(c, e->loc, "matrix shape mismatch%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
			return lt;
		}
		const svsl_type_t *s = ltt->kind == svsl_type_matrix ? rtt : ltt;
		if (s->kind != svsl_type_scalar) { cerr(c, e->loc, "matrix op needs a matrix or scalar%.*s", (svsl_str_t){0}); return SVSL_TYPE_NONE; }
		return ltt->kind == svsl_type_matrix ? lt : rt;
	}

	svsl_type_id_t common = common_shape(c, e->binary.lhs, e->binary.rhs);
	if (common == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
	if (bitwise && !scalar_is_int(shape_scalar(types, common))) {
		cerr(c, e->loc, "bitwise operators need integers%.*s", (svsl_str_t){0});
		return SVSL_TYPE_NONE;
	}
	if (!convert_to(c, e->binary.lhs, common) || !convert_to(c, e->binary.rhs, common))
		return SVSL_TYPE_NONE;
	return comparison ? make_shape(types, svsl_scalar_bool, shape_count(types, common)) : common;
}

static svsl_type_id_t check_expr(check_t *c, svsl_ast_expr_t *e) {
	svsl_types_t *types = &c->prog->types;
	svsl_type_id_t result = SVSL_TYPE_NONE;

	switch (e->kind) {
	case svsl_expr_int_lit:
		result = svsl_type_scalar_id(types,
			e->int_lit.suffix == svsl_suffix_u  ? svsl_scalar_uint32 :
			e->int_lit.suffix == svsl_suffix_l  ? svsl_scalar_int64 :
			e->int_lit.suffix == svsl_suffix_ul ? svsl_scalar_uint64 : svsl_scalar_int32);
		break;
	case svsl_expr_float_lit:
		result = svsl_type_scalar_id(types,
			e->float_lit.suffix == svsl_suffix_h  ? svsl_scalar_half :
			e->float_lit.suffix == svsl_suffix_lf ? svsl_scalar_float64 : svsl_scalar_float32);
		break;
	case svsl_expr_bool_lit:
		result = svsl_type_scalar_id(types, svsl_scalar_bool);
		break;
	case svsl_expr_string_lit:
		cerr(c, e->loc, "string literals are only valid in attributes%.*s", (svsl_str_t){0});
		break;
	case svsl_expr_ident:
		result = resolve_ident(c, e);
		break;
	case svsl_expr_member:
		result = check_member(c, e);
		break;
	case svsl_expr_index: {
		svsl_type_id_t obj = check_expr(c, e->index.object);
		svsl_type_id_t idx = check_expr(c, e->index.index);
		if (obj == SVSL_TYPE_NONE || idx == SVSL_TYPE_NONE) break;
		const svsl_type_t *t = svsl_type_get(types, obj);
		if (t->kind == svsl_type_image) { // img[coord] load/store
			convert_to(c, e->index.index, make_shape(types, svsl_scalar_int32, texdim_coord_count(t)));
			result = t->elem;
			break;
		}
		if (!convert_to(c, e->index.index,
		                shape_count(types, idx) == 1 && shape_scalar(types, idx) == svsl_scalar_uint32
		                ? svsl_type_scalar_id(types, svsl_scalar_uint32)
		                : svsl_type_scalar_id(types, svsl_scalar_int32))) break;
		switch (t->kind) {
		case svsl_type_array:  result = t->elem; break;
		case svsl_type_buffer: result = t->elem; break;
		case svsl_type_vector: result = svsl_type_scalar_id(types, t->scalar); break;
		case svsl_type_matrix: result = svsl_type_vector_id(types, t->scalar, t->cols); break; // HLSL m[i] = row i
		default:
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "cannot index a value of type '%s'", tname(c, obj));
			break;
		}
		break;
	}
	case svsl_expr_unary: {
		svsl_type_id_t ot = check_expr(c, e->unary.operand);
		if (ot == SVSL_TYPE_NONE) break;
		if (e->unary.op == svsl_tok_not) {
			int32_t n = shape_count(types, ot);
			if (n == 0) { cerr(c, e->loc, "'!' needs a scalar or vector%.*s", (svsl_str_t){0}); break; }
			svsl_type_id_t b = make_shape(types, svsl_scalar_bool, n);
			if (convert_to(c, e->unary.operand, b)) result = b;
			break;
		}
		if (e->unary.op == svsl_tok_tilde && !scalar_is_int(shape_scalar(types, ot))) {
			cerr(c, e->loc, "'~' needs integers%.*s", (svsl_str_t){0});
			break;
		}
		if ((e->unary.op == svsl_tok_plusplus || e->unary.op == svsl_tok_minusminus) &&
		    (!expr_is_lvalue(c, e->unary.operand) || swizzle_write_has_dup(e->unary.operand)))
			cerr(c, e->loc, "'++'/'--' needs a writable value%.*s", (svsl_str_t){0});
		result = ot;
		break;
	}
	case svsl_expr_post: {
		svsl_type_id_t ot = check_expr(c, e->post.operand);
		if (ot == SVSL_TYPE_NONE) break;
		if (!expr_is_lvalue(c, e->post.operand) || swizzle_write_has_dup(e->post.operand))
			cerr(c, e->loc, "'++'/'--' needs a writable value%.*s", (svsl_str_t){0});
		result = ot;
		break;
	}
	case svsl_expr_ternary: {
		svsl_type_id_t ct = check_expr(c, e->ternary.cond);
		svsl_type_id_t at = check_expr(c, e->ternary.then_expr);
		svsl_type_id_t bt = check_expr(c, e->ternary.else_expr);
		if (ct == SVSL_TYPE_NONE || at == SVSL_TYPE_NONE || bt == SVSL_TYPE_NONE) break;
		svsl_type_id_t common = common_shape(c, e->ternary.then_expr, e->ternary.else_expr);
		if (common == SVSL_TYPE_NONE) break;
		if (!convert_to(c, e->ternary.then_expr, common) ||
		    !convert_to(c, e->ternary.else_expr, common)) break;
		svsl_type_id_t cond = make_shape(types, svsl_scalar_bool,
		                                 shape_count(types, ct) == 1 ? 1 : shape_count(types, common));
		if (!convert_to(c, e->ternary.cond, cond)) break;
		result = common;
		break;
	}
	case svsl_expr_binary:
		result = check_binary(c, e);
		break;
	case svsl_expr_cast: {
		svsl_type_id_t ot = check_expr(c, e->cast.operand);
		if (ot == SVSL_TYPE_NONE) break;
		// resolve the target through sema's own resolver (kept in sema.c); the cast
		// target was already validated syntactically, so resolve leniently here
		extern svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *arena, svsl_program_t *prog,
		                                             svsl_diag_list_t *diags, const svsl_ast_type_t *ref);
		svsl_type_id_t to = svsl_sema_resolve_type(c->arena, c->prog, c->diags, e->cast.type);
		if (to == SVSL_TYPE_NONE) break;
		const svsl_type_t *tt = svsl_type_get(types, to);
		const svsl_type_t *ft = svsl_type_get(types, ot);
		bool ok =
			(shape_count(types, to) > 0 && shape_count(types, ot) > 0 &&
			 (shape_count(types, ot) == 1 || shape_count(types, to) <= shape_count(types, ot))) ||
			(tt->kind == svsl_type_matrix && ft->kind == svsl_type_matrix &&
			 tt->rows <= ft->rows && tt->cols <= ft->cols) ||
			(tt->kind == svsl_type_matrix && shape_count(types, ot) == 1) || // (float3x3)0
			(tt->kind == svsl_type_struct && shape_count(types, ot) == 1);   // (psIn)0 fill
		if (!ok) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc, "cannot cast '%s' to '%s'",
			              svsl_type_name(types, ot), svsl_type_name(types, to));
			break;
		}
		result = to;
		break;
	}
	case svsl_expr_ctor: {
		extern svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *arena, svsl_program_t *prog,
		                                             svsl_diag_list_t *diags, const svsl_ast_type_t *ref);
		svsl_type_id_t to = svsl_sema_resolve_type(c->arena, c->prog, c->diags, e->ctor.type);
		if (to == SVSL_TYPE_NONE) break;
		const svsl_type_t *t = svsl_type_get(types, to);

		int32_t want = t->kind == svsl_type_vector ? t->count :
		               t->kind == svsl_type_matrix ? t->rows * t->cols :
		               t->kind == svsl_type_scalar ? 1 : -1;
		if (want < 0) {
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%s' has no constructor; only scalars, vectors, and matrices do",
			              tname(c, to));
			break;
		}

		int32_t have = 0;
		bool    bad  = false;
		for (int32_t i = 0; i < e->ctor.arg_count; i++) {
			svsl_type_id_t at = check_expr(c, e->ctor.args[i]);
			if (at == SVSL_TYPE_NONE) { bad = true; break; }
			int32_t n = shape_count(types, at);
			if (n == 0) { cerr(c, e->ctor.args[i]->loc, "constructor arguments must be scalars or vectors%.*s", (svsl_str_t){0}); bad = true; break; }
			have += n;
		}
		if (bad) break;
		// scalar broadcast (float4(1)) and matrix diagonal (float4x4(1)) take one scalar
		if (!(have == want || (e->ctor.arg_count == 1 && have == 1)))
			svsl_diag_add(c->arena, c->diags, svsl_severity_error, e->loc,
			              "'%s' constructor needs %d components (or 1 to broadcast), got %d",
			              tname(c, to), want, have);
		result = to;
		break;
	}
	case svsl_expr_call: {
		svsl_ast_expr_t *callee = e->call.callee;
		if (callee->kind == svsl_expr_member) {
			result = check_method_call(c, e);
			break;
		}
		// user functions shadow intrinsics of the same name
		bool user_exists = false;
		for (int32_t f = 0; f < c->prog->functions.count; f++)
			if (svsl_str_eq(c->prog->functions.items[f].func->name, callee->ident)) user_exists = true;
		if (user_exists) {
			for (int32_t i = 0; i < e->call.arg_count; i++)
				if (check_expr(c, e->call.args[i]) == SVSL_TYPE_NONE) return SVSL_TYPE_NONE;
			result = check_user_call(c, e);
			break;
		}
		int32_t intr = svsl_intrinsic_find(callee->ident);
		if (intr >= 0) {
			result = check_intrinsic_call(c, e, intr);
			break;
		}
		cerr(c, callee->loc, "unknown function '%.*s'", callee->ident);
		break;
	}
	case svsl_expr_init_list:
		cerr(c, e->loc, "initializer lists are only valid as initializers%.*s", (svsl_str_t){0});
		break;
	case svsl_expr_spirv_asm: {
		extern svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *arena, svsl_program_t *prog,
		                                             svsl_diag_list_t *diags, const svsl_ast_type_t *ref);
		svsl_type_id_t rt = svsl_sema_resolve_type(c->arena, c->prog, c->diags, e->spirv_asm.result_type);
		if (rt == SVSL_TYPE_NONE) break;
		bool ok = true, has_result = false;
		for (int32_t k = 0; k < e->spirv_asm.inst_count; k++) {
			svsl_ast_spv_inst_t *inst = &e->spirv_asm.insts[k];
			inst->spv_op = svsl_spirv_opcode_lookup(inst->opcode);
			if (inst->spv_op < 0) {
				cerr(c, inst->loc, "unknown SPIR-V opcode '%.*s'", inst->opcode);
				ok = false;
			}
			for (int32_t o = 0; o < inst->operand_count; o++) {
				svsl_ast_spv_operand_t *op = &inst->operands[o];
				if (op->kind == svsl_spv_operand_value) {
					if (check_expr(c, op->value) == SVSL_TYPE_NONE) ok = false;
				} else if (op->kind == svsl_spv_operand_type) {
					op->type_id = svsl_sema_resolve_type(c->arena, c->prog, c->diags, op->type);
					if (op->type_id == SVSL_TYPE_NONE) ok = false;
				} else if (op->kind == svsl_spv_operand_local &&
				           svsl_str_eq_cstr(op->local, "result")) {
					has_result = true;
				}
			}
		}
		if (ok && !has_result) {
			cerr(c, e->loc, "spirv_asm block defines no %%result id to carry its value%.*s",
			     (svsl_str_t){0});
			ok = false;
		}
		if (ok) result = rt;
		break;
	}
	}

	e->sema_type = result;
	return result;
}

// --- statements ------------------------------------------------------------------------------------------------

// exposed by sema.c for local declarations and cast/ctor targets
extern svsl_type_id_t svsl_sema_resolve_type(svsl_arena_t *arena, svsl_program_t *prog,
                                             svsl_diag_list_t *diags, const svsl_ast_type_t *ref);

static void check_stmt(check_t *c, svsl_ast_stmt_t *s);

static void check_block(check_t *c, svsl_ast_stmt_t **stmts, int32_t count) {
	c->depth++;
	int32_t mark = c->locals.count;
	for (int32_t i = 0; i < count; i++)
		check_stmt(c, stmts[i]);
	c->locals.count = mark; // pop scope
	c->depth--;
}

static void check_var_decl(check_t *c, svsl_ast_stmt_t *s) {
	for (int32_t i = 0; i < s->var_decl.count; i++) {
		svsl_ast_var_t *var  = s->var_decl.vars[i];
		svsl_type_id_t  type = svsl_sema_resolve_type(c->arena, c->prog, c->diags, var->type);
		if (type == SVSL_TYPE_NONE) continue;
		const svsl_type_t *t = svsl_type_get(&c->prog->types, type);
		if (svsl_type_is_resource(t)) {
			cerr(c, var->loc, "resources cannot be declared locally ('%.*s')", var->name);
			continue;
		}
		if (var->init && var->init->kind != svsl_expr_init_list) {
			if (check_expr(c, var->init) != SVSL_TYPE_NONE)
				convert_to(c, var->init, type);
		} else if (var->init) { // initializer list: check items only (IR assembles them)
			for (int32_t k = 0; k < var->init->init_list.count; k++)
				if (var->init->init_list.items[k]->kind != svsl_expr_init_list)
					check_expr(c, var->init->init_list.items[k]);
			var->init->sema_type = type;
		}
		svsl_array_push(c->arena, &c->locals, (check_local_t){
			.name = var->name, .type = type, .depth = c->depth });
	}
}

static void check_cond(check_t *c, svsl_ast_expr_t *cond) {
	if (!cond) return;
	if (check_expr(c, cond) == SVSL_TYPE_NONE) return;
	convert_to(c, cond, svsl_type_scalar_id(&c->prog->types, svsl_scalar_bool));
}

static void check_stmt(check_t *c, svsl_ast_stmt_t *s) {
	if (!s) return;
	svsl_attrs_check(c->arena, c->diags, &s->attrs, svsl_attr_ctx_stmt, c->prog->porting);
	switch (s->kind) {
	case svsl_stmt_block:
		check_block(c, s->block.stmts, s->block.count);
		break;
	case svsl_stmt_expr:
		if (s->expr) check_expr(c, s->expr);
		break;
	case svsl_stmt_var_decl:
		check_var_decl(c, s);
		break;
	case svsl_stmt_if:
		check_cond(c, s->if_stmt.cond);
		check_stmt(c, s->if_stmt.then_stmt);
		check_stmt(c, s->if_stmt.else_stmt);
		break;
	case svsl_stmt_for: {
		c->depth++;
		int32_t mark = c->locals.count;
		check_stmt(c, s->for_stmt.init);
		check_cond(c, s->for_stmt.cond);
		if (s->for_stmt.inc) check_expr(c, s->for_stmt.inc);
		check_stmt(c, s->for_stmt.body);
		c->locals.count = mark;
		c->depth--;
		break;
	}
	case svsl_stmt_while:
	case svsl_stmt_do:
		check_cond(c, s->while_stmt.cond);
		check_stmt(c, s->while_stmt.body);
		break;
	case svsl_stmt_switch: {
		if (check_expr(c, s->switch_stmt.value) != SVSL_TYPE_NONE) {
			const svsl_type_t *t = svsl_type_get(&c->prog->types, s->switch_stmt.value->sema_type);
			if (t->kind != svsl_type_scalar || !scalar_is_int(t->scalar))
				cerr(c, s->switch_stmt.value->loc, "switch value must be an integer%.*s", (svsl_str_t){0});
		}
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++) {
			svsl_ast_case_t *cs = &s->switch_stmt.cases[i];
			if (cs->value) check_expr(c, cs->value);
			check_block(c, cs->stmts, cs->stmt_count);
		}
		break;
	}
	case svsl_stmt_return: {
		if (c->return_type == SVSL_TYPE_NONE) { // unresolved return type, already errored
			if (s->return_value) check_expr(c, s->return_value);
			break;
		}
		const svsl_type_t *rt = svsl_type_get(&c->prog->types, c->return_type);
		if (s->return_value) {
			if (rt->kind == svsl_type_void)
				cerr(c, s->loc, "void function cannot return a value%.*s", (svsl_str_t){0});
			else if (check_expr(c, s->return_value) != SVSL_TYPE_NONE)
				convert_to(c, s->return_value, c->return_type);
		} else if (rt->kind != svsl_type_void) {
			cerr(c, s->loc, "non-void function must return a value%.*s", (svsl_str_t){0});
		}
		break;
	}
	case svsl_stmt_break:
	case svsl_stmt_continue:
	case svsl_stmt_discard:
	case svsl_stmt_demote:
		break;
	}
}

// --- recursion (full inlining requires a call DAG) ---------------------------------------------------------------

static bool find_cycle(check_t *c, int32_t func, uint8_t *state) {
	state[func] = 1; // visiting
	for (int32_t i = 0; i < c->call_edges.count; i += 2) {
		if (c->call_edges.items[i] != func) continue;
		int32_t callee = c->call_edges.items[i + 1];
		if (state[callee] == 1) return true;
		if (state[callee] == 0 && find_cycle(c, callee, state)) return true;
	}
	state[func] = 2;
	return false;
}

void svsl_check_functions(svsl_arena_t *arena, svsl_program_t *prog, svsl_diag_list_t *ref_diags) {
	check_t c = { .arena = arena, .prog = prog, .diags = ref_diags };

	for (int32_t f = 0; f < prog->functions.count; f++) {
		svsl_func_info_t *info = &prog->functions.items[f];
		if (!info->func->body) continue;

		c.func         = info->func;
		c.func_index   = f;
		c.return_type  = info->return_type;
		c.locals.count = 0;
		c.depth        = 0;
		check_stmt(&c, info->func->body);
		info->checked = true;
	}

	uint8_t *state = svsl_arena_alloc(arena, (size_t)(prog->functions.count > 0 ? prog->functions.count : 1));
	for (int32_t f = 0; f < prog->functions.count; f++) {
		memset(state, 0, (size_t)prog->functions.count);
		if (find_cycle(&c, f, state)) {
			svsl_diag_add(arena, ref_diags, svsl_severity_error, prog->functions.items[f].func->loc,
			              "recursion involving '%.*s' (SPIR-V forbids recursion)",
			              prog->functions.items[f].func->name.len, prog->functions.items[f].func->name.ptr);
			break;
		}
	}
}
