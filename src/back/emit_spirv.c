// IR → SPIR-V. Matches glslang's HLSL model where it matters for StereoKit:
// combined image samplers named after the texture (binding t+100), matrices as
// row-representation (SPIR-V vector i = HLSL row i, RowMajor layout, swapped
// mul operands), half = float32 + RelaxedPrecision, bindings b+0/t,s+100/u+200.

#include "emit_spirv.h"

#include "spirv_builder.h"
#include "../ir/usage.h"
#include "../ir/ir_operands.h"
#include "../tables/formats.h"
#include "../tables/intrinsics.h"
#include "../tables/semantics.h"
#include "../../vendor/GLSL.std.450.h"

#include <string.h>

// data-table rows initialize what they need; zero-fill is the point
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

typedef struct emit_t {
	svsl_arena_t         *arena;
	const svsl_program_t *prog;
	const svsl_ir_func_t *fn;
	svsl_diag_list_t     *diags;
	svsl_spv_t            spv;

	uint32_t *value_ids;    // per IR instruction
	uint32_t *value_class;  // storage class for pointer values
	// constant grade per value: 0 = runtime, 1 = plain constant,
	// 2 = specialization-dependent constant expression
	uint8_t  *value_spec;

	// per-svsl-type value-type ids (no layout decorations)
	uint32_t *type_ids;
	// layout-decorated composite types, unique per (type, layout)
	uint32_t *laid_ids[4];
	int32_t   type_cap; // size of type_ids/laid_ids; ids >= this were interned mid-emit
	uint8_t  *value_layout; // svsl_layout_ of the buffer a pointer points into

	// globals
	uint32_t *buffer_vars;
	uint32_t *resource_vars;    // combined-sampler/image/buffer variable per resource
	uint32_t *resource_img_type;// image type id (for OpImage/etc.)
	uint32_t *workgroup_ids;
	uint32_t *const_global_ids;
	uint32_t *param_shadow;     // entry params → function-local shadow vars
	// scalar-replaced struct params: read-only inputs accessed only by constant
	// member chains skip the shadow var + whole-struct construct entirely; each
	// member chain resolves straight to the member's stage-input variable.
	uint8_t  *param_sroa;       // per entry param: 1 = scalar-replaced
	uint32_t *param_member_var; // [param * SVSL_EMIT_MAX_MEMBERS + member] → input var (0 if unused)
	uint32_t *builtin_input;    // subgroup builtins etc., created on demand
	uint32_t *spec_const_ids;   // one OpSpecConstant per spec-constant index
	uint32_t *sampler_vars;     // standalone sampler variables for cross-paired sampling
	uint32_t *output_vars;      // flattened return outputs (sized to the return type)
	int32_t   output_count;
	int32_t  *vs_input_locations; // per prog->vertex_inputs entry; vertex stage only
	// scalar-replaced return struct: a local var whose members are stored then
	// returned whole is elided — member stores go straight to the stage-output
	// variables (output_vars), so no local struct is built and torn apart again.
	uint32_t  output_sroa_var;  // IR index of the SROA'd output var, or SVSL_IR_NONE

	svsl_array_t(uint32_t) interface;   // entry-point interface ids (Input/Output)
	svsl_array_t(uint32_t) relaxed_ids; // RelaxedPrecision decorated once per id

	// control flow
	struct cf_frame {
		uint8_t   kind; // 'i' if, 'l' loop, 's' switch
		uint32_t  merge, cont;
		uint32_t *case_labels; // loop: [0]=header; switch: one label per case (arena-sized)
		uint32_t  case_count;
		uint32_t  next_case; // emission cursor
		bool      cont_seen; // loop_continue marker emitted (wrapper loops have none)
	} *cf; // arena-sized to the function's max control-flow nesting
	int32_t  cf_depth;
	uint32_t current_block; // 0 = no open block
	bool     terminated;

	// pre-scan results: matching info for if/else
	uint8_t *if_has_else; // per inst index of svsl_ir_if

	bool    needs_depth_replacing;
	uint8_t depth_mode; // conservative depth: 1 = DepthGreater, 2 = DepthLess

	bool failed;
} emit_t;

static void eerr(emit_t *e, svsl_loc_t loc, const char *msg) {
	svsl_diag_add(e->arena, e->diags, svsl_severity_error, loc, "%s", msg);
	e->failed = true;
}

// --- svsl type → SPIR-V type ---------------------------------------------------

static uint32_t spv_type_for(emit_t *e, svsl_type_id_t id);

static uint32_t spv_scalar_type(emit_t *e, svsl_scalar_ scalar) {
	svsl_spv_t *spv = &e->spv;
	switch (scalar) {
	case svsl_scalar_bool:    return svsl_spv_type(spv, SpvOpTypeBool, NULL, 0);
	case svsl_scalar_int8:    svsl_spv_cap(spv, SpvCapabilityInt8);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 8, 1 }, 2);
	case svsl_scalar_uint8:   svsl_spv_cap(spv, SpvCapabilityInt8);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 8, 0 }, 2);
	case svsl_scalar_int16:   svsl_spv_cap(spv, SpvCapabilityInt16);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 16, 1 }, 2);
	case svsl_scalar_uint16:  svsl_spv_cap(spv, SpvCapabilityInt16);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 16, 0 }, 2);
	case svsl_scalar_int32:   return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 32, 1 }, 2);
	case svsl_scalar_uint32:  return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 32, 0 }, 2);
	case svsl_scalar_int64:   svsl_spv_cap(spv, SpvCapabilityInt64);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 64, 1 }, 2);
	case svsl_scalar_uint64:  svsl_spv_cap(spv, SpvCapabilityInt64);
	                          return svsl_spv_type(spv, SpvOpTypeInt, (uint32_t[]){ 64, 0 }, 2);
	case svsl_scalar_float16: svsl_spv_cap(spv, SpvCapabilityFloat16);
	                          return svsl_spv_type(spv, SpvOpTypeFloat, (uint32_t[]){ 16 }, 1);
	case svsl_scalar_float64: svsl_spv_cap(spv, SpvCapabilityFloat64);
	                          return svsl_spv_type(spv, SpvOpTypeFloat, (uint32_t[]){ 64 }, 1);
	case svsl_scalar_half:    // relaxed-precision float32
	case svsl_scalar_float32:
	default:                  return svsl_spv_type(spv, SpvOpTypeFloat, (uint32_t[]){ 32 }, 1);
	}
}

static uint32_t spv_uint_const(emit_t *e, uint32_t v) {
	uint32_t u32 = spv_scalar_type(e, svsl_scalar_uint32);
	return svsl_spv_const(&e->spv, u32, v, false, false);
}
static uint32_t spv_int_const(emit_t *e, int32_t v) {
	uint32_t i32 = spv_scalar_type(e, svsl_scalar_int32);
	return svsl_spv_const(&e->spv, i32, (uint32_t)v, false, false);
}
static uint32_t spv_float_const(emit_t *e, float v) {
	uint32_t f32 = spv_scalar_type(e, svsl_scalar_float32);
	uint32_t bits;
	memcpy(&bits, &v, 4);
	return svsl_spv_const(&e->spv, f32, bits, false, false);
}
// A storage image's format: the explicit template/attribute format if named
static SpvDim spv_dim(svsl_texdim_ dim) {
	switch (dim) {
	case svsl_texdim_1d:   return SpvDim1D;
	case svsl_texdim_3d:   return SpvDim3D;
	case svsl_texdim_cube: return SpvDimCube;
	case svsl_texdim_2d:
	default:               return SpvDim2D;
	}
}

// image type for a texture/image resource (sampled type is always scalar)
static uint32_t spv_image_type(emit_t *e, const svsl_type_t *t) {
	const svsl_type_t *elem   = svsl_type_get(&e->prog->types, t->elem);
	uint32_t           scalar = spv_scalar_type(e, elem->scalar); // sampled type is always scalar
	bool     storage = t->kind == svsl_type_image;
	uint32_t format  = storage ? svsl_image_format_for(&e->prog->types, t) : SpvImageFormatUnknown;
	if (storage && svsl_image_format_extended(format))
		svsl_spv_cap(&e->spv, SpvCapabilityStorageImageExtendedFormats);
	if (t->dim == svsl_texdim_1d) svsl_spv_cap(&e->spv, storage ? SpvCapabilityImage1D : SpvCapabilitySampled1D);
	uint32_t operands[8] = {
		scalar, (uint32_t)spv_dim(t->dim),
		0,                                // depth: 0 (not a depth image; Dref works regardless)
		t->arrayed ? 1u : 0u,
		t->multisampled ? 1u : 0u,
		storage ? 2u : 1u,                // sampled: 1 = sampled, 2 = storage
		format };
	return svsl_spv_type(&e->spv, SpvOpTypeImage, operands, 7);
}

static uint32_t spv_type_for(emit_t *e, svsl_type_id_t id) {
	if (id == SVSL_TYPE_NONE) return svsl_spv_type(&e->spv, SpvOpTypeVoid, NULL, 0);
	// types interned during emission land beyond the memoization cache; skip the
	// cache for them (svsl_spv_type still dedups the underlying SPIR-V type)
	bool cacheable = id < e->type_cap;
	if (cacheable && e->type_ids[id]) return e->type_ids[id];

	const svsl_type_t *t  = svsl_type_get(&e->prog->types, id);
	uint32_t           result = 0;
	switch (t->kind) {
	case svsl_type_void:
		result = svsl_spv_type(&e->spv, SpvOpTypeVoid, NULL, 0);
		break;
	case svsl_type_scalar:
		result = spv_scalar_type(e, t->scalar);
		break;
	case svsl_type_vector: {
		uint32_t scalar = spv_scalar_type(e, t->scalar);
		result = svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ scalar, t->count }, 2);
		break;
	}
	case svsl_type_matrix: {
		// SPIR-V vector i = HLSL row i: row vectors have `cols` components, `rows` of them
		uint32_t scalar = spv_scalar_type(e, t->scalar);
		uint32_t row    = svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ scalar, t->cols }, 2);
		result = svsl_spv_type(&e->spv, SpvOpTypeMatrix, (uint32_t[]){ row, t->rows }, 2);
		break;
	}
	case svsl_type_array: {
		uint32_t elem = spv_type_for(e, t->elem);
		if (t->array_count == 0) {
			result = svsl_spv_type(&e->spv, SpvOpTypeRuntimeArray, (uint32_t[]){ elem }, 1);
		} else {
			uint32_t len = spv_uint_const(e, (uint32_t)t->array_count);
			result = svsl_spv_type(&e->spv, SpvOpTypeArray, (uint32_t[]){ elem, len }, 2);
		}
		break;
	}
	case svsl_type_struct: {
		const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
		int32_t   count = info->members.count;
		uint32_t *words = svsl_arena_alloc(e->arena, (size_t)(count + 1) * sizeof(uint32_t));
		for (int32_t i = 0; i < count; i++)
			words[1 + i] = spv_type_for(e, info->members.items[i].type);
		uint32_t id2 = words[0] = svsl_spv_id(&e->spv);
		svsl_spv_inst(&e->spv, &e->spv.types, SpvOpTypeStruct, words, (uint32_t)count + 1);
		svsl_spv_inst_str(&e->spv, &e->spv.debug, SpvOpName, (uint32_t[]){ id2 }, 1, info->name);
		result = id2;
		break;
	}
	case svsl_type_sampler:
		result = svsl_spv_type(&e->spv, SpvOpTypeSampler, NULL, 0);
		break;
	case svsl_type_texture:
	case svsl_type_image:
		result = spv_image_type(e, t);
		break;
	case svsl_type_subpass:
	case svsl_type_buffer:
	case svsl_type_tileimage:
		// resource handles have dedicated variable emission (create_globals);
		// a VALUE of one of these types never materializes
		result = svsl_spv_type(&e->spv, SpvOpTypeVoid, NULL, 0);
		break;
	// no default: -Wswitch flags every site when a type kind is added
	}
	if (cacheable) e->type_ids[id] = result;
	return result;
}

static uint32_t spv_ptr_type(emit_t *e, SpvStorageClass class_, uint32_t pointee) {
	return svsl_spv_type(&e->spv, SpvOpTypePointer, (uint32_t[]){ (uint32_t)class_, pointee }, 2);
}

// half types get RelaxedPrecision on their values
static bool type_is_relaxed(const emit_t *e, svsl_type_id_t id) {
	if (id == SVSL_TYPE_NONE) return false;
	const svsl_type_t *t = svsl_type_get(&e->prog->types, id);
	return (t->kind == svsl_type_scalar || t->kind == svsl_type_vector ||
	        t->kind == svsl_type_matrix) && t->scalar == svsl_scalar_half;
}
static void relax(emit_t *e, uint32_t id, svsl_type_id_t type) {
	if (!type_is_relaxed(e, type)) return;
	for (int32_t i = 0; i < e->relaxed_ids.count; i++)
		if (e->relaxed_ids.items[i] == id) return; // aliased values decorate once
	svsl_array_push(e->arena, &e->relaxed_ids, id);
	svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, id, SpvDecorationRelaxedPrecision);
}

// --- laid-out struct types for buffers -------------------------------------------

static uint32_t spv_type_laid    (emit_t *e, svsl_type_id_t type, svsl_layout_ layout);
static uint32_t matrix_stride_for(emit_t *e, const svsl_type_t *m, svsl_layout_ layout);

// one member's Offset (+ RowMajor/MatrixStride for matrices) and MemberName,
// shared by buffer blocks (precomputed offsets) and laid-out struct types
static void decorate_member(emit_t *e, uint32_t struct_id, int32_t index, svsl_type_id_t mtype,
                            uint32_t offset, svsl_str_t name, svsl_layout_ layout) {
	svsl_spv_inst4(&e->spv, &e->spv.decor, SpvOpMemberDecorate, struct_id, (uint32_t)index,
	               SpvDecorationOffset, offset);
	const svsl_type_t *mt = svsl_type_get(&e->prog->types, mtype);
	const svsl_type_t *m  = mt->kind == svsl_type_array ? svsl_type_get(&e->prog->types, mt->elem) : mt;
	if (m->kind == svsl_type_matrix) {
		svsl_spv_inst4(&e->spv, &e->spv.decor, SpvOpMemberDecorate, struct_id, (uint32_t)index,
		               SpvDecorationMatrixStride, matrix_stride_for(e, m, layout));
		svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpMemberDecorate, struct_id, (uint32_t)index,
		               SpvDecorationRowMajor);
	}
	svsl_spv_inst_str(&e->spv, &e->spv.debug, SpvOpMemberName,
	                  (uint32_t[]){ struct_id, (uint32_t)index }, 2, name);
}

// emits a struct type with Offset (+ matrix layout) decorations for a buffer
static uint32_t spv_block_struct(emit_t *e, const svsl_buffer_t *buf) {
	int32_t   count = buf->members.count;
	uint32_t *words = svsl_arena_alloc(e->arena, (size_t)(count + 1) * sizeof(uint32_t));
	uint32_t  id    = words[0] = svsl_spv_id(&e->spv);
	for (int32_t i = 0; i < count; i++) {
		svsl_type_id_t mt = buf->members.items[i].type;
		words[1 + i] = spv_type_laid(e, mt, buf->layout);
		decorate_member(e, id, i, mt, buf->members.items[i].offset,
		                buf->members.items[i].name, buf->layout);
	}
	svsl_spv_inst(&e->spv, &e->spv.types, SpvOpTypeStruct, words, (uint32_t)count + 1);
	svsl_spv_inst_str(&e->spv, &e->spv.debug, SpvOpName, (uint32_t[]){ id }, 1, buf->name);
	svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, id, SpvDecorationBlock);
	return id;
}

// layout-decorated composite type: arrays get ArrayStride, structs get member
// Offsets (+ matrix layout), each unique per (svsl type, layout). Scalars,
// vectors, and matrices carry no type-level layout and share the value types.
static uint32_t matrix_stride_for(emit_t *e, const svsl_type_t *m, svsl_layout_ layout) {
	return svsl_layout_array_stride(&e->prog->types,
		svsl_type_vector_id((svsl_types_t *)&e->prog->types, m->scalar, m->rows), layout);
}

// A bool scalar/vector stores in a buffer as the matching uint type (glslang
// lowering); the svsl type id of that uint shape.
static svsl_type_id_t bool_buffer_uint(emit_t *e, const svsl_type_t *t) {
	svsl_types_t *types = (svsl_types_t *)&e->prog->types;
	return t->kind == svsl_type_vector
		? svsl_type_vector_id(types, svsl_scalar_uint32, t->count)
		: svsl_type_scalar_id(types, svsl_scalar_uint32);
}

static uint32_t spv_type_laid(emit_t *e, svsl_type_id_t type, svsl_layout_ layout) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	// OpTypeBool has no external layout: buffer members store uint 0/1 instead,
	// converting at load/store (matches glslang; see convert_composite)
	if (t->scalar == svsl_scalar_bool &&
	    (t->kind == svsl_type_scalar || t->kind == svsl_type_vector))
		return spv_type_for(e, bool_buffer_uint(e, t));
	if (t->kind != svsl_type_array && t->kind != svsl_type_struct)
		return spv_type_for(e, type);
	bool cacheable = type < e->type_cap;
	if (cacheable && e->laid_ids[layout][type]) return e->laid_ids[layout][type];

	uint32_t result;
	if (t->kind == svsl_type_array) {
		uint32_t elem   = spv_type_laid(e, t->elem, layout);
		uint32_t stride = svsl_layout_array_stride(&e->prog->types, t->elem, layout);
		result = svsl_spv_id(&e->spv);
		if (t->array_count == 0) {
			svsl_spv_inst2(&e->spv, &e->spv.types, SpvOpTypeRuntimeArray, result, elem);
		} else {
			uint32_t len = spv_uint_const(e, (uint32_t)t->array_count);
			svsl_spv_inst3(&e->spv, &e->spv.types, SpvOpTypeArray, result, elem, len);
		}
		svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, result, SpvDecorationArrayStride, stride);
	} else {
		const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
		int32_t   count   = info->members.count;
		uint32_t *offsets = svsl_arena_alloc(e->arena, (size_t)count * sizeof(uint32_t));
		int32_t   bad;
		svsl_layout_members(&e->prog->types, info->members.items, count, layout, offsets, &bad);
		uint32_t *words = svsl_arena_alloc(e->arena, (size_t)(count + 1) * sizeof(uint32_t));
		for (int32_t i = 0; i < count; i++)
			words[1 + i] = spv_type_laid(e, info->members.items[i].type, layout);
		result = words[0] = svsl_spv_id(&e->spv);
		svsl_spv_inst(&e->spv, &e->spv.types, SpvOpTypeStruct, words, (uint32_t)count + 1);
		svsl_spv_inst_str(&e->spv, &e->spv.debug, SpvOpName, (uint32_t[]){ result }, 1, info->name);
		for (int32_t i = 0; i < count; i++)
			decorate_member(e, result, i, info->members.items[i].type, offsets[i],
			                info->members.items[i].name, layout);
	}
	if (cacheable) e->laid_ids[layout][type] = result;
	return result;
}

// --- globals ---------------------------------------------------------------------

static uint32_t binding_value(const svsl_binding_t *bind) {
	if (bind->direct) return (uint32_t)bind->slot;
	switch (bind->cls) {
	case 't': case 's': return (uint32_t)bind->slot + 100;
	case 'u':           return (uint32_t)bind->slot + 200;
	default:            return (uint32_t)bind->slot; // 'b'
	}
}

// 8/16-bit scalars inside buffer blocks need storage-class-specific access
// capabilities (16-bit is core in SPIR-V 1.3; 8-bit still needs its extension)
static void small_scalar_caps(emit_t *e, svsl_type_id_t type, SpvStorageClass class_) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	switch (t->kind) {
	case svsl_type_array:
		small_scalar_caps(e, t->elem, class_);
		return;
	case svsl_type_struct: {
		const svsl_struct_info_t *info = &e->prog->types.structs.items[t->struct_index];
		for (int32_t i = 0; i < info->members.count; i++)
			small_scalar_caps(e, info->members.items[i].type, class_);
		return;
	}
	case svsl_type_scalar:
	case svsl_type_vector:
	case svsl_type_matrix: {
		int32_t size = svsl_scalar_size(t->scalar);
		if (t->scalar == svsl_scalar_bool || size > 2) return;
		if (size == 2) {
			svsl_spv_cap(&e->spv,
				class_ == SpvStorageClassUniform      ? SpvCapabilityUniformAndStorageBuffer16BitAccess :
				class_ == SpvStorageClassPushConstant ? SpvCapabilityStoragePushConstant16 :
				                                        SpvCapabilityStorageBuffer16BitAccess);
		} else {
			svsl_spv_extension(&e->spv, "SPV_KHR_8bit_storage");
			svsl_spv_cap(&e->spv,
				class_ == SpvStorageClassUniform      ? SpvCapabilityUniformAndStorageBuffer8BitAccess :
				class_ == SpvStorageClassPushConstant ? SpvCapabilityStoragePushConstant8 :
				                                        SpvCapabilityStorageBuffer8BitAccess);
		}
		return;
	}
	case svsl_type_void:
	case svsl_type_texture:
	case svsl_type_image:
	case svsl_type_sampler:
	case svsl_type_subpass:
	case svsl_type_buffer:
	case svsl_type_tileimage:
		return; // opaque handles hold no small scalars
	}
}

// common tail for a resource variable: pointer type, the OpVariable, its name,
// and the standard Binding + DescriptorSet decorations. Callers add any
// resource-specific decorations (NonWritable, InputAttachmentIndex) afterward.
static uint32_t emit_resource_var(emit_t *e, SpvStorageClass class_, uint32_t type,
                                  const svsl_binding_t *bind, svsl_str_t name) {
	svsl_spv_t *spv = &e->spv;
	uint32_t    ptr = spv_ptr_type(e, class_, type);
	uint32_t    var = svsl_spv_id(spv);
	svsl_spv_inst3(spv, &spv->types, SpvOpVariable, ptr, var, (uint32_t)class_);
	if (name.len) svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, name);
	svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationBinding, binding_value(bind));
	svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationDescriptorSet, (uint32_t)bind->space);
	return var;
}

// numeric constant folding for initializers: literals, +-*/, negation, and
// references to other constant globals
static bool const_eval_num(emit_t *e, const svsl_ast_expr_t *v, double *out) {
	switch (v->kind) {
	case svsl_expr_float_lit: *out = v->float_lit.value; return true;
	case svsl_expr_int_lit:   *out = (double)(int64_t)v->int_lit.value; return true;
	case svsl_expr_bool_lit:  *out = v->bool_lit ? 1 : 0; return true;
	case svsl_expr_unary:
		if (v->unary.op != svsl_tok_minus && v->unary.op != svsl_tok_plus) return false;
		if (!const_eval_num(e, v->unary.operand, out)) return false;
		if (v->unary.op == svsl_tok_minus) *out = -*out;
		return true;
	case svsl_expr_ident: {
		// global initializers aren't body-checked, so resolve by name
		for (int32_t i = 0; i < e->prog->const_globals.count; i++) {
			const svsl_global_t *g = &e->prog->const_globals.items[i];
			if (svsl_str_eq(g->name, v->ident) && g->var && g->var->init)
				return const_eval_num(e, g->var->init, out);
		}
		return false;
	}
	case svsl_expr_binary: {
		double a, b;
		if (!const_eval_num(e, v->binary.lhs, &a) || !const_eval_num(e, v->binary.rhs, &b)) return false;
		switch (v->binary.op) {
		case svsl_tok_plus:  *out = a + b; return true;
		case svsl_tok_minus: *out = a - b; return true;
		case svsl_tok_star:  *out = a * b; return true;
		case svsl_tok_slash: if (b == 0) return false; *out = a / b; return true;
		default: return false;
		}
	}
	case svsl_expr_cast:
		return const_eval_num(e, v->cast.operand, out);
	default:
		return false;
	}
}

// componentwise fold for vector-typed constant expressions: constructors and
// init lists fill, scalars splat, and +-*/ combine (1.0 / float3(...) et al.)
static bool const_eval_vec(emit_t *e, const svsl_ast_expr_t *v, int32_t n, double *out) {
	switch (v->kind) {
	case svsl_expr_init_list:
	case svsl_expr_ctor: {
		const svsl_ast_expr_t **items = v->kind == svsl_expr_init_list
			? (const svsl_ast_expr_t **)v->init_list.items
			: (const svsl_ast_expr_t **)v->ctor.args;
		int32_t count = v->kind == svsl_expr_init_list ? v->init_list.count : v->ctor.arg_count;
		if (count == 1) {
			double s;
			if (!const_eval_num(e, items[0], &s)) return false;
			for (int32_t i = 0; i < n; i++) out[i] = s;
			return true;
		}
		if (count != n) return false;
		for (int32_t i = 0; i < n; i++)
			if (!const_eval_num(e, items[i], &out[i])) return false;
		return true;
	}
	case svsl_expr_binary: {
		double a[4], b[4];
		if (!const_eval_vec(e, v->binary.lhs, n, a) ||
		    !const_eval_vec(e, v->binary.rhs, n, b)) return false;
		for (int32_t i = 0; i < n; i++) {
			switch (v->binary.op) {
			case svsl_tok_plus:  out[i] = a[i] + b[i]; break;
			case svsl_tok_minus: out[i] = a[i] - b[i]; break;
			case svsl_tok_star:  out[i] = a[i] * b[i]; break;
			case svsl_tok_slash: if (b[i] == 0) return false; out[i] = a[i] / b[i]; break;
			default: return false;
			}
		}
		return true;
	}
	case svsl_expr_ident: // another const global of vector type
		for (int32_t i = 0; i < e->prog->const_globals.count; i++) {
			const svsl_global_t *g = &e->prog->const_globals.items[i];
			if (svsl_str_eq(g->name, v->ident) && g->var && g->var->init)
				return const_eval_vec(e, g->var->init, n, out);
		}
		return false;
	default: { // scalar expression splats across the vector
		double s;
		if (!const_eval_num(e, v, &s)) return false;
		for (int32_t i = 0; i < n; i++) out[i] = s;
		return true;
	}
	}
}

static uint32_t spv_const_scalar(emit_t *e, svsl_scalar_ scalar, double f) {
	svsl_spv_t *spv = &e->spv;
	uint32_t    st  = spv_scalar_type(e, scalar);
	int64_t     i   = (int64_t)f;
	if (scalar == svsl_scalar_bool)
		return svsl_spv_const(spv, st, (uint64_t)(i != 0), false, true);
	if (scalar == svsl_scalar_float32 || scalar == svsl_scalar_half ||
	    scalar == svsl_scalar_float16) {
		float fv = (float)f;
		uint32_t bits;
		memcpy(&bits, &fv, 4);
		return svsl_spv_const(spv, st, bits, false, false);
	}
	if (scalar == svsl_scalar_float64) {
		uint64_t bits;
		memcpy(&bits, &f, 8);
		return svsl_spv_const(spv, st, bits, true, false);
	}
	return svsl_spv_const(spv, st, (uint64_t)i, svsl_scalar_size(scalar) == 8, false);
}

// constant value from an AST initializer (const-global private variables)
static uint32_t spv_const_expr(emit_t *e, const svsl_ast_expr_t *expr, svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	svsl_spv_t        *spv = &e->spv;

	if (t->kind == svsl_type_scalar) {
		double f;
		if (!const_eval_num(e, expr, &f)) return 0;
		return spv_const_scalar(e, t->scalar, f);
	}
	if (t->kind == svsl_type_vector) { // componentwise, including constant math
		double vals[4];
		if (!const_eval_vec(e, expr, t->count, vals)) return 0;
		uint32_t parts[4];
		for (int32_t i = 0; i < t->count; i++)
			if (!(parts[i] = spv_const_scalar(e, t->scalar, vals[i]))) return 0;
		uint32_t id = svsl_spv_id(spv);
		uint32_t words[6] = { spv_type_for(e, type), id, parts[0], parts[1], parts[2], parts[3] };
		svsl_spv_inst(spv, &spv->types, SpvOpConstantComposite, words, 2 + (uint32_t)t->count);
		return id;
	}

	// composites from init lists / ctor forms
	const svsl_ast_expr_t **items = NULL;
	int32_t                 count = 0;
	if (expr->kind == svsl_expr_init_list) { items = (const svsl_ast_expr_t **)expr->init_list.items; count = expr->init_list.count; }
	else if (expr->kind == svsl_expr_ctor) { items = (const svsl_ast_expr_t **)expr->ctor.args; count = expr->ctor.arg_count; }
	else return 0;

	uint32_t *parts = svsl_arena_alloc(e->arena, (size_t)count * sizeof(uint32_t));
	int32_t  want = 0;
	if (t->kind == svsl_type_array && count == t->array_count) {
		want = count;
		for (int32_t i = 0; i < want; i++)
			if (!(parts[i] = spv_const_expr(e, items[i], t->elem))) return 0;
	} else if (t->kind == svsl_type_matrix && count == t->rows * t->cols) {
		want = t->rows; // rows of vec(cols)
		svsl_type_id_t st = svsl_type_scalar_id((svsl_types_t *)&e->prog->types, t->scalar);
		uint32_t rt = svsl_spv_type(&e->spv, SpvOpTypeVector,
		                            (uint32_t[]){ spv_scalar_type(e, t->scalar), t->cols }, 2);
		for (int32_t r = 0; r < t->rows; r++) {
			uint32_t comps[4];
			for (int32_t c = 0; c < t->cols; c++)
				if (!(comps[c] = spv_const_expr(e, items[r * t->cols + c], st))) return 0;
			uint32_t id = svsl_spv_id(&e->spv);
			uint32_t words[7] = { rt, id };
			for (int32_t c = 0; c < t->cols; c++) words[2 + c] = comps[c];
			svsl_spv_inst(&e->spv, &e->spv.types, SpvOpConstantComposite, words, 2 + (uint32_t)t->cols);
			parts[r] = id;
		}
	} else return 0;

	uint32_t id = svsl_spv_id(spv);
	uint32_t *words = svsl_arena_alloc(e->arena, (size_t)(want + 2) * sizeof(uint32_t));
	words[0] = spv_type_for(e, type);
	words[1] = id;
	for (int32_t i = 0; i < want; i++) words[2 + i] = parts[i];
	svsl_spv_inst(spv, &spv->types, SpvOpConstantComposite, words, 2 + (uint32_t)want);
	return id;
}

static void create_globals(emit_t *e) {
	const svsl_program_t *prog = e->prog;
	svsl_spv_t           *spv  = &e->spv;

	// Emit only the globals this stage actually references. A fragment shader
	// that returns its color input should not declare the system cbuffer, the
	// instance buffer, etc. — glslang+spirv-opt strip these, and emitting them
	// is pure bloat (matches the used-only reflection the container already does).
	uint8_t *buf_used = svsl_arena_alloc(e->arena, (size_t)(prog->buffers.count   > 0 ? prog->buffers.count   : 1));
	uint8_t *res_used = svsl_arena_alloc(e->arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1));
	svsl_ir_func_globals(prog, e->fn, buf_used, res_used);

	// buffer blocks (uniform, storage, push constants)
	for (int32_t i = 0; i < prog->buffers.count; i++) {
		if (!buf_used[i]) continue;
		const svsl_buffer_t *buf = &prog->buffers.items[i];
		SpvStorageClass class_ =
			buf->kind == svsl_block_pushconstant ? SpvStorageClassPushConstant :
			buf->kind == svsl_block_storagebuffer ? SpvStorageClassStorageBuffer :
			SpvStorageClassUniform;
		uint32_t struct_id = spv_block_struct(e, buf);
		for (int32_t m = 0; m < buf->members.count; m++)
			small_scalar_caps(e, buf->members.items[m].type, class_);
		uint32_t ptr = spv_ptr_type(e, class_, struct_id);
		uint32_t var = svsl_spv_id(spv);
		svsl_spv_inst3(spv, &spv->types, SpvOpVariable, ptr, var, (uint32_t)class_);
		svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, buf->name);
		e->buffer_vars[i] = var;

		if (buf->kind == svsl_block_uniform) {
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationBinding,
			               binding_value(&buf->bind));
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationDescriptorSet,
			               (uint32_t)buf->bind.space);
		} else if (buf->kind == svsl_block_storagebuffer) {
			// binding comes from the linked resource entry
			for (int32_t r = 0; r < prog->resources.count; r++) {
				if (prog->resources.items[r].buffer_index != i) continue;
				const svsl_resource_t *res = &prog->resources.items[r];
				svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationBinding,
				               binding_value(&res->bind));
				svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationDescriptorSet,
				               (uint32_t)res->bind.space);
				if (res->kind == svsl_res_structured)
					svsl_spv_inst2(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationNonWritable);
				e->resource_vars[r] = var;
			}
		}
	}

	// resources (textures, samplers, object-form buffers, images, subpass inputs)
	for (int32_t i = 0; i < prog->resources.count; i++) {
		if (!res_used[i]) continue;           // stage never touches it
		const svsl_resource_t *res = &prog->resources.items[i];
		if (res->buffer_index >= 0) continue; // block form handled above
		const svsl_type_t *t = svsl_type_get(&prog->types, res->type);

		if (res->kind == svsl_res_texture) {
			uint32_t img  = spv_image_type(e, t);
			e->resource_img_type[i] = img;
			uint32_t type = res->sampler_slot >= 0
			              ? svsl_spv_type(spv, SpvOpTypeSampledImage, (uint32_t[]){ img }, 1)
			              : img;
			e->resource_vars[i] = emit_resource_var(e, SpvStorageClassUniformConstant, type,
			                                        &res->bind, res->name);
			continue;
		}
		if (res->kind == svsl_res_sampler) {
			// paired samplers fuse into their texture and emit nothing
			bool paired = false;
			for (int32_t k = 0; k < prog->resources.count; k++)
				if (prog->resources.items[k].kind == svsl_res_texture &&
				    prog->resources.items[k].sampler_slot == res->bind.slot &&
				    prog->resources.items[k].bind.space == res->bind.space) paired = true;
			if (paired) continue;
			uint32_t type = svsl_spv_type(spv, SpvOpTypeSampler, NULL, 0);
			e->resource_vars[i] = emit_resource_var(e, SpvStorageClassUniformConstant, type,
			                                        &res->bind, res->name);
			continue;
		}
		if (res->kind == svsl_res_structured || res->kind == svsl_res_rw_structured) {
			// object form: struct { T @data[]; } Block, elements in the declared layout
			small_scalar_caps(e, t->elem, SpvStorageClassStorageBuffer);
			uint32_t elem = spv_type_laid(e, t->elem, (svsl_layout_)res->layout);
			uint32_t rt   = svsl_spv_id(spv);
			svsl_spv_inst2(spv, &spv->types, SpvOpTypeRuntimeArray, rt, elem);
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, rt, SpvDecorationArrayStride, res->element_size);
			uint32_t wrap = svsl_spv_id(spv);
			svsl_spv_inst2(spv, &spv->types, SpvOpTypeStruct, wrap, rt);
			svsl_spv_inst4(spv, &spv->decor, SpvOpMemberDecorate, wrap, 0, SpvDecorationOffset, 0);
			svsl_spv_inst2(spv, &spv->decor, SpvOpDecorate, wrap, SpvDecorationBlock);
			svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ wrap }, 1, res->name);
			uint32_t var = emit_resource_var(e, SpvStorageClassStorageBuffer, wrap, &res->bind, res->name);
			if (res->kind == svsl_res_structured)
				svsl_spv_inst2(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationNonWritable);
			e->resource_vars[i] = var;
			continue;
		}
		if (res->kind == svsl_res_image) {
			uint32_t img = spv_image_type(e, t);
			e->resource_img_type[i] = img;
			e->resource_vars[i] = emit_resource_var(e, SpvStorageClassUniformConstant, img,
			                                        &res->bind, res->name);
			continue;
		}
		if (res->kind == svsl_res_subpass) {
			svsl_spv_cap(spv, SpvCapabilityInputAttachment);
			const svsl_type_t *elem = svsl_type_get(&prog->types, t->elem);
			uint32_t scalar = spv_scalar_type(e, elem->scalar);
			uint32_t img = svsl_spv_type(spv, SpvOpTypeImage,
				(uint32_t[]){ scalar, SpvDimSubpassData, 0, 0, t->multisampled ? 1u : 0u,
				              2, SpvImageFormatUnknown }, 7);
			e->resource_img_type[i] = img;
			uint32_t var = emit_resource_var(e, SpvStorageClassUniformConstant, img, &res->bind, res->name);
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationInputAttachmentIndex,
			               res->subpass_index >= 0 ? (uint32_t)res->subpass_index : 0);
			e->resource_vars[i] = var;
			continue;
		}
		if (res->kind == svsl_res_tileimage) {
			// tile memory, not a descriptor: TileImageEXT storage class with a
			// Location naming the color attachment, no binding or set
			svsl_spv_cap(spv, SpvCapabilityTileImageColorReadAccessEXT);
			svsl_spv_extension(spv, "SPV_EXT_shader_tile_image");
			const svsl_type_t *elem = svsl_type_get(&prog->types, t->elem);
			uint32_t scalar = spv_scalar_type(e, elem->scalar);
			uint32_t img = svsl_spv_type(spv, SpvOpTypeImage,
				(uint32_t[]){ scalar, SpvDimTileImageDataEXT, 0, 0, t->multisampled ? 1u : 0u,
				              2, SpvImageFormatUnknown }, 7);
			e->resource_img_type[i] = img;
			uint32_t ptr = spv_ptr_type(e, SpvStorageClassTileImageEXT, img);
			uint32_t var = svsl_spv_id(spv);
			svsl_spv_inst3(spv, &spv->types, SpvOpVariable, ptr, var, SpvStorageClassTileImageEXT);
			svsl_spv_inst3(spv, &spv->decor, SpvOpDecorate, var, SpvDecorationLocation,
			               res->subpass_index >= 0 ? (uint32_t)res->subpass_index : 0);
			svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, res->name);
			e->resource_vars[i] = var;
			continue;
		}
	}

	// workgroup variables
	for (int32_t i = 0; i < prog->workgroup_vars.count; i++) {
		const svsl_global_t *g = &prog->workgroup_vars.items[i];
		uint32_t ptr = spv_ptr_type(e, SpvStorageClassWorkgroup, spv_type_for(e, g->type));
		uint32_t var = svsl_spv_id(spv);
		svsl_spv_inst3(spv, &spv->types, SpvOpVariable, ptr, var, SpvStorageClassWorkgroup);
		svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, g->name);
		e->workgroup_ids[i] = var;
	}

	// const globals with constant initializers → Private variables
	for (int32_t i = 0; i < prog->const_globals.count; i++) {
		const svsl_global_t *g = &prog->const_globals.items[i];
		if (g->type == SVSL_TYPE_NONE || !g->var || !g->var->init) continue;
		uint32_t init = spv_const_expr(e, g->var->init, g->type);
		if (!init) continue; // only referenced consts matter; error surfaces on use
		uint32_t ptr = spv_ptr_type(e, SpvStorageClassPrivate, spv_type_for(e, g->type));
		uint32_t var = svsl_spv_id(spv);
		svsl_spv_inst4(spv, &spv->types, SpvOpVariable, ptr, var, SpvStorageClassPrivate, init);
		svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, g->name);
		e->const_global_ids[i] = var;
	}
}

// --- stage IO ---------------------------------------------------------------------

typedef struct io_var_t {
	uint32_t       var;
	svsl_type_id_t type;
	bool           builtin;
} io_var_t;

static uint32_t make_io_var(emit_t *e, svsl_type_id_t type, bool output, svsl_str_t name) {
	uint32_t tid = spv_type_for(e, type);
	uint32_t ptr = spv_ptr_type(e, output ? SpvStorageClassOutput : SpvStorageClassInput, tid);
	uint32_t var = svsl_spv_id(&e->spv);
	svsl_spv_inst3(&e->spv, &e->spv.types, SpvOpVariable, ptr, var,
	               output ? SpvStorageClassOutput : SpvStorageClassInput);
	if (name.len) svsl_spv_inst_str(&e->spv, &e->spv.debug, SpvOpName, (uint32_t[]){ var }, 1, name);
	svsl_array_push(e->arena, &e->interface, var);
	relax(e, var, type);
	return var;
}

// How many interface locations a stage-IO member of this type consumes. A
// scalar/vector (≤4 32-bit components) is one; a matrix is one per column vector
// (its emitted OpTypeMatrix has `rows` of them); an array multiplies by its
// length; a nested struct sums its members. One-per-member would overlap an
// array/matrix with the following member's location.
static int32_t location_span(const svsl_types_t *types, svsl_type_id_t id) {
	const svsl_type_t *t = svsl_type_get(types, id);
	switch (t->kind) {
	case svsl_type_matrix:
		return t->rows;
	case svsl_type_array:
		return (t->array_count > 0 ? t->array_count : 1) * location_span(types, t->elem);
	case svsl_type_struct: {
		const svsl_struct_info_t *si = &types->structs.items[t->struct_index];
		int32_t sum = 0;
		for (int32_t m = 0; m < si->members.count; m++)
			sum += location_span(types, si->members.items[m].type);
		return sum;
	}
	default:
		return 1; // scalar / vector
	}
}

static void io_decorate(emit_t *e, uint32_t var, svsl_str_t semantic, svsl_sem_io_ io,
                        int32_t *ref_location, svsl_type_id_t type, uint8_t interp) {
	const svsl_type_t *vt = svsl_type_get(&e->prog->types, type);
	bool integer_type = (vt->kind == svsl_type_scalar || vt->kind == svsl_type_vector) &&
	                    (vt->scalar >= svsl_scalar_int8 && vt->scalar <= svsl_scalar_uint64);

	svsl_semantic_info_t info;
	if (svsl_semantic_lookup(semantic, io, &info) && info.is_builtin) {
		svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationBuiltIn, info.builtin);
		if (info.builtin == SpvBuiltInViewIndex) svsl_spv_cap(&e->spv, SpvCapabilityMultiView);
		if (info.builtin == SpvBuiltInFragDepth) {
			e->needs_depth_replacing = true;
			e->depth_mode            = info.depth_mode;
		}
		if (io == svsl_sem_ps_in && integer_type)
			svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationFlat);
		if (interp & svsl_interp_invariant) // invariant SV_Position (depth prepass)
			svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationInvariant);
		return;
	}
	int32_t location = *ref_location;
	if (io == svsl_sem_ps_out) { // SV_TargetN picks its own location
		svsl_semantic_info_t ti;
		if (svsl_semantic_lookup(semantic, io, &ti)) location = ti.target_index;
	}
	svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationLocation, (uint32_t)location);
	*ref_location += location_span(&e->prog->types, type);

	// Vulkan requires integer fragment inputs to be flat
	if (io == svsl_sem_ps_in && (integer_type || (interp & svsl_interp_flat)))
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationFlat);
	else if (io == svsl_sem_ps_in && (interp & svsl_interp_noperspective))
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationNoPerspective);
	if (io == svsl_sem_ps_in && (interp & svsl_interp_centroid))
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationCentroid);
	if (interp & svsl_interp_invariant)
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationInvariant);
}

// Records the location assigned to the next prog->vertex_inputs entry (-1 = its
// OpVariable was stripped). The prologue's walk must mirror collect_vertex_inputs
// (sema.c) — same order, same generated-semantic exclusions — so each entry is
// name-checked and a divergence fails the compile instead of writing metadata
// that lies about the SPIR-V interface.
static void record_vs_input(emit_t *e, int32_t *ref_index, svsl_str_t name,
                            svsl_loc_t loc, int32_t location) {
	if (!e->vs_input_locations) return;
	int32_t i = (*ref_index)++;
	if (i >= e->prog->vertex_inputs.count ||
	    !svsl_str_eq(e->prog->vertex_inputs.items[i].name, name)) {
		eerr(e, loc, "vertex input metadata diverged from the emitted interface");
		return;
	}
	if (location > 255) {
		eerr(e, loc, "vertex input location exceeds the container's limit of 255");
		return;
	}
	e->vs_input_locations[i] = location;
}

// --- function body -----------------------------------------------------------------

static void begin_block(emit_t *e, uint32_t label) {
	svsl_spv_inst1(&e->spv, &e->spv.funcs, SpvOpLabel, label);
	e->current_block = label;
	e->terminated    = false;
}
static void branch_to(emit_t *e, uint32_t target) {
	if (e->terminated) return;
	svsl_spv_inst1(&e->spv, &e->spv.funcs, SpvOpBranch, target);
	e->terminated = true;
}
// values may appear after a terminator (unreachable but valid): open a fresh block
static void ensure_block(emit_t *e) {
	if (e->terminated) begin_block(e, svsl_spv_id(&e->spv));
}

static struct cf_frame *cf_innermost_loop(emit_t *e) {
	for (int32_t i = e->cf_depth - 1; i >= 0; i--)
		if (e->cf[i].kind == 'l') return &e->cf[i];
	return NULL;
}
static struct cf_frame *cf_innermost_breakable(emit_t *e) {
	for (int32_t i = e->cf_depth - 1; i >= 0; i--)
		if (e->cf[i].kind == 'l' || e->cf[i].kind == 's') return &e->cf[i];
	return NULL;
}

static bool scalar_is_floaty(svsl_scalar_ s) {
	return s == svsl_scalar_half || s == svsl_scalar_float16 ||
	       s == svsl_scalar_float32 || s == svsl_scalar_float64;
}
static bool scalar_is_signed(svsl_scalar_ s) {
	return s == svsl_scalar_int8 || s == svsl_scalar_int16 ||
	       s == svsl_scalar_int32 || s == svsl_scalar_int64;
}

// component scalar of a value's type
static svsl_scalar_ value_scalar(const emit_t *e, uint32_t ir_id) {
	return svsl_type_get(&e->prog->types, e->fn->insts.items[ir_id].type)->scalar;
}

static uint32_t emit_value_inst(emit_t *e, svsl_spv_stream_t *fs, SpvOp op, uint32_t type_id,
                                const uint32_t *operands, uint32_t count) {
	uint32_t  id       = svsl_spv_id(&e->spv);
	uint32_t  stack[18];
	uint32_t *words = count + 2 <= 18 ? stack
	                : svsl_arena_alloc(e->arena, (size_t)(count + 2) * 4);
	words[0] = type_id;
	words[1] = id;
	for (uint32_t i = 0; i < count; i++) words[2 + i] = operands[i];
	svsl_spv_inst(&e->spv, fs, op, words, count + 2);
	return id;
}

// standalone sampler variable, created on demand (cross-paired sampling: a
// sampler fused into one texture used to sample another)
static uint32_t sampler_var_for(emit_t *e, uint32_t sampler_res) {
	if (e->sampler_vars[sampler_res]) return e->sampler_vars[sampler_res];
	const svsl_resource_t *smp = &e->prog->resources.items[sampler_res];
	uint32_t type = svsl_spv_type(&e->spv, SpvOpTypeSampler, NULL, 0);
	uint32_t var  = emit_resource_var(e, SpvStorageClassUniformConstant, type, &smp->bind, smp->name);
	e->sampler_vars[sampler_res] = var;
	return var;
}

// sampled-image value for a texture resource (+ the sampler used)
static uint32_t load_sampled_image(emit_t *e, int32_t tex, uint32_t sampler_res) {
	const svsl_resource_t *res = &e->prog->resources.items[tex];
	svsl_spv_stream_t     *fs  = &e->spv.funcs;
	uint32_t img_type = e->resource_img_type[tex];
	uint32_t si_type  = svsl_spv_type(&e->spv, SpvOpTypeSampledImage, (uint32_t[]){ img_type }, 1);

	if (res->sampler_slot >= 0) { // combined variable
		uint32_t combined = emit_value_inst(e, fs, SpvOpLoad, si_type,
		                                    (uint32_t[]){ e->resource_vars[tex] }, 1);
		if (sampler_res == SVSL_IR_NONE) return combined;
		const svsl_resource_t *smp = &e->prog->resources.items[sampler_res];
		if (smp->bind.slot == res->sampler_slot && smp->bind.space == res->bind.space)
			return combined;
		// different sampler: re-pair through a standalone sampler variable
		uint32_t image   = emit_value_inst(e, fs, SpvOpImage, img_type, (uint32_t[]){ combined }, 1);
		uint32_t svar    = e->resource_vars[sampler_res] ? e->resource_vars[sampler_res]
		                                                 : sampler_var_for(e, sampler_res);
		uint32_t sampler = emit_value_inst(e, fs, SpvOpLoad,
		                                   svsl_spv_type(&e->spv, SpvOpTypeSampler, NULL, 0),
		                                   (uint32_t[]){ svar }, 1);
		return emit_value_inst(e, fs, SpvOpSampledImage, si_type, (uint32_t[]){ image, sampler }, 2);
	}
	uint32_t image = emit_value_inst(e, fs, SpvOpLoad, img_type,
	                                 (uint32_t[]){ e->resource_vars[tex] }, 1);
	if (sampler_res == SVSL_IR_NONE) return image;
	uint32_t svar    = e->resource_vars[sampler_res] ? e->resource_vars[sampler_res]
	                                                 : sampler_var_for(e, sampler_res);
	uint32_t sampler = emit_value_inst(e, fs, SpvOpLoad,
	                                   svsl_spv_type(&e->spv, SpvOpTypeSampler, NULL, 0),
	                                   (uint32_t[]){ svar }, 1);
	return emit_value_inst(e, fs, SpvOpSampledImage, si_type, (uint32_t[]){ image, sampler }, 2);
}

// input variable for a subgroup builtin (SubgroupSize, SubgroupLocalInvocationId,
// SubgroupId, NumSubgroups), created on demand and added to the interface
static uint32_t builtin_input_var(emit_t *e, uint32_t index) {
	index &= 7;
	if (e->builtin_input[index]) return e->builtin_input[index];
	static const SpvBuiltIn builtins[] = {
		SpvBuiltInSubgroupSize, SpvBuiltInSubgroupLocalInvocationId,
		SpvBuiltInSubgroupId, SpvBuiltInNumSubgroups };
	svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniform);
	uint32_t u32 = spv_scalar_type(e, svsl_scalar_uint32);
	uint32_t ptr = spv_ptr_type(e, SpvStorageClassInput, u32);
	uint32_t var = svsl_spv_id(&e->spv);
	svsl_spv_inst3(&e->spv, &e->spv.types, SpvOpVariable, ptr, var, SpvStorageClassInput);
	svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationBuiltIn,
	               builtins[index < 4 ? index : 0]);
	svsl_array_push(e->arena, &e->interface, var);
	e->builtin_input[index] = var;
	return var;
}

// Converts a raw vec4 sample result to the declared element type. Sampled
// results are always 32-bit, so a float16 destination narrows with an
// FConvert after the component-count shuffle.
static uint32_t shrink_to(emit_t *e, uint32_t value, uint32_t have_comps, svsl_type_id_t want) {
	const svsl_type_t *t       = svsl_type_get(&e->prog->types, want);
	bool               convert = t->scalar == svsl_scalar_float16;
	svsl_types_t      *types   = (svsl_types_t *)&e->prog->types;
	svsl_type_id_t     wide    = !convert ? want :
		t->kind == svsl_type_vector ? svsl_type_vector_id(types, svsl_scalar_float32, t->count)
		                            : svsl_type_scalar_id(types, svsl_scalar_float32);
	uint32_t tid = spv_type_for(e, wide);
	if (t->kind == svsl_type_scalar) {
		value = emit_value_inst(e, &e->spv.funcs, SpvOpCompositeExtract, tid, (uint32_t[]){ value, 0 }, 2);
	} else if (t->kind == svsl_type_vector && t->count < (int32_t)have_comps) {
		uint32_t ops[8] = { value, value };
		for (int32_t i = 0; i < t->count; i++) ops[2 + i] = (uint32_t)i;
		value = emit_value_inst(e, &e->spv.funcs, SpvOpVectorShuffle, tid, ops, 2 + (uint32_t)t->count);
	}
	if (convert)
		value = emit_value_inst(e, &e->spv.funcs, SpvOpFConvert, spv_type_for(e, want),
		                        (uint32_t[]){ value }, 1);
	return value;
}

// --- entry-input scalar replacement (SROA) ----------------------------------------
// A read-only struct stage parameter accessed only through constant member chains
// need not be reassembled into a Function shadow variable: each member chain can
// resolve straight to that member's stage-input variable, and only referenced
// members are declared. Matches skshaderc, which loads the used member directly.

#define SVSL_EMIT_MAX_MEMBERS 32

// struct-typed entry-parameter op? returns param index or -1
static int32_t sroa_param_index(const emit_t *e, uint32_t id) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	if (in->op != svsl_ir_param) return -1;
	const svsl_type_t *t = svsl_type_get(&e->prog->types, in->type);
	return t->kind == svsl_type_struct ? (int32_t)in->args[0] : -1;
}

// member chain off a struct param, with a constant first (member) index? fills
// param + member. Accepts trailing sub-indices (e.g. input.world_pos.y): those
// become an access chain into the member's stage-input variable at emit.
static bool sroa_member_chain(const emit_t *e, uint32_t id, int32_t *out_param, int32_t *out_member) {
	const svsl_ir_inst_t *in = &e->fn->insts.items[id];
	if (in->op != svsl_ir_chain || in->aux_count < 1) return false;
	int32_t p = sroa_param_index(e, in->args[0]);
	if (p < 0) return false;
	const svsl_ir_inst_t *idx = &e->fn->insts.items[e->fn->aux.items[in->aux]];
	if (idx->op != svsl_ir_const) return false;
	*out_param  = p;
	*out_member = (int32_t)idx->args[0];
	return true;
}

static bool sroa_sem_is_builtin(svsl_str_t semantic, svsl_sem_io_ io) {
	svsl_semantic_info_t info;
	return svsl_semantic_lookup(semantic, io, &info) && info.is_builtin;
}

// Marks each struct param SROA-able unless it is used in any way other than as
// the base of a constant member chain that is only loaded. Also records which
// members are referenced (param_member_var used as a 0/1 marker until the
// prologue replaces it with the real input variable id).
static void analyze_param_sroa(emit_t *e) {
	const svsl_ir_func_t   *fn   = e->fn;
	int32_t                 pc   = fn->entry->func->param_count;
	const svsl_func_info_t *info = svsl_program_func_info(e->prog, fn->entry->func);

	for (int32_t p = 0; p < pc; p++) {
		bool ok = false;
		if (info) {
			const svsl_type_t *pt = svsl_type_get(&e->prog->types, info->param_types[p]);
			if (pt->kind == svsl_type_struct)
				ok = e->prog->types.structs.items[pt->struct_index].members.count <= SVSL_EMIT_MAX_MEMBERS;
		}
		e->param_sroa[p] = ok ? 1 : 0;
	}

	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		uint32_t              mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++) {
			if (!(mask & (1u << a)) || inst->args[a] >= (uint32_t)fn->insts.count) continue;
			uint32_t o = inst->args[a];
			int32_t  cp, cm;
			int32_t  pidx = sroa_param_index(e, o);
			if (pidx >= 0) { // direct use of the param op — only a member-chain base is ok
				if (!((svsl_ir_op_)inst->op == svsl_ir_chain && a == 0 &&
				      sroa_member_chain(e, (uint32_t)i, &cp, &cm)))
					e->param_sroa[pidx] = 0;
			}
			if (sroa_member_chain(e, o, &cp, &cm)) { // use of a member chain — only load's pointer
				if ((svsl_ir_op_)inst->op == svsl_ir_load && a == 0)
					e->param_member_var[cp * SVSL_EMIT_MAX_MEMBERS + cm] = 1; // referenced
				else
					e->param_sroa[cp] = 0;
			}
		}
		if (svsl_ir_aux_holds_values(inst)) // any param/member-chain in aux is a disallowed use
			for (uint32_t k = 0; k < inst->aux_count; k++) {
				uint32_t o = fn->aux.items[inst->aux + k];
				if (o >= (uint32_t)fn->insts.count) continue;
				int32_t pidx = sroa_param_index(e, o), cp, cm;
				if (pidx >= 0)                            e->param_sroa[pidx] = 0;
				if (sroa_member_chain(e, o, &cp, &cm))    e->param_sroa[cp]   = 0;
			}
	}
}

// constant single-index member chain off a specific base var? fills member
static bool sroa_out_member_chain(const svsl_ir_func_t *fn, uint32_t id, uint32_t base, int32_t *out_member) {
	if (id >= (uint32_t)fn->insts.count) return false;
	const svsl_ir_inst_t *in = &fn->insts.items[id];
	if (in->op != svsl_ir_chain || in->args[0] != base || in->aux_count != 1) return false;
	const svsl_ir_inst_t *idx = &fn->insts.items[fn->aux.items[in->aux]];
	if (idx->op != svsl_ir_const) return false;
	*out_member = (int32_t)idx->args[0];
	return true;
}

// Finds a return-struct local var that can be scalar-replaced: the var is loaded
// whole into every return, written only through constant member chains, and never
// otherwise referenced (no whole-struct store, no member read-back, no aliasing
// into aux). Such a var is pure output plumbing — its members go straight to the
// stage-output variables, so the intermediate struct is never materialized. This
// is the mirror of analyze_param_sroa for the return path.
static void analyze_output_sroa(emit_t *e, svsl_type_id_t ret_type) {
	e->output_sroa_var = SVSL_IR_NONE;
	const svsl_ir_func_t *fn = e->fn;
	const svsl_type_t    *rt = svsl_type_get(&e->prog->types, ret_type);
	if (rt->kind != svsl_type_struct ||
	    e->prog->types.structs.items[rt->struct_index].members.count > SVSL_EMIT_MAX_MEMBERS)
		return;

	// candidate = the var loaded whole into every value-returning return
	uint32_t cand = SVSL_IR_NONE;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != svsl_ir_return || in->args[0] == SVSL_IR_NONE) continue;
		const svsl_ir_inst_t *rv = &fn->insts.items[in->args[0]];
		if (rv->op != svsl_ir_load) return;                  // returns a computed value → no SROA
		uint32_t v = rv->args[0];
		if (fn->insts.items[v].op != svsl_ir_var || fn->insts.items[v].type != ret_type) return;
		if      (cand == SVSL_IR_NONE) cand = v;
		else if (cand != v)            return;               // returns load different vars
	}
	if (cand == SVSL_IR_NONE) return;

	// every reference to the var (and to its member chains / whole loads) must fit
	// the pure-output-plumbing shape, or we keep the local struct.
	int32_t dummy;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in   = &fn->insts.items[i];
		uint32_t              mask  = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++) {
			if (!(mask & (1u << a)) || in->args[a] >= (uint32_t)fn->insts.count) continue;
			uint32_t o = in->args[a];
			if (o == cand) { // direct use: only a const member-chain base or a whole-load pointer
				bool ok = (a == 0) && (((svsl_ir_op_)in->op == svsl_ir_chain &&
				                        sroa_out_member_chain(fn, (uint32_t)i, cand, &dummy)) ||
				                       (svsl_ir_op_)in->op == svsl_ir_load);
				if (!ok) return;
			}
			if (sroa_out_member_chain(fn, o, cand, &dummy)) // member chain: only a store pointer
				if (!((svsl_ir_op_)in->op == svsl_ir_store && a == 0)) return;
			if (fn->insts.items[o].op == svsl_ir_load && fn->insts.items[o].args[0] == cand)
				if ((svsl_ir_op_)in->op != svsl_ir_return) return; // whole load feeds only return
		}
		if (svsl_ir_aux_holds_values(in)) // any use of the var / its chains in aux → veto
			for (uint32_t k = 0; k < in->aux_count; k++) {
				uint32_t o = fn->aux.items[in->aux + k];
				if (o == cand || sroa_out_member_chain(fn, o, cand, &dummy)) return;
			}
	}
	e->output_sroa_var = cand;
}

// Emit-level liveness: mark which IR values an *emitted* operand actually reads,
// mirroring emit_inst's chain path. Values feeding only address computations
// that emit re-inlines (a buffer/resource-member `ptr` folded into its chain) or
// resolves away (an SROA'd member index — the chain becomes the interface
// variable, so its constant index is never emitted) end up referenced in the IR
// but orphaned in the output. `referenced` lets the body loop skip emitting such
// dead constants/pointers. Constants/pointers/undefs are value leaves (they read
// nothing), so a single pass with no fixpoint is exact.
static void analyze_emit_liveness(emit_t *e, uint8_t *referenced) {
	const svsl_ir_func_t *fn = e->fn;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_nop || op == svsl_ir_var || op == svsl_ir_param) continue;
		if (op == svsl_ir_load && inst->args[0] == e->output_sroa_var) continue; // whole-load elided

		if (op == svsl_ir_chain) {
			const svsl_ir_inst_t *base = &fn->insts.items[inst->args[0]];
			int32_t sp, sm;
			if (sroa_member_chain(e, (uint32_t)i, &sp, &sm) && e->param_sroa[sp]) {
				for (uint32_t k = 1; k < inst->aux_count; k++)   // single-index: none; multi: trailing
					referenced[fn->aux.items[inst->aux + k]] = 1;
			} else if (inst->args[0] == e->output_sroa_var) {
				// resolves to the stage-output variable: references nothing
			} else if (base->op == svsl_ir_ptr &&
			           (base->args[0] == svsl_ref_buffer_member || base->args[0] == svsl_ref_resource)) {
				for (uint32_t k = 0; k < inst->aux_count; k++)   // ptr folded in: only the indices
					referenced[fn->aux.items[inst->aux + k]] = 1;
			} else {
				referenced[inst->args[0]] = 1;
				for (uint32_t k = 0; k < inst->aux_count; k++)
					referenced[fn->aux.items[inst->aux + k]] = 1;
			}
			continue;
		}

		uint32_t mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++)
			if ((mask & (1u << a)) && inst->args[a] < (uint32_t)fn->insts.count)
				referenced[inst->args[a]] = 1;
		if (svsl_ir_aux_holds_values(inst))
			for (uint32_t k = 0; k < inst->aux_count; k++)
				referenced[fn->aux.items[inst->aux + k]] = 1;
	}
}

#include "emit_intrinsics.inc"

static void emit_body(emit_t *e, uint32_t fn_id, uint32_t void_type, uint32_t fn_type) {
	const svsl_ir_func_t *fn  = e->fn;
	svsl_spv_t           *spv = &e->spv;
	svsl_spv_stream_t    *fs  = &spv->funcs;
	const svsl_entry_t   *entry = fn->entry;

	// nesting can't exceed the instruction count; size the CF stack and the
	// if/else-matching stack to that bound (no fixed cap to overflow)
	int32_t nest_max = fn->insts.count > 0 ? fn->insts.count : 1;
	e->cf = svsl_arena_alloc(e->arena, (size_t)nest_max * sizeof(*e->cf));

	// pre-scan: which ifs have an else
	{
		int32_t *stack = svsl_arena_alloc(e->arena, (size_t)nest_max * sizeof(int32_t));
		int32_t  depth = 0;
		for (int32_t i = 0; i < fn->insts.count; i++) {
			switch ((svsl_ir_op_)fn->insts.items[i].op) {
			case svsl_ir_if:     stack[depth++] = i; break;
			case svsl_ir_else:   if (depth > 0) e->if_has_else[stack[depth - 1]] = 1; break;
			case svsl_ir_end_if: depth--; break;
			default: break;
			}
		}
	}

	const svsl_func_info_t *entry_info = svsl_program_func_info(e->prog, entry->func);
	svsl_type_id_t          ret_type   = entry_info ? entry_info->return_type : SVSL_TYPE_NONE;

	analyze_param_sroa(e);              // which struct params skip the shadow-var materialization
	analyze_output_sroa(e, ret_type);   // whether the return struct skips its local var entirely

	uint8_t *referenced = svsl_arena_alloc(e->arena, (size_t)(fn->insts.count > 0 ? fn->insts.count : 1));
	analyze_emit_liveness(e, referenced); // constants/ptrs orphaned by chain re-inlining / SROA

	svsl_spv_inst4(spv, fs, SpvOpFunction, void_type, fn_id, SpvFunctionControlMaskNone, fn_type);
	begin_block(e, svsl_spv_id(spv));

	// all function-local variables first (SPIR-V requires them at block start)
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op == svsl_ir_var) {
			if ((uint32_t)i == e->output_sroa_var) continue; // scalar-replaced return struct
			uint32_t ptr = spv_ptr_type(e, SpvStorageClassFunction, spv_type_for(e, inst->type));
			uint32_t var = svsl_spv_id(spv);
			svsl_spv_inst3(spv, fs, SpvOpVariable, ptr, var, SpvStorageClassFunction);
			if (inst->name.len)
				svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, inst->name);
			e->value_ids[i]   = var;
			e->value_class[i] = SpvStorageClassFunction;
		} else if (inst->op == svsl_ir_param) {
			uint32_t param = inst->args[0];
			if (e->param_sroa[param]) continue; // members resolve to input vars directly
			if (!e->param_shadow[param]) {
				uint32_t ptr = spv_ptr_type(e, SpvStorageClassFunction, spv_type_for(e, inst->type));
				uint32_t var = svsl_spv_id(spv);
				svsl_spv_inst3(spv, fs, SpvOpVariable, ptr, var, SpvStorageClassFunction);
				if (inst->name.len)
					svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, inst->name);
				e->param_shadow[param] = var;
			}
			e->value_ids[i]   = e->param_shadow[param];
			e->value_class[i] = SpvStorageClassFunction;
		}
	}

	// prologue: flatten stage inputs into the param shadow variables
	{
		int32_t      location    = 0;
		int32_t      vs_input_at = 0; // running index into prog->vertex_inputs
		svsl_sem_io_ io = entry->stage == svsl_stage_vertex ? svsl_sem_vs_in :
		                  entry->stage == svsl_stage_pixel  ? svsl_sem_ps_in : svsl_sem_cs_in;
		for (int32_t p = 0; p < entry->func->param_count; p++) {
			svsl_type_id_t     ptype = entry_info ? entry_info->param_types[p] : SVSL_TYPE_NONE;
			const svsl_type_t *pt    = svsl_type_get(&e->prog->types, ptype);

			uint32_t value;
			if (pt->kind == svsl_type_struct && e->param_sroa[p]) {
				// scalar-replaced: declare only referenced members, mapping each to
				// its input variable. Location must still advance for every member so
				// the used ones (and later params) keep the locations the other stage
				// expects; only non-builtin members consume a location.
				const svsl_struct_info_t *si = &e->prog->types.structs.items[pt->struct_index];
				for (int32_t m = 0; m < si->members.count && m < SVSL_EMIT_MAX_MEMBERS; m++) {
					const svsl_member_t *member = &si->members.items[m];
					if (member->explicit_location >= 0) location = member->explicit_location;
					bool counted = !sroa_sem_is_builtin(member->semantic, io);
					if (e->param_member_var[p * SVSL_EMIT_MAX_MEMBERS + m]) { // referenced
						if (counted)
							record_vs_input(e, &vs_input_at, member->name, entry->func->loc, location);
						uint32_t var = make_io_var(e, member->type, false, member->name);
						io_decorate(e, var, member->semantic, io, &location, member->type, member->interp);
						e->param_member_var[p * SVSL_EMIT_MAX_MEMBERS + m] = var;
					} else if (counted) {
						record_vs_input(e, &vs_input_at, member->name, entry->func->loc, -1);
						location += location_span(&e->prog->types, member->type); // slot still consumed
					}
				}
				continue; // no shadow var, no construct, no store
			}
			if (pt->kind == svsl_type_struct) {
				const svsl_struct_info_t *si = &e->prog->types.structs.items[pt->struct_index];
				uint32_t *parts = svsl_arena_alloc(e->arena, (size_t)si->members.count * sizeof(uint32_t));
				for (int32_t m = 0; m < si->members.count; m++) {
					const svsl_member_t *member = &si->members.items[m];
					uint32_t var = make_io_var(e, member->type, false, member->name);
					if (member->explicit_location >= 0) location = member->explicit_location;
					if (!sroa_sem_is_builtin(member->semantic, io))
						record_vs_input(e, &vs_input_at, member->name, entry->func->loc, location);
					io_decorate(e, var, member->semantic, io, &location, member->type, member->interp);
					parts[m] = emit_value_inst(e, fs, SpvOpLoad, spv_type_for(e, member->type),
					                           (uint32_t[]){ var }, 1);
				}
				value = emit_value_inst(e, fs, SpvOpCompositeConstruct, spv_type_for(e, ptype),
				                        parts, (uint32_t)si->members.count);
			} else {
				bool counted = !sroa_sem_is_builtin(entry->func->params[p]->semantic, io);
				// glslang strips vertex inputs nothing reads (their location is
				// still consumed); match it so this module, its metadata, and the
				// reference compiler agree on the input interface
				if (counted && entry->stage == svsl_stage_vertex && !e->param_shadow[p]) {
					record_vs_input(e, &vs_input_at, entry->func->params[p]->name,
					                entry->func->loc, -1);
					location += location_span(&e->prog->types, ptype);
					continue;
				}
				uint32_t var = make_io_var(e, ptype, false, entry->func->params[p]->name);
				if (counted)
					record_vs_input(e, &vs_input_at, entry->func->params[p]->name,
					                entry->func->loc, location);
				io_decorate(e, var, entry->func->params[p]->semantic, io, &location,
				            ptype, entry->func->params[p]->interp);
				value = emit_value_inst(e, fs, SpvOpLoad, spv_type_for(e, ptype), (uint32_t[]){ var }, 1);
			}
			if (e->param_shadow[p]) // unreferenced params have no shadow variable
				svsl_spv_inst2(spv, fs, SpvOpStore, e->param_shadow[p], value);
		}
		// every vertex_inputs entry must have been visited, or the walk above no
		// longer mirrors collect_vertex_inputs
		if (e->vs_input_locations && vs_input_at != e->prog->vertex_inputs.count)
			eerr(e, entry->func->loc, "vertex input metadata diverged from the emitted interface");
	}

	// outputs: flattened from the entry's return type
	const svsl_type_t *rt = svsl_type_get(&e->prog->types, ret_type);
	{
		int32_t      location = 0;
		svsl_sem_io_ io = entry->stage == svsl_stage_vertex ? svsl_sem_vs_out : svsl_sem_ps_out;
		int32_t      max_out = rt->kind == svsl_type_struct ?
			e->prog->types.structs.items[rt->struct_index].members.count : 1;
		e->output_vars = svsl_arena_alloc(e->arena, (size_t)(max_out > 0 ? max_out : 1) * sizeof(uint32_t));
		if (rt->kind == svsl_type_struct) {
			const svsl_struct_info_t *si = &e->prog->types.structs.items[rt->struct_index];
			for (int32_t m = 0; m < si->members.count; m++) {
				const svsl_member_t *member = &si->members.items[m];
				uint32_t var = make_io_var(e, member->type, true, member->name);
				if (member->explicit_location >= 0) location = member->explicit_location;
				io_decorate(e, var, member->semantic, io, &location, member->type, member->interp);
				e->output_vars[e->output_count++] = var;
			}
		} else if (rt->kind != svsl_type_void) {
			uint32_t var = make_io_var(e, ret_type, true, entry->func->name);
			io_decorate(e, var, entry->func->return_semantic, io, &location, ret_type, 0);
			e->output_vars[e->output_count++] = var;
		}
	}

	// body
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_nop || op == svsl_ir_var || op == svsl_ir_param) continue;
		if (op == svsl_ir_load && inst->args[0] == e->output_sroa_var) continue; // whole output load elided
		if ((op == svsl_ir_const || op == svsl_ir_ptr || op == svsl_ir_undef) && !referenced[i])
			continue; // a constant/pointer left orphaned by chain re-inlining or SROA
		emit_inst(e, i, inst, rt, ret_type);
	}

	if (!e->terminated) svsl_spv_inst(spv, fs, SpvOpReturn, NULL, 0);
	svsl_spv_inst(spv, fs, SpvOpFunctionEnd, NULL, 0);
}

// --- entry ---------------------------------------------------------------------------

bool svsl_spirv_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                     const svsl_ir_func_t *fn, svsl_spirv_blob_t *out_blob,
                     svsl_diag_list_t *ref_diags) {
	emit_t e = { .arena = arena, .prog = prog, .fn = fn, .diags = ref_diags };
	svsl_spv_init(&e.spv, arena);

	int32_t inst_count = fn->insts.count > 0 ? fn->insts.count : 1;
	e.value_ids         = svsl_arena_alloc(arena, (size_t)inst_count * 4);
	e.value_class       = svsl_arena_alloc(arena, (size_t)inst_count * 4);
	e.value_spec        = svsl_arena_alloc(arena, (size_t)inst_count);
	e.if_has_else       = svsl_arena_alloc(arena, (size_t)inst_count);
	e.type_cap          = prog->types.types.count > 0 ? prog->types.types.count : 1;
	e.type_ids          = svsl_arena_alloc(arena, (size_t)e.type_cap * 4);
	for (int32_t l = 0; l < 4; l++)
		e.laid_ids[l] = svsl_arena_alloc(arena, (size_t)e.type_cap * 4);
	e.value_layout      = svsl_arena_alloc(arena, (size_t)inst_count);
	e.buffer_vars       = svsl_arena_alloc(arena, (size_t)(prog->buffers.count > 0 ? prog->buffers.count : 1) * 4);
	e.resource_vars     = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	e.resource_img_type = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	e.workgroup_ids     = svsl_arena_alloc(arena, (size_t)(prog->workgroup_vars.count > 0 ? prog->workgroup_vars.count : 1) * 4);
	e.const_global_ids  = svsl_arena_alloc(arena, (size_t)(prog->const_globals.count > 0 ? prog->const_globals.count : 1) * 4);
	int32_t pcount      = fn->entry->func->param_count > 0 ? fn->entry->func->param_count : 1;
	e.param_shadow      = svsl_arena_alloc(arena, (size_t)pcount * 4);
	e.param_sroa        = svsl_arena_alloc(arena, (size_t)pcount);
	e.param_member_var  = svsl_arena_alloc(arena, (size_t)pcount * SVSL_EMIT_MAX_MEMBERS * 4);
	e.builtin_input     = svsl_arena_alloc(arena, 16 * 4);
	e.spec_const_ids    = svsl_arena_alloc(arena, (size_t)(prog->spec_consts.count > 0 ? prog->spec_consts.count : 1) * 4);
	e.sampler_vars      = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	if (fn->entry->stage == svsl_stage_vertex) {
		e.vs_input_locations = svsl_arena_alloc(arena, (size_t)(prog->vertex_inputs.count > 0 ? prog->vertex_inputs.count : 1) * 4);
		for (int32_t i = 0; i < prog->vertex_inputs.count; i++)
			e.vs_input_locations[i] = -1;
	}

	svsl_spv_cap(&e.spv, SpvCapabilityShader);
	e.spv.glsl450 = svsl_spv_id(&e.spv);
	svsl_spv_inst_str(&e.spv, &e.spv.imports, SpvOpExtInstImport,
	                  (uint32_t[]){ e.spv.glsl450 }, 1, svsl_str("GLSL.std.450"));
	svsl_spv_inst2(&e.spv, &e.spv.memory, SpvOpMemoryModel,
	               SpvAddressingModelLogical, SpvMemoryModelGLSL450);

	create_globals(&e);

	uint32_t void_type = svsl_spv_type(&e.spv, SpvOpTypeVoid, NULL, 0);
	uint32_t fn_type   = svsl_spv_type(&e.spv, SpvOpTypeFunction, (uint32_t[]){ void_type }, 1);
	uint32_t fn_id     = svsl_spv_id(&e.spv);
	svsl_spv_inst_str(&e.spv, &e.spv.debug, SpvOpName, (uint32_t[]){ fn_id }, 1, fn->entry->name);

	emit_body(&e, fn_id, void_type, fn_type);

	// entry point + execution modes
	SpvExecutionModel model = fn->entry->stage == svsl_stage_vertex ? SpvExecutionModelVertex :
	                          fn->entry->stage == svsl_stage_pixel  ? SpvExecutionModelFragment :
	                          SpvExecutionModelGLCompute;
	// operands after the name string: build manually since the string is inline
	{
		svsl_str_t name = fn->entry->name;
		uint32_t str_words = ((uint32_t)name.len + 1 + 3) / 4;
		uint32_t iface     = (uint32_t)e.interface.count;
		svsl_array_push(arena, &e.spv.entries,
		                ((2 + str_words + iface + 1) << SpvWordCountShift) | SpvOpEntryPoint);
		svsl_array_push(arena, &e.spv.entries, (uint32_t)model);
		svsl_array_push(arena, &e.spv.entries, fn_id);
		for (uint32_t w = 0; w < str_words; w++) {
			uint32_t word = 0;
			for (uint32_t k = 0; k < 4; k++) {
				uint32_t index = w * 4 + k;
				if (index < (uint32_t)name.len) word |= (uint32_t)(uint8_t)name.ptr[index] << (k * 8);
			}
			svsl_array_push(arena, &e.spv.entries, word);
		}
		for (uint32_t i = 0; i < iface; i++)
			svsl_array_push(arena, &e.spv.entries, e.interface.items[i]);
	}
	if (fn->entry->stage == svsl_stage_pixel) {
		svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id, SpvExecutionModeOriginUpperLeft);
		if (e.needs_depth_replacing)
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id, SpvExecutionModeDepthReplacing);
		// conservative depth keeps early-Z alive: the shader promises the
		// written depth only moves in one direction from the rasterized value
		if (e.depth_mode == 1)
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id, SpvExecutionModeDepthGreater);
		if (e.depth_mode == 2)
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id, SpvExecutionModeDepthLess);
	}
	if (fn->entry->stage == svsl_stage_compute) {
		uint32_t ops[5] = { fn_id, SpvExecutionModeLocalSize,
		                    (uint32_t)fn->entry->workgroup[0],
		                    (uint32_t)fn->entry->workgroup[1],
		                    (uint32_t)fn->entry->workgroup[2] };
		svsl_spv_inst(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, ops, 5);
	}
	for (int32_t a = 0; a < fn->entry->func->attrs.count; a++)
		if (svsl_str_eq_cstr(fn->entry->func->attrs.items[a].name, "early_depth_stencil") ||
		    svsl_str_eq_cstr(fn->entry->func->attrs.items[a].name, "earlydepthstencil")) // HLSL spelling
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id,
			               SpvExecutionModeEarlyFragmentTests);

	out_blob->words              = svsl_spv_finalize(&e.spv, &out_blob->word_count);
	out_blob->vs_input_locations = e.vs_input_locations;
	return !e.failed;
}
