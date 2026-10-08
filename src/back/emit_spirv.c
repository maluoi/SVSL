// IR -> SPIR-V. Matches glslang's HLSL model where it matters for StereoKit:
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
#if defined(__GNUC__) || defined(__clang__)
	#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

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
	uint8_t  *qcom_res_use;     // QCOM image-processing use class per resource
	                            // (a conflicting later use is an error)
	svsl_array_t(uint64_t) qcom_decorated; // (var id << 32 | decoration) emitted so far
	uint32_t *workgroup_ids;
	uint32_t *const_global_ids;
	uint32_t *private_ids;      // one Private OpVariable per private (non-const static) global
	uint32_t *io_vars;          // per entry io slot: its Input/Output variable, made on first use
	uint32_t *builtin_input;    // subgroup builtins etc., created on demand
	uint32_t *spec_const_ids;   // one OpSpecConstant per spec-constant index
	uint32_t *sampler_vars;     // standalone sampler variables for cross-paired sampling
	int32_t  *io_locations;     // per entry io slot: its decorated location, -1 = none (record_io_locations)

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
	uint8_t *loop_exit;   // per inst index: loop_exit_ role (see analyze_loop_exits)

	bool    needs_depth_replacing;
	uint8_t depth_mode; // conservative depth: 1 = DepthGreater, 2 = DepthLess

	bool failed;
} emit_t;

static void eerr(emit_t *e, svsl_loc_t loc, const char *msg) {
	svsl_diag_add(e->arena, e->diags, svsl_severity_error, loc, "%s", msg);
	e->failed = true;
}

// --- svsl type -> SPIR-V type ---------------------------------------------------

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
	uint32_t format  = storage ? svsl_image_format_for(t) : SpvImageFormatUnknown;
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

// [tile_attachment] (VK_QCOM_tile_shading): the variable keeps its set/binding
// descriptor but lives in the TileAttachmentQCOM storage class
static SpvStorageClass tile_or_uniform_class(emit_t *e, const svsl_resource_t *res) {
	if (!res->tile_attachment) return SpvStorageClassUniformConstant;
	if (e->fn->entry->stage == svsl_stage_vertex)
		eerr(e, res->loc, "tile attachments need a pixel or compute shader");
	svsl_spv_cap(&e->spv, SpvCapabilityTileShadingQCOM);
	svsl_spv_extension(&e->spv, "SPV_QCOM_tile_shading");
	return SpvStorageClassTileAttachmentQCOM;
}

// one scalar leaf of a constant global's value (const_eval.h encoding = IR const bits)
static uint32_t spv_const_leaf(emit_t *e, svsl_scalar_ scalar, uint64_t bits) {
	return svsl_spv_const(&e->spv, spv_scalar_type(e, scalar), bits, svsl_scalar_size(scalar) == 8,
	                      scalar == svsl_scalar_bool);
}

// a constant global's value as a SPIR-V constant of `type`, consuming its leaves
static uint32_t spv_const_value(emit_t *e, svsl_type_id_t type, const uint64_t **ref_leaf) {
	const svsl_type_t *t = svsl_type_get(&e->prog->types, type);
	if (t->kind == svsl_type_scalar) return spv_const_leaf(e, t->scalar, *(*ref_leaf)++);

	uint32_t  count = (uint32_t)(t->kind == svsl_type_vector ? t->count : t->kind == svsl_type_matrix ? t->rows : t->array_count);
	uint32_t *parts = svsl_arena_alloc(e->arena, (size_t)count * sizeof(uint32_t));
	if (t->kind == svsl_type_vector) {
		for (uint32_t i = 0; i < count; i++) parts[i] = spv_const_leaf(e, t->scalar, *(*ref_leaf)++);
	} else if (t->kind == svsl_type_matrix) { // rows of vec(cols): SPIR-V matrix vectors are HLSL rows
		uint32_t row_type = svsl_spv_type(&e->spv, SpvOpTypeVector,
		                                  (uint32_t[]){ spv_scalar_type(e, t->scalar), (uint32_t)t->cols }, 2);
		for (uint32_t r = 0; r < count; r++) {
			uint32_t comps[4];
			for (int32_t c = 0; c < t->cols; c++) comps[c] = spv_const_leaf(e, t->scalar, *(*ref_leaf)++);
			parts[r] = svsl_spv_const_composite(&e->spv, SpvOpConstantComposite, row_type, comps, (uint32_t)t->cols);
		}
	} else { // array
		for (uint32_t i = 0; i < count; i++) parts[i] = spv_const_value(e, t->elem, ref_leaf);
	}
	return svsl_spv_const_composite(&e->spv, SpvOpConstantComposite, spv_type_for(e, type), parts, count);
}

static void create_globals(emit_t *e) {
	const svsl_program_t *prog = e->prog;
	svsl_spv_t           *spv  = &e->spv;

	// Emit only the globals this stage actually references. A fragment shader
	// that returns its color input should not declare the system cbuffer, the
	// instance buffer, etc. - glslang+spirv-opt strip these, and emitting them
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
			e->resource_vars[i] = emit_resource_var(e, tile_or_uniform_class(e, res), type,
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
			e->resource_vars[i] = emit_resource_var(e, tile_or_uniform_class(e, res), img,
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

	// private and constant globals: only the ones this entry still references
	// after optimization are declared (a constant table every load of which
	// folded away, or one belonging to a function this entry never calls, isn't)
	uint8_t *private_used = svsl_arena_alloc(e->arena, (size_t)(prog->private_globals.count > 0 ? prog->private_globals.count : 1));
	uint8_t *const_used   = svsl_arena_alloc(e->arena, (size_t)(prog->const_globals.count > 0 ? prog->const_globals.count : 1));
	for (int32_t i = 0; i < e->fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &e->fn->insts.items[i];
		if (in->op != svsl_ir_ptr) continue;
		if ((svsl_ref_)in->args[0] == svsl_ref_private_global) private_used[in->args[1]] = 1;
		if ((svsl_ref_)in->args[0] == svsl_ref_const_global)   const_used[in->args[1]]   = 1;
	}

	// private globals: zero-initialized Private variables (initializers are
	// stores in each entry's prologue - see ir_build's lower_private_globals)
	for (int32_t i = 0; i < prog->private_globals.count; i++) {
		if (!private_used[i]) continue;
		const svsl_global_t *g    = &prog->private_globals.items[i];
		uint32_t             type = spv_type_for(e, g->type);
		uint32_t             ptr  = spv_ptr_type(e, SpvStorageClassPrivate, type);
		uint32_t             var  = svsl_spv_id(spv);
		svsl_spv_inst4(spv, &spv->types, SpvOpVariable, ptr, var, SpvStorageClassPrivate,
		               svsl_spv_const_null(spv, type));
		svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, g->name);
		e->private_ids[i] = var;
	}

	// const globals with constant initializers -> Private variables
	for (int32_t i = 0; i < prog->const_globals.count; i++) {
		const svsl_global_t *g = &prog->const_globals.items[i];
		if (!const_used[i] || !g->value) continue; // a non-constant one errors at its use
		const uint64_t *leaf = g->value;
		uint32_t        init = spv_const_value(e, g->type, &leaf);
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

// Decorates an interface variable for its slot: the builtin it is, or its
// location (sema numbered every slot) and interpolation.
static void io_decorate(emit_t *e, uint32_t var, const svsl_io_slot_t *slot) {
	svsl_stage_        stage = e->fn->entry->stage;
	svsl_sem_io_       io    = slot->output ? (stage == svsl_stage_vertex ? svsl_sem_vs_out : svsl_sem_ps_out)
	                                        : (stage == svsl_stage_vertex ? svsl_sem_vs_in :
	                                           stage == svsl_stage_pixel  ? svsl_sem_ps_in : svsl_sem_cs_in);
	const svsl_type_t *vt    = svsl_type_get(&e->prog->types, slot->type);
	bool integer_type = (vt->kind == svsl_type_scalar || vt->kind == svsl_type_vector) &&
	                    (vt->scalar >= svsl_scalar_int8 && vt->scalar <= svsl_scalar_uint64);
	uint8_t interp = slot->interp;

	svsl_semantic_info_t info;
	if (slot->location < 0 && svsl_semantic_lookup(slot->semantic, io, &info) && info.is_builtin) {
		svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationBuiltIn, info.builtin);
		if (info.builtin == SpvBuiltInViewIndex) svsl_spv_cap(&e->spv, SpvCapabilityMultiView);
		if (info.builtin == SpvBuiltInLayer) {
			if (io == svsl_sem_vs_out) { // SV_RenderTargetArrayIndex from the vertex stage
				svsl_spv_cap(&e->spv, SpvCapabilityShaderViewportIndexLayerEXT);
				svsl_spv_extension(&e->spv, "SPV_EXT_shader_viewport_index_layer");
			} else { // reading it back in the pixel stage takes the Geometry capability
				svsl_spv_cap(&e->spv, SpvCapabilityGeometry);
			}
		}
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
	svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationLocation, (uint32_t)slot->location);

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

// The Input/Output variable for io slot `slot`, declared on first use: only the
// slots the optimized body still reads or writes reach the interface.
static uint32_t io_var(emit_t *e, int32_t slot) {
	if (e->io_vars[slot]) return e->io_vars[slot];
	const svsl_io_slot_t *s = &e->fn->entry->io.items[slot];
	uint32_t var = make_io_var(e, s->type, s->output, s->name);
	io_decorate(e, var, s);
	e->io_vars[slot] = var;
	return var;
}

// Each io slot's decorated location, or -1: a builtin, or a slot the optimized
// body never touches, so its variable was never declared (glslang strips unread
// inputs too; the location stays consumed). The SKS writer's vertex-input
// records come from this, so they mirror the module exactly.
static void record_io_locations(emit_t *e) {
	const svsl_entry_t *entry = e->fn->entry;
	for (int32_t i = 0; i < entry->io.count; i++) {
		const svsl_io_slot_t *slot = &entry->io.items[i];
		e->io_locations[i] = e->io_vars[i] ? slot->location : -1;
		if (svsl_io_is_attribute(entry, slot) && slot->location > 255)
			eerr(e, slot->loc, "vertex input location exceeds the container's limit of 255");
	}
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

// QCOM image processing (VK_QCOM_image_processing[2]): decorations are inferred
// from use, and a decorated resource is exclusive to its op family - the runtime
// binds it through a dedicated descriptor type / sampler create flag, so mixing
// uses cannot be satisfied by any binding. First use classifies and decorates;
// a conflicting later use is a compile error. Classes: svsl_qcom_use_
// (emit_spirv.h - the SKS writer consumes the recorded array).
static const char *qcom_use_names[] = { "", "ordinary texturing", "a weight texture",
	"a block-match texture", "an image-processing sampler", "a window block-match sampler" };

// conflict tracking only - decorations are per *variable* (qcom_decorate_once),
// since one sampler resource can be reached through a combined variable in one
// use and its own variable in another
static void qcom_classify(emit_t *e, int32_t res_index, uint8_t use, svsl_loc_t loc) {
	uint8_t *cur = &e->qcom_res_use[res_index];
	if (*cur == use) return;
	if (*cur == 0) { *cur = use; return; }
	const svsl_resource_t *res = &e->prog->resources.items[res_index];
	svsl_diag_add(e->arena, e->diags, svsl_severity_error, loc,
	              "'%.*s' is used both as %s and %s - QCOM image-processing resources are exclusive to one use",
	              res->name.len, res->name.ptr, qcom_use_names[*cur], qcom_use_names[use]);
}

static void qcom_decorate_once(emit_t *e, uint32_t var, uint32_t dec) {
	uint64_t key = (uint64_t)var << 32 | dec;
	for (int32_t i = 0; i < e->qcom_decorated.count; i++)
		if (e->qcom_decorated.items[i] == key) return;
	svsl_array_push(e->arena, &e->qcom_decorated, key);
	svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, dec);
}

// window block-match sampled image. The validator traces the QCOM decorations
// through *direct* OpLoads, so the pair is either the texture's own combined
// variable (both decorations land on it) or separate image + sampler variable
// loads - a texture fused with a different sampler cannot be expressed.
static uint32_t qcom_window_pair(emit_t *e, int32_t tex, uint32_t sampler_res, svsl_loc_t loc) {
	const svsl_resource_t *res = &e->prog->resources.items[tex];
	svsl_spv_stream_t     *fs  = &e->spv.funcs;
	uint32_t img_type = e->resource_img_type[tex];
	uint32_t si_type  = svsl_spv_type(&e->spv, SpvOpTypeSampledImage, (uint32_t[]){ img_type }, 1);

	qcom_classify(e, tex, svsl_qcom_use_block_match, loc);
	qcom_classify(e, (int32_t)sampler_res, svsl_qcom_use_bm_window_sampler, loc);

	if (res->sampler_slot >= 0) { // fused: combined variable, both decorations
		const svsl_resource_t *smp = &e->prog->resources.items[sampler_res];
		if (smp->bind.slot != res->sampler_slot || smp->bind.space != res->bind.space) {
			eerr(e, loc, "a window block-match texture fused with one sampler cannot be used with another");
			return 0;
		}
		qcom_decorate_once(e, e->resource_vars[tex], SpvDecorationBlockMatchTextureQCOM);
		qcom_decorate_once(e, e->resource_vars[tex], SpvDecorationBlockMatchSamplerQCOM);
		return emit_value_inst(e, fs, SpvOpLoad, si_type, (uint32_t[]){ e->resource_vars[tex] }, 1);
	}
	uint32_t svar = e->resource_vars[sampler_res] ? e->resource_vars[sampler_res]
	                                              : sampler_var_for(e, sampler_res);
	qcom_decorate_once(e, e->resource_vars[tex], SpvDecorationBlockMatchTextureQCOM);
	qcom_decorate_once(e, svar, SpvDecorationBlockMatchSamplerQCOM);
	uint32_t image   = emit_value_inst(e, fs, SpvOpLoad, img_type,
	                                   (uint32_t[]){ e->resource_vars[tex] }, 1);
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
	// order matches builtin_vars[] in tables/intrinsics.c
	static const struct { SpvBuiltIn builtin; uint8_t comps; } rows[] = {
		{ SpvBuiltInSubgroupSize, 1 }, { SpvBuiltInSubgroupLocalInvocationId, 1 },
		{ SpvBuiltInSubgroupId, 1 },   { SpvBuiltInNumSubgroups, 1 },
		{ SpvBuiltInTileOffsetQCOM, 2 }, { SpvBuiltInTileDimensionQCOM, 3 },
		{ SpvBuiltInTileApronSizeQCOM, 2 },
	};
	if (index >= sizeof(rows) / sizeof(rows[0])) index = 0;
	if (rows[index].builtin >= SpvBuiltInTileOffsetQCOM &&
	    rows[index].builtin <= SpvBuiltInTileApronSizeQCOM) {
		svsl_spv_cap(&e->spv, SpvCapabilityTileShadingQCOM);
		svsl_spv_extension(&e->spv, "SPV_QCOM_tile_shading");
	} else {
		svsl_spv_cap(&e->spv, SpvCapabilityGroupNonUniform);
	}
	uint32_t u32  = spv_scalar_type(e, svsl_scalar_uint32);
	uint32_t type = rows[index].comps == 1 ? u32
	              : svsl_spv_type(&e->spv, SpvOpTypeVector, (uint32_t[]){ u32, rows[index].comps }, 2);
	uint32_t ptr = spv_ptr_type(e, SpvStorageClassInput, type);
	uint32_t var = svsl_spv_id(&e->spv);
	svsl_spv_inst3(&e->spv, &e->spv.types, SpvOpVariable, ptr, var, SpvStorageClassInput);
	svsl_spv_inst3(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationBuiltIn,
	               rows[index].builtin);
	if (e->fn->entry->stage == svsl_stage_pixel) // integer inputs must be Flat
		svsl_spv_inst2(&e->spv, &e->spv.decor, SpvOpDecorate, var, SpvDecorationFlat);
	svsl_array_push(e->arena, &e->interface, var);
	e->builtin_input[index] = var;
	return var;
}

// component count for a builtin_input_var index (the .inc load needs the type)
static uint32_t builtin_input_comps(uint32_t index) {
	static const uint8_t comps[] = { 1, 1, 1, 1, 2, 3, 2 };
	return index < sizeof(comps) ? comps[index] : 1;
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

// Emit-level liveness: mark which IR values an *emitted* operand actually reads,
// mirroring emit_inst's chain path. A buffer/resource-member `ptr` that emit
// folds into its chain ends up referenced in the IR but orphaned in the output.
// `referenced` lets the body loop skip emitting such dead constants/pointers.
// Constants/pointers/undefs are value leaves (they read nothing), so a single
// pass with no fixpoint is exact.
static void analyze_emit_liveness(emit_t *e, uint8_t *referenced) {
	const svsl_ir_func_t *fn = e->fn;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_nop || op == svsl_ir_var) continue;

		if (op == svsl_ir_chain) {
			const svsl_ir_inst_t *base = &fn->insts.items[inst->args[0]];
			if (base->op == svsl_ir_ptr &&
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

// Loop exit tests. `for`/`while` lower their condition to `if (!cond) break;`
// (flagged svsl_ir_flag_loop_exit by ir_build). Emitted literally, that is a
// selection construct wrapped around a break block. glslang instead ends the
// condition's block with one conditional branch straight to the loop merge, and
// drivers key their loop analysis on that shape: Adreno (Quest 3) could not
// compile a large encoder in the selection form in any usable time
// (docs/dev/case-study-astc-encoders.md has the measurements). So each flagged
// test is emitted as that exit branch - usually right after the header, or after
// whatever constructs the condition itself lowered to (an inlined call's early
// return). A `log_not` condition with no other user folds into the branch by
// swapping its targets. Encoding only: the IR keeps the plain if/break form every
// pass already understands.
typedef enum loop_exit_ {
	loop_exit_none = 0,
	loop_exit_branch, // svsl_ir_if emitted as the conditional exit branch
	loop_exit_skip,   // that if's break/end_if, or its folded log_not: emit nothing
} loop_exit_;

static int32_t next_live(const svsl_ir_func_t *fn, int32_t i) {
	for (i++; i < fn->insts.count; i++)
		if (fn->insts.items[i].op != svsl_ir_nop) return i;
	return fn->insts.count;
}

static void analyze_loop_exits(emit_t *e) {
	const svsl_ir_func_t *fn    = e->fn;
	int32_t               n     = fn->insts.count;
	bool                  found = false;
	for (int32_t k = 0; k < n; k++) {
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		if (in->op != svsl_ir_if || !(in->flags & svsl_ir_flag_loop_exit)) continue;
		if (e->if_has_else[k] || in->args[1] != 0) continue; // defensive: build never makes these
		int32_t brk = next_live(fn, k);
		if (brk >= n || fn->insts.items[brk].op != svsl_ir_break) continue;
		int32_t end = next_live(fn, brk);
		if (end >= n || fn->insts.items[end].op != svsl_ir_end_if) continue;
		e->loop_exit[k]   = loop_exit_branch;
		e->loop_exit[brk] = loop_exit_skip;
		e->loop_exit[end] = loop_exit_skip;
		found = true;
	}
	if (!found) return;

	// a negated condition used only by its exit branch is folded into it
	uint8_t *uses = svsl_arena_alloc(e->arena, (size_t)n);
	for (int32_t i = 0; i < n; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op == svsl_ir_nop) continue;
		uint32_t mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++)
			if ((mask & (1u << a)) && inst->args[a] < (uint32_t)n && uses[inst->args[a]] < 2)
				uses[inst->args[a]]++;
		if (svsl_ir_aux_holds_values(inst))
			for (uint32_t k = 0; k < inst->aux_count; k++) {
				uint32_t v = fn->aux.items[inst->aux + k];
				if (v < (uint32_t)n && uses[v] < 2) uses[v]++;
			}
	}
	for (int32_t k = 0; k < n; k++) {
		if (e->loop_exit[k] != loop_exit_branch) continue;
		uint32_t cond = fn->insts.items[k].args[0];
		if (fn->insts.items[cond].op == svsl_ir_log_not && uses[cond] == 1)
			e->loop_exit[cond] = loop_exit_skip;
	}
}

#include "emit_intrinsics.inc"

static void emit_body(emit_t *e, uint32_t fn_id, uint32_t void_type, uint32_t fn_type) {
	const svsl_ir_func_t *fn  = e->fn;
	svsl_spv_t           *spv = &e->spv;
	svsl_spv_stream_t    *fs  = &spv->funcs;

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
	analyze_loop_exits(e); // needs if_has_else

	uint8_t *referenced = svsl_arena_alloc(e->arena, (size_t)(fn->insts.count > 0 ? fn->insts.count : 1));
	analyze_emit_liveness(e, referenced); // constants/ptrs orphaned by chain re-inlining

	svsl_spv_inst4(spv, fs, SpvOpFunction, void_type, fn_id, SpvFunctionControlMaskNone, fn_type);
	begin_block(e, svsl_spv_id(spv));

	// all function-local variables first (SPIR-V requires them at block start)
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_var) continue;
		uint32_t ptr = spv_ptr_type(e, SpvStorageClassFunction, spv_type_for(e, inst->type));
		uint32_t var = svsl_spv_id(spv);
		svsl_spv_inst3(spv, fs, SpvOpVariable, ptr, var, SpvStorageClassFunction);
		if (inst->name.len)
			svsl_spv_inst_str(spv, &spv->debug, SpvOpName, (uint32_t[]){ var }, 1, inst->name);
		e->value_ids[i]   = var;
		e->value_class[i] = SpvStorageClassFunction;
	}

	// body (stage inputs and outputs are io-slot pointers: io_var declares them)
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_nop || op == svsl_ir_var) continue;
		if ((op == svsl_ir_const || op == svsl_ir_ptr || op == svsl_ir_undef) && !referenced[i])
			continue; // a constant/pointer left orphaned by chain re-inlining
		if (e->loop_exit[i] == loop_exit_skip) continue; // folded into a loop exit branch
		emit_inst(e, i, inst);
	}
	record_io_locations(e);

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
	e.loop_exit         = svsl_arena_alloc(arena, (size_t)inst_count);
	e.type_cap          = prog->types.types.count > 0 ? prog->types.types.count : 1;
	e.type_ids          = svsl_arena_alloc(arena, (size_t)e.type_cap * 4);
	for (int32_t l = 0; l < 4; l++)
		e.laid_ids[l] = svsl_arena_alloc(arena, (size_t)e.type_cap * 4);
	e.value_layout      = svsl_arena_alloc(arena, (size_t)inst_count);
	e.buffer_vars       = svsl_arena_alloc(arena, (size_t)(prog->buffers.count > 0 ? prog->buffers.count : 1) * 4);
	e.resource_vars     = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	e.resource_img_type = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	e.qcom_res_use      = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1));
	e.workgroup_ids     = svsl_arena_alloc(arena, (size_t)(prog->workgroup_vars.count > 0 ? prog->workgroup_vars.count : 1) * 4);
	e.const_global_ids  = svsl_arena_alloc(arena, (size_t)(prog->const_globals.count > 0 ? prog->const_globals.count : 1) * 4);
	e.private_ids       = svsl_arena_alloc(arena, (size_t)(prog->private_globals.count > 0 ? prog->private_globals.count : 1) * 4);
	e.io_vars           = svsl_arena_alloc(arena, (size_t)(fn->entry->io.count > 0 ? fn->entry->io.count : 1) * 4);
	e.builtin_input     = svsl_arena_alloc(arena, 16 * 4);
	e.spec_const_ids    = svsl_arena_alloc(arena, (size_t)(prog->spec_consts.count > 0 ? prog->spec_consts.count : 1) * 4);
	e.sampler_vars      = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1) * 4);
	e.io_locations      = svsl_arena_alloc(arena, (size_t)(fn->entry->io.count > 0 ? fn->entry->io.count : 1) * 4);

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
		// SPIR-V 1.4+ interfaces list every module-scope variable (a superset of
		// what's referenced is explicitly allowed); pre-1.4 lists Input/Output
		// only. Scan the globals stream so the 1.4 list can't drift from it.
		uint32_t *iface_items = e.interface.items;
		uint32_t  iface       = (uint32_t)e.interface.count;
		if (e.spv.version >= 0x00010400u) {
			svsl_spv_stream_t scan = {0};
			for (int32_t w = 0; w < e.spv.types.count; ) {
				uint32_t wc = e.spv.types.items[w] >> SpvWordCountShift;
				if (wc == 0) break;
				if ((e.spv.types.items[w] & SpvOpCodeMask) == SpvOpVariable)
					svsl_array_push(arena, &scan, e.spv.types.items[w + 2]);
				w += (int32_t)wc;
			}
			iface_items = scan.items;
			iface       = (uint32_t)scan.count;
		}
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
			svsl_array_push(arena, &e.spv.entries, iface_items[i]);
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
		if (fn->entry->non_coherent_tile_reads) { // [non_coherent_tile_reads_qcom]
			svsl_spv_cap(&e.spv, SpvCapabilityTileShadingQCOM);
			svsl_spv_extension(&e.spv, "SPV_QCOM_tile_shading");
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id,
			               SpvExecutionModeNonCoherentTileAttachmentReadQCOM);
		}
	}
	if (fn->entry->stage == svsl_stage_compute) {
		if (fn->entry->tile_rate[0] > 0) {
			// [tile_shading_rate_qcom(x, y, z)] REPLACES LocalSize: the
			// implementation derives the workgroup shape from the rate
			svsl_spv_cap(&e.spv, SpvCapabilityTileShadingQCOM);
			svsl_spv_extension(&e.spv, "SPV_QCOM_tile_shading");
			uint32_t rate[5] = { fn_id, SpvExecutionModeTileShadingRateQCOM,
			                     (uint32_t)fn->entry->tile_rate[0],
			                     (uint32_t)fn->entry->tile_rate[1],
			                     (uint32_t)fn->entry->tile_rate[2] };
			svsl_spv_inst(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, rate, 5);
		} else {
			uint32_t ops[5] = { fn_id, SpvExecutionModeLocalSize,
			                    (uint32_t)fn->entry->workgroup[0],
			                    (uint32_t)fn->entry->workgroup[1],
			                    (uint32_t)fn->entry->workgroup[2] };
			svsl_spv_inst(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, ops, 5);
		}
	}
	for (int32_t a = 0; a < fn->entry->func->attrs.count; a++)
		if (svsl_str_eq_cstr(fn->entry->func->attrs.items[a].name, "early_depth_stencil") ||
		    svsl_str_eq_cstr(fn->entry->func->attrs.items[a].name, "earlydepthstencil")) // HLSL spelling
			svsl_spv_inst2(&e.spv, &e.spv.exec_modes, SpvOpExecutionMode, fn_id,
			               SpvExecutionModeEarlyFragmentTests);

	out_blob->words              = svsl_spv_finalize(&e.spv, &out_blob->word_count);
	out_blob->io_locations = e.io_locations;
	out_blob->qcom_res_use = e.qcom_res_use;
	return !e.failed;
}
