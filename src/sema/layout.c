// pack16 is proper std140. Note: glslang's HLSL front-end (skshaderc) packs
// cbuffers with HLSL rules instead — 4-byte alignment with a 16-byte-straddle
// bump, and no array tail padding. The two agree on every StereoKit corpus
// shader (scalars, float3+float pairs, and 16-byte-multiple arrays lay out
// identically); they differ only for vec2-after-scalar patterns, where the
// glslang variant can even fail spirv-val. SVSL emits the validating layout.

#include "layout.h"

static uint32_t align_up(uint32_t value, uint32_t align) {
	return (value + align - 1) & ~(align - 1);
}

static uint32_t vector_natural_align(svsl_scalar_ scalar, int32_t count) {
	uint32_t n = (uint32_t)svsl_scalar_size(scalar);
	return count == 2 ? 2 * n : 4 * n; // vec3 aligns like vec4
}

// matrices are arrays of column vectors: column count = cols, column type = rows-vector
static uint32_t matrix_col_stride(const svsl_type_t *t, svsl_layout_ layout) {
	uint32_t n        = (uint32_t)svsl_scalar_size(t->scalar);
	uint32_t col_size = (uint32_t)t->rows * n;
	switch (layout) {
	case svsl_layout_pack1:  return align_up(col_size, n);
	case svsl_layout_pack8:  return align_up(col_size, vector_natural_align(t->scalar, t->rows) < 8 ? vector_natural_align(t->scalar, t->rows) : 8);
	case svsl_layout_std430: return align_up(col_size, vector_natural_align(t->scalar, t->rows));
	case svsl_layout_pack16:
	default:                 return align_up(col_size, 16);
	}
}

uint32_t svsl_layout_align(const svsl_types_t *types, svsl_type_id_t id, svsl_layout_ layout) {
	if (id == SVSL_TYPE_NONE) return 1; // unresolved member type, already errored
	const svsl_type_t *t = svsl_type_get(types, id);
	switch (t->kind) {
	case svsl_type_scalar:
		return (uint32_t)svsl_scalar_size(t->scalar);
	case svsl_type_vector: {
		uint32_t natural = vector_natural_align(t->scalar, t->count);
		switch (layout) {
		case svsl_layout_pack1:  return (uint32_t)svsl_scalar_size(t->scalar);
		case svsl_layout_pack8:  return natural < 8 ? natural : 8;
		case svsl_layout_pack16:
		case svsl_layout_std430: // std430 keeps natural vector alignment
		default:                 return natural;
		}
	}
	case svsl_type_matrix:
		switch (layout) {
		case svsl_layout_pack1: return (uint32_t)svsl_scalar_size(t->scalar);
		case svsl_layout_pack8: {
			uint32_t natural = vector_natural_align(t->scalar, t->rows);
			return natural < 8 ? natural : 8;
		}
		case svsl_layout_pack16:
		case svsl_layout_std430:
		default:                return matrix_col_stride(t, layout); // stride-aligned
		}
	case svsl_type_array: {
		uint32_t elem_align = svsl_layout_align(types, t->elem, layout);
		return layout == svsl_layout_pack16 ? (elem_align < 16 ? 16 : elem_align) : elem_align;
	}
	case svsl_type_struct: {
		const svsl_struct_info_t *info  = &types->structs.items[t->struct_index];
		uint32_t                  align = 1;
		for (int32_t i = 0; i < info->members.count; i++) {
			uint32_t a = svsl_layout_align(types, info->members.items[i].type, layout);
			if (a > align) align = a;
		}
		return layout == svsl_layout_pack16 ? (align < 16 ? 16 : align) : align;
	}
	case svsl_type_void:
	case svsl_type_texture:
	case svsl_type_image:
	case svsl_type_sampler:
	case svsl_type_subpass:
	case svsl_type_buffer:
	case svsl_type_tileimage:
		return 1; // opaque types have no buffer layout
	}
	return 1;
}

uint32_t svsl_layout_array_stride(const svsl_types_t *types, svsl_type_id_t elem, svsl_layout_ layout) {
	uint32_t elem_size  = svsl_layout_size(types, elem, layout);
	uint32_t elem_align = svsl_layout_align(types, elem, layout);
	if (layout == svsl_layout_pack16 && elem_align < 16) elem_align = 16;
	return align_up(elem_size, elem_align);
}

uint32_t svsl_layout_size(const svsl_types_t *types, svsl_type_id_t id, svsl_layout_ layout) {
	if (id == SVSL_TYPE_NONE) return 0; // unresolved member type, already errored
	const svsl_type_t *t = svsl_type_get(types, id);
	switch (t->kind) {
	case svsl_type_scalar:
		return (uint32_t)svsl_scalar_size(t->scalar);
	case svsl_type_vector:
		return (uint32_t)t->count * (uint32_t)svsl_scalar_size(t->scalar);
	case svsl_type_matrix:
		return (uint32_t)t->cols * matrix_col_stride(t, layout);
	case svsl_type_array:
		if (t->array_count == 0) return 0; // runtime-sized
		return (uint32_t)t->array_count * svsl_layout_array_stride(types, t->elem, layout);
	case svsl_type_struct: {
		const svsl_struct_info_t *info = &types->structs.items[t->struct_index];
		int32_t bad = -1;
		return svsl_layout_members(types, info->members.items, info->members.count,
		                           layout, NULL, &bad); // size only — no per-member offsets needed
	}
	case svsl_type_void:
	case svsl_type_texture:
	case svsl_type_image:
	case svsl_type_sampler:
	case svsl_type_subpass:
	case svsl_type_buffer:
	case svsl_type_tileimage:
		return 0; // opaque types have no buffer size
	}
	return 0;
}

uint32_t svsl_layout_members(const svsl_types_t *types, const svsl_member_t *members,
                             int32_t count, svsl_layout_ layout,
                             uint32_t *out_offsets, int32_t *out_bad_member) {
	*out_bad_member = -1;
	uint32_t cursor    = 0;
	uint32_t max_align = 1;

	for (int32_t i = 0; i < count; i++) {
		uint32_t align = svsl_layout_align(types, members[i].type, layout);
		uint32_t size  = svsl_layout_size (types, members[i].type, layout);
		if (align > max_align) max_align = align;

		uint32_t offset = align_up(cursor, align);
		if (members[i].explicit_offset >= 0) {
			uint32_t explicit = (uint32_t)members[i].explicit_offset;
			// may only move forward, and must stay aligned
			if (explicit < offset || (explicit % align) != 0) {
				if (*out_bad_member < 0) *out_bad_member = i;
			} else {
				offset = explicit;
			}
		}
		if (out_offsets) out_offsets[i] = offset; // NULL when only the total size is wanted
		cursor = offset + size;
	}
	if (layout == svsl_layout_pack16 && max_align < 16) max_align = 16;
	return align_up(cursor, max_align);
}
