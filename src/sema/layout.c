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

// Vulkan's relaxed block layout (core 1.1, no device feature) is std430 with one
// relaxation: vector members may sit at component alignment as long as they don't
// improperly straddle a 16-byte boundary. Everything else — composite member
// offsets, array strides, matrix column strides — must still satisfy std430
// alignment. A concrete layout violating that needs the scalarBlockLayout device
// feature. (SVSL layouts always keep members component-aligned, so the relaxation
// baseline itself is never violated.)
static bool vector_straddles(uint32_t offset, uint32_t size) {
	if (size > 16) return (offset % 16) != 0;
	return offset / 16 != (offset + size - 1) / 16;
}

// A member's offset given the running cursor and its alignment: the next aligned
// slot, unless a valid explicit [offset] overrides it (must move forward and stay
// aligned). out_ok, when given, reports whether an explicit offset was rejected.
static uint32_t member_offset(uint32_t cursor, uint32_t align, int32_t explicit_offset, bool *out_ok) {
	uint32_t offset = align_up(cursor, align);
	if (out_ok) *out_ok = true;
	if (explicit_offset < 0) return offset;
	uint32_t e = (uint32_t)explicit_offset;
	if (e >= offset && (e % align) == 0) return e;
	if (out_ok) *out_ok = false;
	return offset;
}

bool svsl_layout_needs_scalar(const svsl_types_t *types, svsl_type_id_t id,
                              svsl_layout_ layout, uint32_t base, uint32_t *out_offset) {
	if (id == SVSL_TYPE_NONE) return false;
	const svsl_type_t *t = svsl_type_get(types, id);

	// composites themselves must sit at their std430 alignment
	if (t->kind == svsl_type_matrix || t->kind == svsl_type_array || t->kind == svsl_type_struct) {
		if (base % svsl_layout_align(types, id, svsl_layout_std430) != 0) {
			if (out_offset) *out_offset = base;
			return true;
		}
	}

	switch (t->kind) {
	case svsl_type_vector:
		if (!vector_straddles(base, (uint32_t)t->count * (uint32_t)svsl_scalar_size(t->scalar)))
			return false;
		if (out_offset) *out_offset = base;
		return true;
	case svsl_type_matrix: {
		uint32_t col_size   = (uint32_t)t->rows * (uint32_t)svsl_scalar_size(t->scalar);
		uint32_t col_stride = matrix_col_stride(t, layout);
		uint32_t col_align  = vector_natural_align(t->scalar, t->rows); // std430 column alignment
		for (int32_t c = 0; c < t->cols; c++) {
			uint32_t at = base + (uint32_t)c * col_stride;
			if (at % col_align != 0 || vector_straddles(at, col_size)) {
				if (out_offset) *out_offset = at;
				return true;
			}
		}
		return false;
	}
	case svsl_type_array: {
		uint32_t stride = svsl_layout_array_stride(types, t->elem, layout);
		if (stride % svsl_layout_align(types, t->elem, svsl_layout_std430) != 0) {
			if (out_offset) *out_offset = base;
			return true;
		}
		// offsets repeat modulo 16 after at most 16 elements (strides are 4-aligned)
		int32_t count = t->array_count > 0 && t->array_count < 16 ? t->array_count : 16;
		for (int32_t i = 0; i < count; i++)
			if (svsl_layout_needs_scalar(types, t->elem, layout, base + (uint32_t)i * stride, out_offset))
				return true;
		return false;
	}
	case svsl_type_struct: {
		const svsl_struct_info_t *info   = &types->structs.items[t->struct_index];
		uint32_t                  cursor = 0;
		for (int32_t i = 0; i < info->members.count; i++) {
			const svsl_member_t *m      = &info->members.items[i];
			uint32_t             align  = svsl_layout_align(types, m->type, layout);
			uint32_t             offset = member_offset(cursor, align, m->explicit_offset, NULL);
			if (svsl_layout_needs_scalar(types, m->type, layout, base + offset, out_offset))
				return true;
			cursor = offset + svsl_layout_size(types, m->type, layout);
		}
		return false;
	}
	default:
		return false;
	}
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

		bool     ok;
		uint32_t offset = member_offset(cursor, align, members[i].explicit_offset, &ok);
		if (!ok && *out_bad_member < 0) *out_bad_member = i; // explicit moved back or misaligned
		if (out_offsets) out_offsets[i] = offset; // NULL when only the total size is wanted
		cursor = offset + size;
	}
	if (layout == svsl_layout_pack16 && max_align < 16) max_align = 16;
	return align_up(cursor, max_align);
}
