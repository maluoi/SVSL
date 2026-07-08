// Layout engine: pack1/pack8/pack16 offset/size/stride computation.
// Pure functions over the type table — no state, no diagnostics; callers
// (sema) validate the results that can be user errors (explicit offsets).

#pragma once

#include "types.h"

#include <stdint.h>

typedef enum svsl_layout_ {
	svsl_layout_pack1 = 0, // scalar layout (needs the scalarBlockLayout device feature)
	svsl_layout_pack8,     // relaxed: vectors align to min(natural, 8)
	svsl_layout_pack16,    // std140
	svsl_layout_std430,    // storage-buffer/push-constant default (validates everywhere,
	                       // matches glslang's HLSL structured buffers)
} svsl_layout_;

uint32_t svsl_layout_align(const svsl_types_t *types, svsl_type_id_t id, svsl_layout_ layout);
uint32_t svsl_layout_size (const svsl_types_t *types, svsl_type_id_t id, svsl_layout_ layout);

// array element stride for elem-typed arrays (also matrix column stride source)
uint32_t svsl_layout_array_stride(const svsl_types_t *types, svsl_type_id_t elem, svsl_layout_ layout);

// Member offsets for a struct/block. Writes count offsets to out_offsets and
// returns the padded total size. Explicit member offsets ([offset(N)]) are
// honored when they don't move backwards or break alignment; the index of the
// first violating member is written to out_bad_member (-1 when all fine).
// out_offsets may be NULL when only the returned total size is wanted.
uint32_t svsl_layout_members(const svsl_types_t *types, const svsl_member_t *members,
                             int32_t count, svsl_layout_ layout,
                             uint32_t *out_offsets, int32_t *out_bad_member);
