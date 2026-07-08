// Type system: interned types in a flat table. Type ids are indices, so
// deduplicated types compare with ==.

#pragma once

#include "../front/ast.h"
#include "../util/arena.h"
#include "../util/array.h"
#include "../util/str.h"

#include <stdbool.h>
#include <stdint.h>

typedef int32_t svsl_type_id_t;
#define SVSL_TYPE_NONE ((svsl_type_id_t)-1)

typedef enum svsl_scalar_ {
	svsl_scalar_bool = 0,
	svsl_scalar_int8,  svsl_scalar_int16,  svsl_scalar_int32,  svsl_scalar_int64,
	svsl_scalar_uint8, svsl_scalar_uint16, svsl_scalar_uint32, svsl_scalar_uint64,
	svsl_scalar_float16, svsl_scalar_float32, svsl_scalar_float64,
	svsl_scalar_half, // "at least 16-bit": RelaxedPrecision float32, 4 bytes in buffers
} svsl_scalar_;

typedef enum svsl_type_kind_ {
	svsl_type_void = 0,
	svsl_type_scalar,
	svsl_type_vector,   // scalar + count
	svsl_type_matrix,   // scalar + rows x cols, column-major
	svsl_type_array,    // elem + count (count 0 = runtime-sized)
	svsl_type_struct,   // index into struct table
	svsl_type_texture,  // sampled image: dim + arrayed + elem
	svsl_type_image,    // storage image: dim + arrayed + elem + format
	svsl_type_sampler,  // is_comparison
	svsl_type_subpass,  // elem (+ input_attachment_index at the declaration)
	svsl_type_buffer,   // StructuredBuffer-style resource handle: elem + is_rw
	svsl_type_tileimage,// SPV_EXT_shader_tile_image color attachment: elem, no descriptor
} svsl_type_kind_;

typedef enum svsl_texdim_ {
	svsl_texdim_1d = 0, svsl_texdim_2d, svsl_texdim_3d, svsl_texdim_cube,
} svsl_texdim_;

typedef struct svsl_type_t {
	svsl_type_kind_ kind;
	svsl_scalar_    scalar;        // scalar/vector/matrix
	uint8_t         count;         // vector components
	uint8_t         rows, cols;    // matrix
	svsl_type_id_t  elem;          // array/texture/image/subpass/buffer element
	int32_t         array_count;   // array (0 = runtime-sized)
	int32_t         struct_index;  // struct
	svsl_texdim_    dim;           // texture/image
	bool            arrayed;       // texture/image array-ness (Texture2DArray)
	bool            multisampled;  // Texture2DMS / SubpassInputMS / TileImage MS
	bool            is_comparison; // sampler
	bool            is_rw;         // buffer/image writability
	svsl_str_t      format;        // image format name ("rgba8", ...), empty = unknown
} svsl_type_t;

typedef struct svsl_member_t {
	svsl_str_t     name;
	svsl_type_id_t type;
	svsl_str_t     semantic;          // stage-IO structs
	uint8_t        interp;            // svsl_interp_
	int32_t        explicit_offset;   // [offset(N)], -1 if none
	int32_t        explicit_location; // [location(N)] stage IO, -1 if none
	svsl_loc_t     loc;
} svsl_member_t;

// a packed struct's logical bit fields; the `members` array meanwhile holds the
// physical backing (uint32 × backing_words), so all layout/emit code sees an
// ordinary struct and only field *resolution* consults this.
typedef struct svsl_field_t {
	svsl_str_t     name;
	svsl_type_id_t type;       // resolve type — what `p.field` reads as / is assigned
	int16_t        bit_offset; // LSB position within the backing words
	uint8_t        bit_width;
	uint8_t        bit_format; // svsl_bitfmt_
	svsl_loc_t     loc;
} svsl_field_t;

typedef struct svsl_struct_info_t {
	svsl_str_t name;
	svsl_array_t(svsl_member_t) members;      // packed: the backing uint32 words
	bool        packed;
	int32_t     backing_words;
	svsl_array_t(svsl_field_t) fields;        // packed: the logical bit fields
} svsl_struct_info_t;

typedef struct svsl_types_t {
	svsl_arena_t *arena;
	svsl_array_t(svsl_type_t)        types;
	svsl_array_t(svsl_struct_info_t) structs;
	bool          half_strict16; // --half=strict16: every `half` interns as float16
} svsl_types_t;

// interning — returns existing id when an equal type is already present
svsl_type_id_t svsl_type_intern(svsl_types_t *types, svsl_type_t type);
svsl_type_id_t svsl_type_scalar_id(svsl_types_t *types, svsl_scalar_ scalar);
svsl_type_id_t svsl_type_vector_id(svsl_types_t *types, svsl_scalar_ scalar, int32_t count);
svsl_type_id_t svsl_type_matrix_id(svsl_types_t *types, svsl_scalar_ scalar, int32_t rows, int32_t cols);
svsl_type_id_t svsl_type_array_id (svsl_types_t *types, svsl_type_id_t elem, int32_t count);

const svsl_type_t *svsl_type_get(const svsl_types_t *types, svsl_type_id_t id);

// IEEE 754 binary16 bit pattern for a float (round-to-nearest-even)
uint32_t svsl_f32_to_f16_bits(float f);

// scalar/vector/matrix name → parts: "float4x4" → (float32, 4, 4); "half3" → (half, 3)
// rows=cols=0 for scalars; cols=0 for vectors (count in *out_rows)
bool svsl_scalar_name_parse(svsl_str_t name, svsl_scalar_ *out_scalar,
                            int32_t *out_rows, int32_t *out_cols);

int32_t     svsl_scalar_size(svsl_scalar_ scalar); // bytes in buffers (half = 4)
const char *svsl_type_name  (const svsl_types_t *types, svsl_type_id_t id); // arena-owned, for diagnostics

// texture/sampler/image/buffer/subpass/tileimage — types that must bind as resources
bool svsl_type_is_resource(const svsl_type_t *t);
