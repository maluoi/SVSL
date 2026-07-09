// SVSL — SPIR-V Shading Language
// Public API. This is the only header a consumer includes.
//
// The whole surface is here: compile SVSL source to SPIR-V and StereoKit's SKS
// container, and parse SKS containers back into a structured view. The core links
// nothing but libc; this header pulls in no internal types. All memory for a
// compile (or a parse) lives in one arena owned by the returned handle and is
// released with a single svsl_result_free / svsl_sks_free.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SVSL_VERSION_MAJOR 0
#define SVSL_VERSION_MINOR 1
#define SVSL_VERSION_PATCH 0

// The SKS container version this build reads and writes. One version at a time:
// the field lets runtimes refuse foreign files, it is not a compatibility knob.
#define SVSL_SKS_VERSION 10

// ============================================================================
// Diagnostics
// ============================================================================

typedef enum svsl_severity_ {
	svsl_severity_error,
	svsl_severity_warning,
	svsl_severity_porting, // legacy-HLSL spelling with a native alternative
	svsl_severity_info,
} svsl_severity_;

typedef struct svsl_loc_t {
	const char *file; // NULL if unknown; owned by the result
	int32_t     line; // 1-based
	int32_t     col;  // 1-based, 0 if unknown
} svsl_loc_t;

typedef struct svsl_diag_t {
	svsl_severity_ severity;
	svsl_loc_t     loc;
	const char    *message; // owned by the result
} svsl_diag_t;

// ============================================================================
// Shader stages
// ============================================================================

typedef enum svsl_stage_ { // bit flags; matches StereoKit's skr_stage_
	svsl_stage_vertex  = 1 << 0,
	svsl_stage_pixel   = 1 << 1,
	svsl_stage_compute = 1 << 2,
} svsl_stage_;

// ============================================================================
// Compilation input
// ============================================================================

typedef struct svsl_define_t {
	const char *name;
	const char *value; // NULL = defined as 1
} svsl_define_t;

// Result of an include callback. `content` may be freed by the caller once
// svsl_compile returns — the compiler copies everything it keeps.
typedef struct svsl_include_src_t {
	const char *content; // NULL = not found
	int32_t     length;  // -1 = use strlen(content)
	const char *path;    // canonical path for diagnostics; NULL = the requested path
} svsl_include_src_t;

typedef svsl_include_src_t (*svsl_include_cb_t)(void *user, const char *path, const char *requester);

typedef enum svsl_opt_level_ {
	svsl_opt_none = 0,   // -O0: dead-code elimination only (structured-CF hygiene)
	svsl_opt_default,    // -O1: value-preserving passes + integer identities (default)
	svsl_opt_aggressive, // -O2: adds float-algebraic identities
} svsl_opt_level_;

typedef struct svsl_source_t {
	const char *text;     // the SVSL source
	int32_t     length;   // 0 or -1 = use strlen(text)
	const char *filename; // shown in diagnostics; NULL is allowed
} svsl_source_t;

typedef struct svsl_options_t {
	// preprocessor
	const svsl_define_t *defines;
	int32_t              define_count;
	svsl_include_cb_t    include_cb;   // resolves #include; NULL = includes error
	void                *include_user; // passed back to include_cb

	// entry-point names (NULL falls back to "vs" / "ps" / "cs")
	const char *entry_vertex;
	const char *entry_pixel;
	const char *entry_compute;

	// code generation
	svsl_opt_level_ opt_level;     // 0 == svsl_opt_default
	bool            half_strict16; // treat every `half` as an exact 16-bit float

	// diagnostics
	bool            porting_hints; // emit `porting` hints on legacy HLSL spellings (default off)
} svsl_options_t;

// ============================================================================
// Compilation result
// ============================================================================

typedef struct svsl_bytes_t {
	const uint8_t *data;
	int32_t        size;
} svsl_bytes_t;

// One compiled entry point.
typedef struct svsl_stage_output_t {
	svsl_stage_     stage;
	const char     *entry;            // entry-point name
	const uint32_t *spirv;            // SPIR-V module (words)
	int32_t         spirv_word_count;
} svsl_stage_output_t;

// Returned by svsl_compile. Transparent for the fields you read; `_impl` owns all
// the memory the pointers refer to. Free exactly once with svsl_result_free.
typedef struct svsl_result_t {
	bool                       ok;               // no error diagnostics
	int32_t                    error_count;
	int32_t                    warning_count;
	int32_t                    diagnostic_count;
	const svsl_diag_t         *diagnostics;
	int32_t                    stage_count;      // 0 when compilation failed
	const svsl_stage_output_t *stages;
	void                      *_impl;            // internal; do not touch
} svsl_result_t;

// Compiles source to SPIR-V. Never returns NULL except on allocation failure;
// inspect `.ok` and `.diagnostics`. Free with svsl_result_free.
svsl_result_t *svsl_compile(const svsl_source_t *source, const svsl_options_t *opt_options);

// Frees the result and everything reachable from it.
void svsl_result_free(svsl_result_t *result);

// --- container / text products (valid until svsl_result_free) --------------
// All return empty / NULL if the compile failed. Bytes and strings are owned by
// the result's arena — copy them if you need to outlive it.

svsl_bytes_t svsl_result_sks       (svsl_result_t *result);              // SKS container
const char  *svsl_result_header    (svsl_result_t *result, const char *name); // embeddable C header
const char  *svsl_result_reflection(svsl_result_t *result);             // human-readable table
const char  *svsl_result_ir        (svsl_result_t *result);             // IR dump (debugging)

// True when a pack1/pack8 buffer layout breaks core relaxed block layout rules:
// the device needs VK_EXT_scalar_block_layout, and spirv-val needs
// --scalar-block-layout. Also recorded as bit 16 of the SKS feature mask.
bool svsl_result_needs_scalar_layout(svsl_result_t *result);

// ============================================================================
// SKS container reading
// ============================================================================
// A structured, read-only view of a StereoKit SKS file. Numeric enum fields
// (register_type, var type, vertex format, semantic) carry StereoKit's skr_*
// values; the svsl_sks_*_ enums below name the ones worth interpreting.

typedef enum svsl_sks_register_ { // skr_register_
	svsl_sks_register_default = 0,
	svsl_sks_register_vertex,
	svsl_sks_register_index,
	svsl_sks_register_constant,        // uniform / constant buffer
	svsl_sks_register_texture,         // sampled texture or sampler
	svsl_sks_register_read_buffer,     // StructuredBuffer
	svsl_sks_register_readwrite,       // RWStructuredBuffer
	svsl_sks_register_readwrite_tex,   // storage image
	svsl_sks_register_input_attachment,
} svsl_sks_register_;

typedef enum svsl_sks_vartype_ { // sksc_shader_var_
	svsl_sks_vartype_none = 0, // struct-typed; type_name carries the shape
	svsl_sks_vartype_int,
	svsl_sks_vartype_uint,
	svsl_sks_vartype_uint8,
	svsl_sks_vartype_float,
	svsl_sks_vartype_double,
} svsl_sks_vartype_;

typedef struct svsl_sks_var_t {
	const char *name;
	const char *extra;      // //-- tag or UI hint
	const char *type_name;  // "float4x4", struct name, ...
	uint32_t    offset;
	uint32_t    size;
	uint16_t    type;       // svsl_sks_vartype_
	uint16_t    type_count; // component/element count
} svsl_sks_var_t;

typedef struct svsl_sks_buffer_t {
	const char           *name;
	uint16_t              slot;
	uint8_t               space;
	uint8_t               stage_bits;    // svsl_stage_ mask
	uint8_t               register_type; // svsl_sks_register_
	uint32_t              size;
	const uint8_t        *defaults;      // NULL when no member has a default
	uint32_t              defaults_size;
	const svsl_sks_var_t *vars;
	int32_t               var_count;
} svsl_sks_buffer_t;

typedef struct svsl_sks_resource_t {
	const char *name;
	const char *value;         // //-- default (e.g. texture name)
	const char *tags;
	uint16_t    slot;
	uint8_t     stage_bits;    // svsl_stage_ mask
	uint8_t     register_type; // svsl_sks_register_
	uint32_t    element_size;  // structured buffers: element stride
	uint8_t     shape;         // bits: 0-2 dim, 3 arrayed, 4 multisampled, 5 comparison
	uint8_t     image_format;  // SpvImageFormat for storage images, else 0
} svsl_sks_resource_t;

typedef struct svsl_sks_vertex_input_t {
	int32_t format;   // skr_vertex_fmt_
	uint8_t count;
	int32_t semantic; // skr_semantic_
	uint8_t slot;
	uint8_t location; // SPIR-V input location (first of the span for arrays/matrices)
} svsl_sks_vertex_input_t;

typedef struct svsl_sks_spec_t {
	const char *name;
	uint32_t    id;
	uint32_t    default_bits; // bit pattern of the default value
	uint16_t    type;         // svsl_sks_vartype_
	uint8_t     stage_bits;   // svsl_stage_ mask
} svsl_sks_spec_t;

typedef struct svsl_sks_stage_t {
	svsl_stage_     stage;
	uint32_t        wave_size;        // 0 = unspecified
	const uint32_t *spirv;            // SPIR-V module (words)
	int32_t         spirv_word_count;
} svsl_sks_stage_t;

typedef struct svsl_sks_file_t {
	uint16_t                       version;
	const char                    *name;      // "" when the shader was unnamed
	uint64_t                       features;  // device-feature mask
	uint32_t                       wave_size;

	const svsl_sks_buffer_t       *buffers;        int32_t buffer_count;
	const svsl_sks_vertex_input_t *vertex_inputs;  int32_t vertex_input_count;
	const svsl_sks_resource_t     *resources;      int32_t resource_count;
	const svsl_sks_spec_t         *spec_consts;    int32_t spec_count;
	const svsl_sks_stage_t        *stages;         int32_t stage_count;

	void *_impl; // internal; do not touch
} svsl_sks_file_t;

// Parses an SKS container. Returns NULL if the bytes are not a readable SKS file
// of version SVSL_SKS_VERSION (or on allocation failure). Free with svsl_sks_free.
svsl_sks_file_t *svsl_sks_parse(const void *bytes, int32_t size);

// Frees the parsed file and everything reachable from it.
void svsl_sks_free(svsl_sks_file_t *file);

#ifdef __cplusplus
}
#endif
