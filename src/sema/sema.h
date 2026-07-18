// Sema: AST → typed program. This header holds the program representation that
// reflection, IR lowering, and the writers consume.

#pragma once

#include "layout.h"
#include "types.h"
#include "../front/ast.h"
#include "../front/pp.h"
#include "../diag.h"

#include <svsl/svsl.h> // svsl_stage_

#include <stdbool.h>
#include <stdint.h>

// register-space binding as written/assigned; SPIR-V descriptor offsets (+100/+200)
// are applied at emit, not here
typedef struct svsl_binding_t {
	char    cls;    // 'b','t','s','u'; 0 for the direct (binding,set) form
	int32_t slot;
	int32_t space;
	bool    direct;
} svsl_binding_t;

typedef struct svsl_buf_member_t {
	svsl_str_t     name;
	svsl_type_id_t type;
	uint32_t       offset;
	uint32_t       size;
	svsl_str_t     extra; // //-- tag or UI hint (SKS 'extra' field)
	svsl_loc_t     loc;
} svsl_buf_member_t;

typedef struct svsl_buffer_t {
	svsl_str_t       name;
	svsl_block_kind_ kind;
	svsl_layout_     layout;
	svsl_binding_t   bind;     // meaningless for pushconstant
	uint32_t         size;
	uint8_t         *defaults; // size bytes; NULL when no member has a default
	svsl_array_t(svsl_buf_member_t) members;
} svsl_buffer_t;

typedef enum svsl_res_kind_ {
	svsl_res_texture = 0,   // sampled texture (fuses with its paired sampler at emit)
	svsl_res_sampler,       // sampler without a paired texture
	svsl_res_structured,    // StructuredBuffer / readonly storagebuffer block
	svsl_res_rw_structured, // RWStructuredBuffer / writable storagebuffer block
	svsl_res_image,         // storage image (RWTexture / Image)
	svsl_res_subpass,       // input attachment
	svsl_res_tileimage,     // tile image color attachment: no descriptor binding,
	                        // subpass_index carries the attachment location
} svsl_res_kind_;

typedef struct svsl_resource_t {
	svsl_str_t     name;
	svsl_res_kind_ kind;
	svsl_type_id_t type;
	svsl_binding_t bind;
	int32_t        sampler_slot; // texture: paired sampler s-slot, -1 if unpaired
	uint32_t       element_size; // structured buffers: element stride
	uint8_t        layout;       // svsl_layout_ — object-form structured buffers: element layout
	int32_t        subpass_index;// subpass inputs: input_attachment_index (-1 = auto)
	int32_t        buffer_index; // block-form storage buffers: index into buffers (-1 = object form)
	bool           tile_attachment; // [tile_attachment]: VK_QCOM_tile_shading storage class
	svsl_str_t     value;        // //-- default texture name
	svsl_str_t     tags;         // //-- tag string
	svsl_loc_t     loc;
} svsl_resource_t;

typedef struct svsl_entry_t {
	svsl_str_t             name;
	svsl_stage_            stage;
	int32_t                workgroup[3]; // compute
	int32_t                wave_size;    // 0 = none
	int32_t                tile_rate[3]; // compute: [tile_shading_rate_qcom], 0 = none
	bool                   non_coherent_tile_reads; // pixel: [non_coherent_tile_reads_qcom]
	const svsl_ast_func_t *func;
} svsl_entry_t;

typedef struct svsl_spec_const_t {
	svsl_str_t     name;
	uint32_t       id;
	uint32_t       default_bits; // bit pattern of the default value
	svsl_type_id_t type;
	svsl_loc_t     loc;
} svsl_spec_const_t;

typedef struct svsl_vertex_input_t {
	svsl_str_t     name;
	svsl_type_id_t type;
	svsl_str_t     semantic;
} svsl_vertex_input_t;

// static/const and workgroup globals (not reflected; used by bodies and IR)
typedef struct svsl_global_t {
	svsl_str_t            name;
	svsl_type_id_t        type;
	const svsl_ast_var_t *var;     // declaration incl. initializer
	bool                  has_int; // initializer folded to an integer constant
	int64_t               int_value;
} svsl_global_t;

typedef struct svsl_func_info_t {
	const svsl_ast_func_t *func;
	svsl_type_id_t         return_type;
	svsl_type_id_t        *param_types; // param_count entries
	bool                   checked;
} svsl_func_info_t;

// A named enum type is an alias for its underlying integer type; its constants are
// plain named integers in global scope (C / HLSL-unscoped flavor).
typedef struct svsl_enum_t {
	svsl_str_t     name;
	svsl_type_id_t underlying; // an integer scalar type
} svsl_enum_t;

typedef struct svsl_enum_const_t {
	svsl_str_t     name;
	svsl_type_id_t type;  // the enum's underlying type
	int64_t        value;
} svsl_enum_const_t;

typedef struct svsl_program_t {
	svsl_types_t types;
	svsl_str_t   name;      // //--name, else source filename without extension
	bool         name_from_meta; // SKS writes an empty name unless //--name was given
	bool         porting;   // emit porting hints on legacy HLSL spellings (-Wporting)
	bool         needs_scalar_layout; // a pack1/pack8 layout straddles: scalarBlockLayout feature
	int32_t      wave_size; // //--wave_size or [wave_size(N)], 0 = none
	int32_t      tile_apron[2]; // //--apron = W[, H] — VK_QCOM_tile_shading render pass
	                            // tileApronSize, applied by the renderer; (0,0) = none

	svsl_array_t(svsl_buffer_t)       buffers;
	svsl_array_t(svsl_resource_t)     resources;
	svsl_array_t(svsl_entry_t)        entries;
	svsl_array_t(svsl_spec_const_t)   spec_consts;
	svsl_array_t(svsl_vertex_input_t) vertex_inputs;
	svsl_array_t(svsl_global_t)       const_globals;
	svsl_array_t(svsl_global_t)       workgroup_vars;
	svsl_array_t(svsl_enum_t)         enums;
	svsl_array_t(svsl_enum_const_t)   enum_consts;
	svsl_array_t(svsl_func_info_t)    functions;

	const svsl_ast_t *ast;
} svsl_program_t;

// the checked signature (return + param types) for a function, or NULL if absent
const svsl_func_info_t *svsl_program_func_info(const svsl_program_t *prog,
                                               const svsl_ast_func_t *func);

typedef struct svsl_sema_options_t {
	const char *entry_vs, *entry_ps, *entry_cs; // NULL → "vs"/"ps"/"cs"
	bool        half_strict16;                  // --half=strict16: half means float16
	bool        porting_hints;                  // -Wporting: hint on legacy HLSL spellings
} svsl_sema_options_t;

// Attributes are honored or rejected, never silently dropped: known attributes
// used out of context warn, unknown [[vk::*]] attributes are errors (spec §6),
// anything else unknown warns. Shared by decl sema and the statement checker.
typedef enum svsl_attr_ctx_ {
	svsl_attr_ctx_func   = 1 << 0,
	svsl_attr_ctx_var    = 1 << 1, // global variables / resources
	svsl_attr_ctx_member = 1 << 2, // struct + buffer block members
	svsl_attr_ctx_stmt   = 1 << 3,
	svsl_attr_ctx_block  = 1 << 4, // cbuffer/storagebuffer/pushconstant blocks
} svsl_attr_ctx_;

struct svsl_ast_attrs_t;
void svsl_attrs_check(svsl_arena_t *arena, svsl_diag_list_t *diags,
                      const struct svsl_ast_attrs_t *attrs, svsl_attr_ctx_ ctx, bool porting);

// Declaration-level sema: types, buffers, resources, bindings, entries,
// $Globals with folded defaults, //-- metadata merge, vertex inputs.
// Returns false if any error diagnostic was emitted.
bool svsl_sema_run(svsl_arena_t *arena, const svsl_ast_t *ast, const svsl_pp_result_t *pp,
                   const char *opt_filename, const svsl_sema_options_t *opt_options,
                   svsl_program_t *out_program, svsl_diag_list_t *ref_diags);
