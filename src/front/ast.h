// AST: tagged unions, arena-allocated, immutable after parse.
// Types are unresolved name references at this stage — sema resolves them.

#pragma once

#include "lexer.h"
#include "../diag.h"
#include "../util/str.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct svsl_ast_type_t svsl_ast_type_t;
typedef struct svsl_ast_expr_t svsl_ast_expr_t;
typedef struct svsl_ast_stmt_t svsl_ast_stmt_t;
typedef struct svsl_ast_var_t  svsl_ast_var_t;
typedef struct svsl_ast_enum_t svsl_ast_enum_t;

// --- attributes: [unroll(4)], [numthreads(8,8,1)], [[vk::binding(0,1)]] -------

typedef struct svsl_ast_attr_t {
	svsl_str_t        name;     // "unroll", "numthreads", "vk::binding", ...
	bool              double_bracket; // [[...]] spelling
	svsl_ast_expr_t **args;
	int32_t           arg_count;
	svsl_loc_t        loc;
} svsl_ast_attr_t;

typedef struct svsl_ast_attrs_t {
	svsl_ast_attr_t *items;
	int32_t          count;
} svsl_ast_attrs_t;

// --- register() bindings -------------------------------------------------------

typedef struct svsl_ast_reg_t {
	bool       present;
	bool       direct;  // register(binding, set) — SVSL native, no prefix letter/offset
	char       cls;     // 'b','t','s','u' when prefixed, 0 otherwise
	int32_t    slot;
	int32_t    space;
	svsl_loc_t loc;
} svsl_ast_reg_t;

// --- type references ------------------------------------------------------------

struct svsl_ast_type_t {
	svsl_str_t        name;         // "float4", "Texture2D", "inst_t", "void" (enum name if inline_enum)
	svsl_loc_t        loc;
	svsl_ast_type_t  *elem;         // Texture2D<float4> / StructuredBuffer<inst_t> element
	svsl_str_t        format;       // Image2D<float4, rgba8> — image format name
	svsl_ast_expr_t  *index;        // SubpassInput<T, N> — attachment index expression
	svsl_ast_enum_t  *inline_enum;  // 'enum {...}' written in this type position; resolves to its int
	svsl_ast_expr_t **array_dims;   // outer-to-inner; NULL entry = runtime-sized []
	int32_t           array_dim_count;
};

// --- expressions -----------------------------------------------------------------

typedef enum svsl_expr_ {
	svsl_expr_int_lit,
	svsl_expr_float_lit,
	svsl_expr_bool_lit,
	svsl_expr_string_lit, // attribute arguments only
	svsl_expr_ident,
	svsl_expr_binary,     // also assignments and comma
	svsl_expr_unary,      // prefix: - ! ~ + ++ --
	svsl_expr_post,       // postfix: ++ --
	svsl_expr_ternary,
	svsl_expr_call,       // callee(args) — callee is ident (function/intrinsic) or member (method)
	svsl_expr_ctor,       // float4(...), float4x4(1) — type name used as function
	svsl_expr_cast,       // (float3x3)m
	svsl_expr_member,     // expr.name — member, swizzle, or method name before a call
	svsl_expr_index,      // expr[expr]
	svsl_expr_init_list,  // {1, 1, 1, 1} — initializers only
	svsl_expr_spirv_asm,  // spirv_asm(T) { OpXxx ... ; ... } — inline SPIR-V
} svsl_expr_;

// --- inline SPIR-V (spirv_asm) --------------------------------------------------
// A spirv_asm block is a sequence of raw SPIR-V instructions written in binary
// operand order. Result ids are named locals (%name); the distinguished %result
// local carries the block's value (typed by the parenthesised result type).

typedef enum svsl_spv_operand_ {
	svsl_spv_operand_local,   // %name        — an id defined/used within the block
	svsl_spv_operand_value,   // $expr        — an SVSL value; its SPIR-V id is spliced in
	svsl_spv_operand_type,    // $$type       — an SVSL type; its SPIR-V type id is spliced in
	svsl_spv_operand_literal, // 42           — a literal 32-bit word (enum values, immediates)
	svsl_spv_operand_glsl450, // glsl450      — the GLSL.std.450 ext-instruction import id
} svsl_spv_operand_;

typedef struct svsl_ast_spv_operand_t {
	uint8_t          kind;    // svsl_spv_operand_
	svsl_str_t       local;   // svsl_spv_operand_local
	svsl_ast_expr_t *value;   // svsl_spv_operand_value
	svsl_ast_type_t *type;    // svsl_spv_operand_type
	int32_t          type_id; // svsl_type_id_t, resolved by sema (svsl_spv_operand_type)
	uint32_t         literal; // svsl_spv_operand_literal
	svsl_loc_t       loc;
} svsl_ast_spv_operand_t;

typedef struct svsl_ast_spv_inst_t {
	svsl_str_t              opcode;  // "OpFAdd"
	int32_t                 spv_op;  // SpvOp, resolved by sema; -1 until then / on error
	svsl_ast_spv_operand_t *operands;
	int32_t                 operand_count;
	svsl_loc_t              loc;
} svsl_ast_spv_inst_t;

// filled by sema: what an ident/call/member resolved to (IR lowering consumes this)
typedef enum svsl_ref_ {
	svsl_ref_none = 0,
	svsl_ref_local,         // a = scope-local index
	svsl_ref_param,         // a = param index (within current function)
	svsl_ref_buffer_member, // a = buffer index, b = member index
	svsl_ref_resource,      // a = resource index
	svsl_ref_spec_const,    // a = spec constant index
	svsl_ref_const_global,  // a = const-global index
	svsl_ref_workgroup,     // a = workgroup-global index
	svsl_ref_function,      // a = AST decl index
	svsl_ref_intrinsic,     // a = intrinsic table index
	svsl_ref_method,        // a = method table index
	svsl_ref_builtin_var,   // a = builtin variable id (subgroup_size, ...)
	svsl_ref_swizzle,       // member: a = packed component indices, b = count
	svsl_ref_matrix_elem,   // member: a = row, b = col
	svsl_ref_bitfield,      // member: a = struct index, b = packed-field index
	svsl_ref_enum_const,    // a = enum-constant index (a named integer constant)
} svsl_ref_;

typedef struct svsl_sema_ref_t {
	uint8_t kind; // svsl_ref_
	int32_t a, b;
} svsl_sema_ref_t;

// packed-struct bit-field storage format (the `: fmt` after a member)
typedef enum svsl_bitfmt_ {
	svsl_bitfmt_raw = 0, // integer / bool / raw 16-bit half
	svsl_bitfmt_unorm,   // [0,1] normalized  (: unN)
	svsl_bitfmt_snorm,   // [-1,1] normalized (: snN)
} svsl_bitfmt_;

struct svsl_ast_expr_t {
	svsl_expr_ kind;
	svsl_loc_t loc;

	// sema annotations (0/NONE until sema runs; the parse shape stays immutable)
	int32_t         sema_type; // svsl_type_id_t; -1 = unset/error
	svsl_sema_ref_t sema_ref;
	union {
		struct { uint64_t value; uint8_t suffix; }                        int_lit;
		struct { double value; uint8_t suffix; }                          float_lit;
		bool                                                              bool_lit;
		svsl_str_t                                                        string_lit;
		svsl_str_t                                                        ident;
		struct { svsl_tok_ op; svsl_ast_expr_t *lhs, *rhs; }              binary;
		struct { svsl_tok_ op; svsl_ast_expr_t *operand; }                unary;
		struct { svsl_tok_ op; svsl_ast_expr_t *operand; }                post;
		struct { svsl_ast_expr_t *cond, *then_expr, *else_expr; }         ternary;
		struct { svsl_ast_expr_t *callee; svsl_ast_expr_t **args; int32_t arg_count; } call;
		struct { svsl_ast_type_t *type; svsl_ast_expr_t **args; int32_t arg_count; }   ctor;
		struct { svsl_ast_type_t *type; svsl_ast_expr_t *operand; }       cast;
		struct { svsl_ast_expr_t *object; svsl_str_t name; }              member;
		struct { svsl_ast_expr_t *object; svsl_ast_expr_t *index; }       index;
		struct { svsl_ast_expr_t **items; int32_t count; }                init_list;
		struct { svsl_ast_type_t *result_type; svsl_ast_spv_inst_t *insts; int32_t inst_count; } spirv_asm;
	};
};

// --- variables (locals, globals, params, struct/block members) --------------------

typedef enum svsl_var_flag_ {
	svsl_var_flag_static         = 1 << 0,
	svsl_var_flag_const          = 1 << 1,
	svsl_var_flag_workgroup      = 1 << 2, // 'workgroup' or legacy 'groupshared'
	svsl_var_flag_specialization = 1 << 3,
	svsl_var_flag_readonly       = 1 << 4,
	svsl_var_flag_writeonly      = 1 << 5,
	svsl_var_flag_coherent       = 1 << 6,
	svsl_var_flag_volatile       = 1 << 7,
	svsl_var_flag_uniform        = 1 << 8, // legacy 'uniform' prefix on a global
	svsl_var_flag_precise        = 1 << 9,  // 'precise': no fma contraction in the value's computation
	svsl_var_flag_legacy_spelling= 1 << 10, // declared with a legacy HLSL alias (drives porting hints)
} svsl_var_flag_;

typedef enum svsl_interp_ {
	svsl_interp_none          = 0,
	svsl_interp_flat          = 1 << 0, // 'flat' or legacy 'nointerpolation'
	svsl_interp_noperspective = 1 << 1,
	svsl_interp_centroid      = 1 << 2,
	svsl_interp_sample        = 1 << 3,
	svsl_interp_invariant     = 1 << 4,
	svsl_interp_linear        = 1 << 5, // HLSL 'linear' (the default; accepted)
} svsl_interp_;

typedef enum svsl_dir_ { svsl_dir_in = 0, svsl_dir_out, svsl_dir_inout } svsl_dir_;

struct svsl_ast_var_t {
	svsl_str_t       name;
	svsl_ast_type_t *type;
	svsl_ast_expr_t *init;     // optional
	svsl_str_t       semantic; // empty if none; case kept as written
	svsl_ast_reg_t   reg;
	uint32_t         flags;    // svsl_var_flag_
	uint8_t          interp;   // svsl_interp_
	uint8_t          dir;      // svsl_dir_ (params)
	uint8_t          pack;     // svsl_pack_ — layout keyword; buffer resources only
	int16_t          bit_width;  // packed-struct bit field width; -1 = not a bit field
	uint8_t          bit_format; // svsl_bitfmt_
	svsl_ast_attrs_t attrs;
	svsl_loc_t       loc;
};

// --- statements --------------------------------------------------------------------

typedef enum svsl_stmt_ {
	svsl_stmt_block,
	svsl_stmt_expr,
	svsl_stmt_var_decl,
	svsl_stmt_if,
	svsl_stmt_for,
	svsl_stmt_while,
	svsl_stmt_do,
	svsl_stmt_switch,
	svsl_stmt_break,
	svsl_stmt_continue,
	svsl_stmt_return,
	svsl_stmt_discard,
	svsl_stmt_demote,
} svsl_stmt_;

typedef struct svsl_ast_case_t {
	svsl_ast_expr_t  *value; // NULL = default
	svsl_ast_stmt_t **stmts;
	int32_t           stmt_count;
	svsl_loc_t        loc;
} svsl_ast_case_t;

struct svsl_ast_stmt_t {
	svsl_stmt_       kind;
	svsl_loc_t       loc;
	svsl_ast_attrs_t attrs; // [unroll], [branch], ...
	union {
		struct { svsl_ast_stmt_t **stmts; int32_t count; }                   block;
		svsl_ast_expr_t                                                     *expr;
		struct { svsl_ast_var_t **vars; int32_t count; }                     var_decl;
		struct { svsl_ast_expr_t *cond; svsl_ast_stmt_t *then_stmt, *else_stmt; } if_stmt;
		struct { svsl_ast_stmt_t *init; svsl_ast_expr_t *cond; svsl_ast_expr_t *inc;
		         svsl_ast_stmt_t *body; }                                    for_stmt;
		struct { svsl_ast_expr_t *cond; svsl_ast_stmt_t *body; }             while_stmt; // also do
		struct { svsl_ast_expr_t *value; svsl_ast_case_t *cases; int32_t case_count; } switch_stmt;
		svsl_ast_expr_t                                                     *return_value; // optional
	};
};

// --- top-level declarations -----------------------------------------------------------

typedef enum svsl_block_kind_ {
	svsl_block_uniform,       // 'uniform' or legacy 'cbuffer'
	svsl_block_storagebuffer,
	svsl_block_pushconstant,
} svsl_block_kind_;

typedef enum svsl_pack_ {
	svsl_pack_default = 0,
	svsl_pack_1,   // 'pack1' / 'scalar' — C layout (scalarBlockLayout when members straddle)
	svsl_pack_8,   // 'pack8' / 'relaxed' — vectors align to min(natural, 8)
	svsl_pack_16,  // 'pack16' / 'std140'
	svsl_pack_430, // 'std430' — the explicit spelling of the storage-buffer block default
} svsl_pack_;

typedef struct svsl_ast_struct_t {
	svsl_str_t       name;
	svsl_ast_var_t **members;
	int32_t          member_count;
	svsl_loc_t       loc;
} svsl_ast_struct_t;

typedef struct svsl_ast_enum_item_t {
	svsl_str_t       name;
	svsl_ast_expr_t *value; // NULL = previous + 1 (0 for the first)
	svsl_loc_t       loc;
} svsl_ast_enum_item_t;

typedef struct svsl_ast_enum_t {
	svsl_str_t             name;       // empty when anonymous
	svsl_ast_type_t       *underlying; // NULL = default int
	svsl_ast_enum_item_t  *items;
	int32_t                item_count;
	svsl_loc_t             loc;
} svsl_ast_enum_t;

typedef struct svsl_ast_block_t {
	svsl_block_kind_ kind;
	bool             legacy;   // spelled 'cbuffer' (porting hint)
	uint32_t         flags;    // access modifiers (svsl_var_flag_readonly etc.)
	svsl_pack_       pack;
	svsl_str_t       name;
	svsl_ast_reg_t   reg;
	svsl_ast_attrs_t attrs;    // [[vk::push_constant]] etc.
	svsl_ast_var_t **members;
	int32_t          member_count;
	svsl_loc_t       loc;
} svsl_ast_block_t;

typedef struct svsl_ast_func_t {
	svsl_str_t       name;
	svsl_ast_type_t *return_type;
	svsl_str_t       return_semantic;
	svsl_ast_var_t **params;
	int32_t          param_count;
	svsl_ast_stmt_t *body;
	svsl_ast_attrs_t attrs;    // [vertex], [compute(x,y,z)], [numthreads], [wave_size], ...
	svsl_loc_t       loc;
} svsl_ast_func_t;

typedef enum svsl_decl_ {
	svsl_decl_struct,
	svsl_decl_enum,
	svsl_decl_block,
	svsl_decl_var,     // globals: resources, bare globals, static const, workgroup
	svsl_decl_func,
	svsl_decl_include, // language-level: include "file"
} svsl_decl_;

typedef struct svsl_ast_decl_t {
	svsl_decl_ kind;
	svsl_loc_t loc;
	union {
		svsl_ast_struct_t struct_decl;
		svsl_ast_enum_t   enum_decl;
		svsl_ast_block_t  block;
		svsl_ast_var_t    var;
		svsl_ast_func_t   func;
		svsl_str_t        include_path;
	};
} svsl_ast_decl_t;

typedef struct svsl_ast_t {
	svsl_ast_decl_t **decls;
	int32_t           decl_count;
} svsl_ast_t;

// --- dump (for golden tests and --dump-ast debugging) ----------------------------------

// Renders the AST as a compact s-expression text, NUL-terminated, arena-owned.
const char *svsl_ast_dump(svsl_arena_t *arena, const svsl_ast_t *ast);
