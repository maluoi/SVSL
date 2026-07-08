// IR: function-scoped, SSA-ish, flat arrays. Values are instruction indices.
// Control flow keeps the AST's structure via marker ops (SPIR-V needs structured
// CF anyway); variables that cross blocks stay as var/load/store — no general phi.
// Opaque types (textures/samplers/images) never appear as values: resource ops
// carry resource-table indices, resolved at inline time for opaque parameters.

#pragma once

#include "../sema/sema.h"
#include "../util/array.h"

#include <stdint.h>

#define SVSL_IR_NONE ((uint32_t)~0u)

typedef enum svsl_ir_op_ {
	svsl_ir_nop = 0,

	// values
	svsl_ir_const,       // bits in args[0] (low) / args[1] (high)
	svsl_ir_spec_const,  // args[0] = spec constant index
	svsl_ir_undef,       // placeholder value of the given type

	// memory (pointer-producing ops, then load/store)
	svsl_ir_var,         // function-local variable; type = pointee
	svsl_ir_param,       // entry-point input: args[0] = param index; type = pointee
	svsl_ir_ptr,         // global storage: args[0] = svsl_ref_ kind, args[1] = a, args[2] = b
	svsl_ir_chain,       // args[0] = base pointer; index value ids in aux; type = pointee
	svsl_ir_load,        // args[0] = pointer
	svsl_ir_store,       // args[0] = pointer, args[1] = value

	// composites
	svsl_ir_construct,   // components in aux → vector/matrix/array/struct
	svsl_ir_extract,     // args[0] = composite value, args[1] = literal index
	svsl_ir_insert,      // args[0] = composite value, args[1] = literal index, args[2] = value
	svsl_ir_shuffle,     // args[0] = vector value, args[1] = packed nibble indices, args[2] = count
	svsl_ir_extract_dynamic, // args[0] = vector value, args[1] = index value (non-const component)

	// arithmetic / logic (int vs float chosen from type at emit)
	svsl_ir_add, svsl_ir_sub, svsl_ir_mul, svsl_ir_div, svsl_ir_rem,
	svsl_ir_neg, svsl_ir_bit_not, svsl_ir_log_not,
	svsl_ir_bit_and, svsl_ir_bit_or, svsl_ir_bit_xor, svsl_ir_shl, svsl_ir_shr,
	svsl_ir_eq, svsl_ir_ne, svsl_ir_lt, svsl_ir_le, svsl_ir_gt, svsl_ir_ge,
	svsl_ir_log_and, svsl_ir_log_or, // componentwise, not short-circuit (HLSL/glslang)
	svsl_ir_select,      // args[0] = cond, args[1] = a, args[2] = b
	svsl_ir_convert,     // args[0] = value; converts operand type → inst type
	svsl_ir_mat_mul,     // args[0], args[1]: mat*mat, mat*vec, vec*mat by operand shapes

	// intrinsics and resources
	svsl_ir_intrinsic,   // args[0] = intrinsic table index; operand value ids in aux
	svsl_ir_tex,         // args[0] = texture res, args[1] = sampler res (or NONE),
	                     // args[2] = svsl_method_ (+ channel<<8); operands in aux
	svsl_ir_image_load,  // args[0] = image res, args[1] = coord
	svsl_ir_image_store, // args[0] = image res, args[1] = coord, args[2] = value
	svsl_ir_image_atomic,// args[0] = image res, args[1] = coord, args[2] = value, args[3] = op
	svsl_ir_atomic,      // args[0]=pointer, args[1]=value, args[2]=compare (cmpxchg), args[3]=op|(order<<8)
	svsl_ir_spirv_asm,   // inline SPIR-V: args[0] = index into fn->asm_nodes; $value ids in aux
	svsl_ir_bitfield_extract, // args[0]=value, [1]=offset id, [2]=width id; S/U by result type
	svsl_ir_bitfield_insert,  // args[0]=base, [1]=insert, [2]=offset id, [3]=width id

	// structured control flow
	svsl_ir_if,          // args[0] = cond
	svsl_ir_else,
	svsl_ir_end_if,
	svsl_ir_loop,        // body … loop_continue … (increment) … end_loop
	svsl_ir_loop_continue,
	svsl_ir_end_loop,
	svsl_ir_break,
	svsl_ir_continue,
	svsl_ir_switch,      // args[0] = value; case literals in aux (parallel to case markers)
	svsl_ir_case,        // args[0] = case ordinal, args[1] = 1 when default
	svsl_ir_end_switch,
	svsl_ir_return,      // args[0] = value or SVSL_IR_NONE
	svsl_ir_discard,
	svsl_ir_demote,
} svsl_ir_op_;

// per-instruction flags (packs into the padding after `op`)
enum { svsl_ir_flag_precise = 1 << 0 }; // no fma contraction (from `precise`) → OpDecorate NoContraction

typedef struct svsl_ir_inst_t {
	uint8_t        op;        // svsl_ir_op_
	uint8_t        flags;     // svsl_ir_flag_
	svsl_type_id_t type;      // result type (SVSL_TYPE_NONE for non-values)
	uint32_t       args[4];
	uint32_t       aux;       // offset into the function's aux pool
	uint32_t       aux_count;
	svsl_loc_t     loc;
	svsl_str_t     name;      // optional, for dumps ("o", "world", …)
} svsl_ir_inst_t;

struct svsl_ast_expr_t;

typedef struct svsl_ir_func_t {
	const svsl_entry_t *entry;
	svsl_array_t(svsl_ir_inst_t) insts;
	svsl_array_t(uint32_t)       aux;
	// spirv_asm blocks keep their opcode/operand structure in the AST; the IR
	// carries only the dataflow ($value ids in aux). svsl_ir_spirv_asm.args[0]
	// indexes here to recover the block to emit.
	svsl_array_t(const struct svsl_ast_expr_t *) asm_nodes;
} svsl_ir_func_t;

typedef struct svsl_ir_module_t {
	svsl_ir_func_t *funcs; // one per program entry point, same order
	int32_t         func_count;
} svsl_ir_module_t;

// svsl_opt_level_ is public (see <svsl/svsl.h>). Higher levels are supersets of
// lower ones. -O1 is the default and the level the pixel/bit correctness oracle
// runs against: value-preserving transforms only (+ exact integer identities), so
// program results are unchanged bit-for-bit. -O2 adds float-algebraic identities
// that may change IEEE edge results (-0.0/NaN/Inf/rounding) and is NOT oracle-
// covered. See docs/OPTIMIZATION_PLAN.md.

// Upper bound on optimizer fixpoint iterations. Passes are monotone (they only
// remove work), so this is a safety backstop, not a tuning knob.
#define SVSL_OPT_MAX_ITERS 8

// Lowers every entry point with full inlining of user calls, then runs the
// optimizer at the given level. Returns false if any error diagnostic was emitted.
bool svsl_ir_build(svsl_arena_t *arena, svsl_program_t *prog, svsl_opt_level_ opt_level,
                   svsl_ir_module_t *out_module, svsl_diag_list_t *ref_diags);

// The optimizer: a fixed, iterated list of pure passes (no pass manager). Each
// pass rewrites in place or nops instructions out — indices stay stable and
// value references stay forward-only. See docs/OPTIMIZATION_PLAN.md §3.
void svsl_ir_optimize(svsl_arena_t *arena, svsl_ir_func_t *fn,
                      const svsl_program_t *prog, svsl_opt_level_ level);

// Individual passes. Each returns true when it changed the function, so the
// driver can iterate to a fixpoint. Dead instructions become nops; indices stay
// stable, so users never need patching.
bool svsl_ir_fold   (svsl_ir_func_t *fn, const svsl_types_t *types); // constant folding
bool svsl_ir_peephole(svsl_arena_t *arena, svsl_ir_func_t *fn,       // pattern simplification
                      const svsl_types_t *types, svsl_opt_level_ level);
bool svsl_ir_forward(svsl_arena_t *arena, svsl_ir_func_t *fn,        // store→load + redundant-load
                     const svsl_program_t *prog);
bool svsl_ir_dse    (svsl_arena_t *arena, svsl_ir_func_t *fn);       // dead-store elimination
bool svsl_ir_cse    (svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_types_t *types); // CSE
bool svsl_ir_dce    (svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_types_t *types); // DCE

// Text dump for --dump-ir and golden tests; arena-owned, NUL-terminated.
const char *svsl_ir_dump(svsl_arena_t *arena, const svsl_ir_module_t *module,
                         const svsl_program_t *prog);
