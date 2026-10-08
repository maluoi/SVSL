// The IR verifier (LLVM's Verifier, for this IR's invariants). Every pass relies
// on these and nothing else enforces them, so debug builds check them after
// lowering and after every optimizer commit:
//   * an operand names an earlier, live, value-producing instruction;
//   * a value is only used where it dominates: in the arm that computed it, or
//     an arm nested inside it (values never escape an arm - SPIR-V's structured
//     dominance). Function-scope entities are exempt: constants, undefs, spec
//     constants, and the vars emit hoists to the entry block;
//   * markers nest: else/end_if close an if, loop_continue/end_loop a loop,
//     case/end_switch a switch; break sits in a loop or switch, continue in a loop;
//   * aux ranges lie inside the aux pool.

#pragma once

#include "ir.h"

typedef struct svsl_ir_verify_error_t {
	uint32_t    inst; // the offending instruction
	const char *what;
} svsl_ir_verify_error_t;

// True when `fn` satisfies every invariant; otherwise fills out_error (first violation).
bool svsl_ir_verify(svsl_arena_t *scratch, const svsl_ir_func_t *fn, svsl_ir_verify_error_t *out_error);
