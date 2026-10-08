// Constant evaluation (LLVM's ConstantFolding): the constant view of IR values
// as scalar lanes, and the exact per-lane evaluation of every op the optimizer
// folds. Shared by the fold pass and combine (reassociation, identities).
//
// Exactness rules, so a folded bit pattern is the one the GPU computes:
//   * integers evaluate at their real width and signedness; no div/rem by zero,
//     no INT_MIN / -1, and no shift by >= the width (undefined in SPIR-V);
//   * float32 (and `half`, stored as relaxed float32) evaluate add/sub/mul/div
//     and compares; C's `==`/`<` are ordered and `!=` unordered, exactly emit's
//     FOrd*/FUnordNotEqual choices. float16 and float64 never evaluate.
// Intrinsics don't evaluate: on the corpus and sk_texenc they all but never met
// constant operands (docs/PLAN_optimizer_llvm.md, "Ablation").

#pragma once

#include "ir_edit.h"

typedef struct svsl_ir_lanes_t {
	uint64_t bits[4]; // svsl_ir_const's encoding per lane
	int32_t  count;
} svsl_ir_lanes_t;

// Lanes of a constant scalar/vector value: a `const`, or a vector `construct`
// of constants (flattened, up to four lanes). False when any lane isn't constant.
bool svsl_ir_const_lanes(const svsl_ir_edit_t *ed, uint32_t id, svsl_ir_lanes_t *out);

// The one bit pattern of a constant whose lanes all agree (a scalar or a splat).
bool svsl_ir_const_splat(const svsl_ir_edit_t *ed, uint32_t id, uint64_t *out_bits);

// Component scalar and lane count of a scalar/vector type; false for others.
bool svsl_ir_lane_type(const svsl_types_t *types, svsl_type_id_t type,
                       svsl_scalar_ *out_scalar, int32_t *out_count);

// One lane of an op over operands of scalar kind `s` (a compare's operands, not
// its bool result). False when the result is not exactly defined.
bool svsl_ir_eval_binary (svsl_ir_op_ op, svsl_scalar_ s, uint64_t a, uint64_t b, uint64_t *out);
bool svsl_ir_eval_shift  (svsl_ir_op_ op, svsl_scalar_ s, svsl_scalar_ amount, uint64_t a, uint64_t b,
                          uint64_t *out);
bool svsl_ir_eval_unary  (svsl_ir_op_ op, svsl_scalar_ s, uint64_t a, uint64_t *out);
bool svsl_ir_eval_convert(svsl_scalar_ from, svsl_scalar_ to, uint64_t a, uint64_t *out);
// The value a constant global (`static const` table) holds at a constant index
// path, as lanes of the loaded scalar/vector `type`: a slice of the global's
// evaluated value (svsl_global_t.value), so the folded bits are exactly the
// variable's. False when it has no value or the path doesn't name a foldable
// scalar or vector.
bool svsl_ir_const_global_lanes(const svsl_program_t *prog, uint32_t global, const uint32_t *index,
                                int32_t index_count, svsl_type_id_t type, svsl_ir_lanes_t *out);

// A constant of `type` (scalar or vector) with these lanes: head scalars, and
// for a vector a construct inserted before `before`. Returns its id.
uint32_t svsl_ir_make_const(svsl_ir_edit_t *ed, svsl_types_t *types, uint32_t before,
                            svsl_type_id_t type, const svsl_ir_lanes_t *lanes);
