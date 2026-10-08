// Constant initializers: a `static const` global's initializer, or a hoisted
// local constant table's, evaluated once after checking into its value. Every
// consumer - SPIR-V and WGSL emit, the IR folder - reads that value; none walks
// the initializer itself.
//
// The value is a flat array of scalar leaves, in the type's natural order:
// array elements, matrix rows, then vector (row) components. Each leaf is the
// scalar's bits in the IR constant encoding: bool 0/1; integers sign- or
// zero-extended from their width to 64 bits; float32 and half the float32 bits;
// float16 the 16-bit bits; float64 the 64-bit bits.
//
// Evaluation follows IR lowering's semantics on the checked expression (each
// node computed in its operands' type, then converted to its sema_type), so a
// constant reads the same as the runtime code it replaces: `1 / 2` is integer
// division. Float math runs in double and rounds once, at the leaf (glslang's
// folding does the same).

#pragma once

#include "sema.h"

#include <stdbool.h>
#include <stdint.h>

// leaves in a value of `type`; 0 for types without a constant form (structs, resources)
int32_t svsl_const_leaf_count(const svsl_types_t *types, svsl_type_id_t type);

// `expr` (checked) as a constant of `type`, or NULL when it isn't one. Names
// resolve through sema_ref: a constant global contributes its own value, which
// must already be evaluated (declaration order).
const uint64_t *svsl_const_eval(svsl_arena_t *arena, svsl_program_t *prog,
                                const struct svsl_ast_expr_t *expr, svsl_type_id_t type);
