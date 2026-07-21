// Numeric constant folding for global initializers: literals, +-*/, negation,
// casts, and references to other constant globals. Shared by the SPIR-V and
// WGSL backends, which both materialize `static const` globals from the AST.

#pragma once

#include "sema.h"

#include <stdbool.h>
#include <stdint.h>

bool svsl_const_eval_num(const svsl_program_t *prog, const struct svsl_ast_expr_t *expr,
                         double *out);

// componentwise fold for vector-typed constant expressions: constructors and
// init lists fill, scalars splat, and +-*/ combine (1.0 / float3(...) et al.)
bool svsl_const_eval_vec(const svsl_program_t *prog, const struct svsl_ast_expr_t *expr,
                         int32_t n, double *out);
