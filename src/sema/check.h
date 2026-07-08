// Body typechecking: annotates every expression with its type and resolved
// symbol, applies implicit conversions, resolves intrinsic/method overloads,
// and rejects recursion (full inlining requires a call DAG).

#pragma once

#include "sema.h"

// Checks every function body in the program. Called by svsl_sema_run.
void svsl_check_functions(svsl_arena_t *arena, svsl_program_t *prog, svsl_diag_list_t *ref_diags);
