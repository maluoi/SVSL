// The optimizer's passes (private to src/ir/passes). Each one reads ed->fn and
// records its changes in the edit buffer; the driver (optimize.c) commits after
// every pass, so a pass always starts on a fully-resolved function. Killed
// instructions may linger as nops until a compacting commit.

#pragma once

#include "../ir_edit.h"

typedef void (*svsl_ir_pass_fn)(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level);

void svsl_ir_fold    (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // constant folding
void svsl_ir_combine (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // simplify + canonicalize
void svsl_ir_cfg     (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // structured CFG simplification
void svsl_ir_cse     (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // common subexpressions
void svsl_ir_sroa    (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // split constant-indexed arrays
void svsl_ir_forward (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // store->load + redundant loads
void svsl_ir_dse     (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // dead stores
void svsl_ir_unroll  (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // [unroll] loops that index arrays
void svsl_ir_dce     (svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level); // dead code
