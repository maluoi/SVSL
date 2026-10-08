// Structured control-flow table: where each marker's arm and construct end,
// and which arm every instruction sits in. One O(n) stack walk; the markers
// stay the source of truth, this is a derived view a pass rebuilds when it
// needs it (never cached across commits).
//
// An *arm* is a straight run of a construct: an if's then-arm (if .. else or
// end_if), its else-arm (else .. end_if), a loop's body (loop .. loop_continue
// or end_loop), its continue section (loop_continue .. end_loop), a switch's
// header (switch .. first case) and each case (case .. next case or end_switch).

#pragma once

#include "ir.h"

typedef struct svsl_ir_cf_t {
	uint32_t *arm;     // per instruction: the marker opening its innermost arm (NONE = top level)
	uint32_t *arm_end; // per opening marker: the marker closing that arm
	uint32_t *end;     // per if/loop/switch: its end_if/end_loop/end_switch
	uint32_t *head;    // per closing or mid marker (else, loop_continue, case, end_*): its if/loop/switch
} svsl_ir_cf_t;

void svsl_ir_cf_build(svsl_arena_t *scratch, const svsl_ir_func_t *fn, svsl_ir_cf_t *out);

// The innermost loop or switch enclosing instruction i (what a `break` there
// leaves); NONE at the top level. `continue_only` skips switches (what a
// `continue` targets).
uint32_t svsl_ir_cf_break_target(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t i,
                                 bool continue_only);

// Whether loop `l` can run its body more than once: the body doesn't end in an
// unconditional exit (an inlined call's early-return wrapper runs once, however
// many breaks it has). Nops are skipped; a pass with pending kills sees through
// them only once they're committed.
bool svsl_ir_cf_loop_repeats(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t l);

// Whether a loop that can repeat encloses instruction i.
bool svsl_ir_cf_in_repeating_loop(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t i);

// Dominance scopes for a single in-order walk (EarlyCSE's scoped tables, with
// structured regions as the dominator tree). Every open arm has a unique
// serial; something established inside an arm stays visible in the arms it
// encloses and dies when its arm closes or a sibling arm (else, the continue
// section, the next case) begins - so a recorded value is reusable exactly
// where it dominates.
typedef struct svsl_ir_scope_t {
	uint32_t *serial; // per depth: the open arm's serial
	int32_t   depth;
	uint32_t  next;
} svsl_ir_scope_t;

void svsl_ir_scope_begin(svsl_ir_scope_t *s, svsl_arena_t *scratch, const svsl_ir_func_t *fn);

// Advances past instruction op: opens, switches, or closes an arm at markers.
void svsl_ir_scope_step(svsl_ir_scope_t *s, svsl_ir_op_ op);

// Whether something recorded at (depth, serial) still dominates the walk.
static inline bool svsl_ir_scope_live(const svsl_ir_scope_t *s, int32_t depth, uint32_t serial) {
	return serial != 0 && depth <= s->depth && s->serial[depth] == serial;
}
