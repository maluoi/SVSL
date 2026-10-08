// The IR edit buffer: every optimizer pass records its changes here, and one
// O(n) commit applies them. The IRBuilder + replaceAllUsesWith of this IR.
//
// A pass may:
//   * replace a value (all uses redirect to an earlier or inserted value, the
//     original dies),
//   * kill an instruction (markers, stores, or values nothing uses),
//   * insert new instructions before an existing one, or constants at the head,
//   * rewrite an instruction's own fields in place (then call svsl_ir_edit_touch).
//
// Inserted instructions get provisional ids (>= base) that are valid operands
// straight away. A commit with inserts (or a compacting one) places each insert
// before its anchor, drops killed instructions and nops, re-packs the aux pool,
// and renumbers every operand: the function comes out dense, still referencing
// only earlier ids. Without inserts, a commit applies in place instead - killed
// instructions become nops and operands are redirected - which is all most
// passes need and keeps the fixpoint loop cheap; the driver's final commit
// compacts. An insert anchored before X may only reference ids defined before X
// (or earlier inserts at the same anchor) - the invariant the commit checks in
// debug builds.

#pragma once

#include "ir.h"

typedef struct svsl_ir_edit_t {
	svsl_ir_func_t *fn;
	svsl_arena_t   *arena;   // owns fn's arrays (result lifetime)
	svsl_arena_t   *scratch; // this edit's and its pass's temporaries
	uint32_t        base;    // instruction count at begin; ids >= base are provisional
	uint32_t       *repl;    // [base] 0 = kept, NONE = killed, else the replacement + 1
	uint32_t        lowest;  // lowest killed/replaced id (base = none): commits start there
	svsl_array_t(uint32_t)       dead;   // killed/replaced ids: a commit with nothing to redirect walks only these
	svsl_array_t(svsl_ir_inst_t) added;  // provisional id = base + index
	svsl_array_t(uint32_t)       anchor; // per added: original id it precedes
	uint32_t       *const_slot;          // edit_const dedup (open addressing, provisional ids)
	uint32_t        const_cap;
	bool            changed;  // anything at all (in-place rewrites included)
	bool            replaced; // some value now resolves elsewhere: operands need redirecting
	bool            resolved; // the pass resolved every operand as it walked: commit needn't
} svsl_ir_edit_t;

void svsl_ir_edit_begin(svsl_ir_edit_t *ed, svsl_arena_t *arena, svsl_arena_t *scratch,
                        svsl_ir_func_t *fn);

// The instruction behind any id, original or provisional.
static inline svsl_ir_inst_t *svsl_ir_edit_inst(const svsl_ir_edit_t *ed, uint32_t id) {
	return id < ed->base ? &ed->fn->insts.items[id] : &ed->added.items[id - ed->base];
}

// Follow replacements to the value that will survive (NONE when killed).
// Inline: every pass calls it per operand.
static inline uint32_t svsl_ir_edit_resolve(const svsl_ir_edit_t *ed, uint32_t id) {
	while (id < ed->base) {
		uint32_t r = ed->repl[id];
		if (r == 0)            return id;
		if (r == SVSL_IR_NONE) return SVSL_IR_NONE;
		id = r - 1;
	}
	return id;
}

// Redirect instruction `id`'s value operands through the replacements recorded
// so far (in place). Passes that key on operand identity mid-walk (CSE) call it
// so a later instruction sees the canonical ids an earlier replacement chose.
// A pass that calls it on every instruction in order may set `resolved`:
// operands only name earlier ids, so nothing is left for the commit to redirect.
void svsl_ir_edit_resolve_operands(svsl_ir_edit_t *ed, uint32_t id);

void svsl_ir_edit_replace(svsl_ir_edit_t *ed, uint32_t id, uint32_t with);
void svsl_ir_edit_kill   (svsl_ir_edit_t *ed, uint32_t id);
void svsl_ir_edit_touch  (svsl_ir_edit_t *ed); // an in-place rewrite happened

// Inserts `inst` before original instruction `before` (base = at the end); its
// aux operands are copied from opt_aux (which may point into fn's own aux pool).
// Returns the provisional id.
uint32_t svsl_ir_edit_insert(svsl_ir_edit_t *ed, uint32_t before, svsl_ir_inst_t inst,
                             const uint32_t *opt_aux, uint32_t aux_count);

// A scalar constant at the function head (dominates every use), deduplicated
// within this edit. Bits use svsl_ir_const's encoding.
uint32_t svsl_ir_edit_const(svsl_ir_edit_t *ed, svsl_type_id_t type, uint64_t bits);

// Applies everything; returns true when the function changed. `compact` forces
// the dense rebuild even without inserts.
bool svsl_ir_edit_commit(svsl_ir_edit_t *ed, bool compact);
