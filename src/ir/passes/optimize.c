// The optimizer driver: a fixed, iterated list of pure passes — not a general
// pass manager. Each pass returns whether it changed anything; the loop runs until the IR
// stops changing (bounded by SVSL_OPT_MAX_ITERS), then a single final DCE
// sweeps everything the value passes orphaned. See docs/OPTIMIZATION_PLAN.md.
//
// Pass order is dependency-driven so each iteration extracts as much as it can
// (fewer iterations to reach the fixpoint — the output is the same regardless):
//   1. fold     — materialize constants,
//   2. peephole — extract/shuffle/identity rewrites into canonical form,
//        …so the following passes see the simplest form of each value.
//   3. cse      — merge duplicate pure ops *including address computation*
//        (ptr/chain). Running it before forwarding is the key ordering choice:
//        once two duplicate pointers collapse to one id, forwarding can dedup
//        their loads in this same sweep instead of waiting a whole iteration.
//   4. forward  — store→load forwarding + redundant-load elimination, using
//        the pointers CSE merged in step 3.
//   5. dse      — forwarding removed the readers; DSE removes the dead stores.
// (Measured: this order converges the corpus in ~9% fewer iterations and a
// third fewer forwarding sweeps than the naive fold→…→cse order.)

#include "../ir.h"

void svsl_ir_optimize(svsl_arena_t *arena, svsl_ir_func_t *fn,
                      const svsl_program_t *prog, svsl_opt_level_ level) {
	const svsl_types_t *types = &prog->types;
	if (level == svsl_opt_none) {
		svsl_ir_dce(arena, fn, types); // DCE always runs: keeps dead branches out of structured CF
		return;
	}

	for (int32_t iter = 0; iter < SVSL_OPT_MAX_ITERS; iter++) {
		bool changed = false;
		changed |= svsl_ir_fold    (fn, types);
		changed |= svsl_ir_peephole(arena, fn, types, level); // extract/shuffle/identity rewrites
		changed |= svsl_ir_cse     (arena, fn, types); // merges duplicate ptr/chain before forwarding
		changed |= svsl_ir_forward (arena, fn, prog); // store→load forwarding + redundant loads
		changed |= svsl_ir_dse     (arena, fn); // dead stores to now-unread local vars
		if (!changed) break;
	}

	svsl_ir_dce(arena, fn, types);
}
