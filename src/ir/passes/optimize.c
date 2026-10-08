// The optimizer driver: a fixed, iterated list of passes - not a general pass
// manager. Each pass records its changes in an edit buffer that the driver
// commits right after it, so the next pass starts on a fully resolved
// function (nops may remain between passes; the final commit compacts). The
// list cycles until every pass has run once since the last change (bounded by
// SVSL_OPT_MAX_ITERS rounds), then a single final DCE sweeps everything the
// value passes orphaned. See docs/OPTIMIZATION_PLAN.md and docs/PLAN_optimizer_llvm.md.
//
// Pass order is dependency-driven so each iteration extracts as much as it can
// (fewer iterations to reach the fixpoint - the output is the same regardless):
//   1. fold     - materialize constants,
//   2. combine  - identities, composite and boolean rewrites, canonical form,
//        ...so the following passes see the simplest form of each value.
//   3. cfg      - constant branches, empty arms, unreachable code, run-once
//        loops: consumes the constants 1-2 made, and removes the markers that
//        scope cse and forwarding (stores it unwraps become unconditional).
//   4. cse      - merge duplicate pure ops *including address computation*
//        (ptr/chain). Running it before forwarding is the key ordering choice:
//        once two duplicate pointers collapse to one id, forwarding can dedup
//        their loads in this same sweep instead of waiting a whole iteration.
//      sroa     - arrays used only through constant indices split into one
//        variable per element (after CSE: one chain per element), for forwarding.
//   5. forward  - store->load forwarding + redundant-load elimination, using
//        the pointers CSE merged in step 4.
//   6. dse      - forwarding removed the readers; DSE removes the dead stores.
//   7. unroll   - last, once the loops are as small as they get: copies out the
//        [unroll] loops whose counter indexes an array; the next cycle's
//        forwarding and folding make each copy's indices constant.
// (Measured: this order converges the corpus in ~9% fewer iterations and a
// third fewer forwarding sweeps than the naive fold->...->cse order.)
//
// Pass temporaries live in a scratch arena released after every commit, so the
// result arena only grows by what the function itself keeps.

#include "passes.h"
#include "../ir_verify.h"

#include <stdio.h>

static const struct { svsl_ir_pass_fn run; const char *name; } pipeline[] = {
	{ svsl_ir_fold,    "fold"    },
	{ svsl_ir_combine, "combine" },
	{ svsl_ir_cfg,     "cfg"     },
	{ svsl_ir_cse,     "cse"     },
	{ svsl_ir_sroa,    "sroa"    },
	{ svsl_ir_forward, "forward" },
	{ svsl_ir_dse,     "dse"     },
	{ svsl_ir_unroll,  "unroll"  },
};

// Debug builds check the IR's invariants after lowering and after every commit
// (ir_verify.h). A violation is a compiler bug: it fails this compile with a
// located error naming the pass that broke it, and the optimizer stops there -
// never the host process, which may be an app embedding libsvsl.
static bool verify(svsl_arena_t *arena, svsl_arena_t *scratch, const svsl_ir_func_t *fn, const char *after,
                   svsl_diag_list_t *ref_diags) {
#ifdef NDEBUG
	(void)arena; (void)scratch; (void)fn; (void)after; (void)ref_diags;
	return true;
#else
	svsl_ir_verify_error_t err;
	if (svsl_ir_verify(scratch, fn, &err)) return true;
	svsl_loc_t loc = err.inst < (uint32_t)fn->insts.count ? fn->insts.items[err.inst].loc : (svsl_loc_t){0};
	if (loc.line == 0) loc = fn->entry->func->loc;
	svsl_diag_add(arena, ref_diags, svsl_severity_error, loc,
	              "internal compiler error: the IR broke after %s (instruction %u: %s); -O0 may work around it",
	              after, err.inst, err.what);
	return false;
#endif
}

// Debug builds name a function the fixpoint stopped short on: its output is
// correct but not fully optimized, so either a pass pair ping-pongs (a bug) or
// the cap needs raising for a new kind of shader.
static void cap_reached(const svsl_ir_func_t *fn) {
#ifdef NDEBUG
	(void)fn;
#else
	svsl_str_t name = fn->entry ? fn->entry->name : (svsl_str_t){0};
	fprintf(stderr, "svsl: optimizer stopped at its %d-round cap on '%.*s'\n", SVSL_OPT_MAX_ITERS, name.len, name.ptr);
#endif
}

// Runs one pass and commits it; returns whether the function changed. A broken
// result clears *ref_ok (and has been reported).
static bool run_pass(svsl_ir_pass_fn pass, const char *name, svsl_arena_t *arena, svsl_arena_t *scratch,
                     svsl_ir_func_t *fn, svsl_program_t *prog, svsl_opt_level_ level, bool compact,
                     svsl_diag_list_t *ref_diags, bool *ref_ok) {
	svsl_ir_edit_t ed;
	svsl_ir_edit_begin(&ed, arena, scratch, fn);
	pass(&ed, prog, level);
	bool changed = svsl_ir_edit_commit(&ed, compact);
	if (changed && !verify(arena, scratch, fn, name, ref_diags)) *ref_ok = false;
	svsl_arena_reset(scratch); // the next pass reuses the memory: no fresh pages to fault in
	return changed;
}

bool svsl_ir_optimize(svsl_arena_t *arena, svsl_ir_func_t *fn, svsl_program_t *prog,
                      svsl_opt_level_ level, svsl_diag_list_t *ref_diags) {
	svsl_arena_t scratch = {0};
	bool         ok      = verify(arena, &scratch, fn, "lowering", ref_diags);
	// cycle through the pipeline until every pass has run once since the last
	// change: a pass whose input hasn't changed since its last run has nothing to do
	int32_t npass = (int32_t)(sizeof(pipeline) / sizeof(pipeline[0]));
	int32_t quiet = 0;
	int32_t step  = 0;
	for (; ok && level != svsl_opt_none && step < SVSL_OPT_MAX_ITERS * npass && quiet < npass; step++) {
		if (run_pass(pipeline[step % npass].run, pipeline[step % npass].name, arena, &scratch, fn, prog, level,
		             false, ref_diags, &ok))
			quiet = 1;
		else
			quiet++;
	}
	if (ok && step == SVSL_OPT_MAX_ITERS * npass && quiet < npass) cap_reached(fn);
	// DCE always runs, even at -O0: it keeps dead branches out of structured CF.
	// Its commit compacts, so the function leaves the optimizer dense.
	if (ok) run_pass(svsl_ir_dce, "dce", arena, &scratch, fn, prog, level, true, ref_diags, &ok);
	svsl_arena_free(&scratch);
	return ok;
}
