// Memory passes: store-to-load forwarding, redundant-load elimination, and
// dead-store elimination. Together they collapse the var/store/load traffic
// that full inlining and out/inout copies generate, plus repeated reads of the
// same local, parameter, or read-only global (cbuffer/pushconstant) location.
//
// Forwarding tracks the value currently at a pointer, keyed by exact pointer id,
// only for *forwardable roots*: function-local vars, entry parameters, private
// globals, and read-only globals (uniform/pushconstant buffers, const globals,
// read-only structured buffers). Those are the storages we can reason about soundly:
//   * distinct vars/params/private globals never alias (per-invocation storage,
//     no escaping pointers, one pointer instruction per private global);
//   * read-only globals are never written, so their loads are always reusable.
// A per-root generation, bumped on every store to a root, invalidates only the
// entries that share that root (same storage object -> may alias; different root
// -> cannot).
//
// Forwarding across control flow (dominance, EarlyCSE-style). An entry is
// stamped with the arm it was recorded in (svsl_ir_scope_t): it is reusable in
// that arm and the arms it encloses - exactly where its value dominates, since
// a stored value is defined in the store's arm or an enclosing one (values
// never escape the arm that computes them) - and dies when its arm closes or a
// sibling arm begins. What can clobber it is tracked by the root generation:
//   * a store or atomic bumps its root, invalidating that root's entries;
//   * entering a loop bumps every root stored anywhere in the loop: the
//     back-edge brings those stores around to the loop's top;
//   * an else arm runs instead of its then arm, so the then arm's bumps and the
//     entries it overwrote are undone for the else arm (an undo log), and the
//     bumps redone at end_if, where either arm's stores may have happened. A switch stays monotone: fallthrough can
//     carry one case's stores into the next.
// Generations are unique (a global counter), so an undone bump never revives
// an entry recorded inside the arm that made it. See docs/PLAN_optimizer_llvm.md
// item 3.
//
// A local holds nothing until its first store: its declaration records the whole
// var as undefined (UNDEF_VALUE), so a load no store reaches - a member of a
// struct returned early, before it was written - becomes an undef value, and
// DSE drops the store of it. (LLVM's mem2reg reads undef there the same way.)
// Not inside a loop that repeats, though: emit hoists the variable, so a read
// before this trip's store sees the previous trip's value, which the reference
// compilers keep (an uninitialized local is undefined, but not divergent).
//
// Writable globals (storage buffers, images, groupshared) are left untouched -
// reasoning about aliasing across duplicate global pointers would need pointer
// canonicalization; the win there is small.

#include "passes.h"
#include "../ir_cf.h"
#include "../ir_operands.h"

// One undo-log record: a root's previous generation, or a pointer's previous
// entry (overwritten inside an arm an else may need to see past).
typedef struct fwd_undo_t {
	uint32_t id;        // the root or pointer
	uint32_t gen;       // root: previous generation; pointer: previous e_gen
	uint32_t value;     // pointer: previous value (NONE marks a root record)
	uint32_t serial;
	int32_t  depth;
} fwd_undo_t;

// The forwarding state: the value known at each pointer, the generation of each
// root, and the if/else undo log (see the file header).
#define UNDEF_VALUE (SVSL_IR_NONE - 1) // forwarding state of a never-stored local

typedef struct fwd_t {
	const svsl_ir_func_t *fn;
	const svsl_program_t *prog;
	svsl_arena_t         *scratch;
	uint32_t *value;    // per pointer: the value at *P (a store's operand or a load)
	uint32_t *e_gen;    // per pointer: root_gen[root(P)] when recorded
	uint32_t *e_serial; // per pointer: the recording arm's serial (0 = empty)
	int32_t  *e_depth;  // per pointer: the recording arm's depth
	uint32_t *root_gen; // per root
	uint32_t  gen_next;
	svsl_array_t(fwd_undo_t) log;
	svsl_array_t(uint32_t)   frames; // per open if: (log start, log end of its then arm or NONE)
	svsl_array_t(uint32_t)   undefs; // undef values made this sweep, one per type
} fwd_t;

// an undef of `type`, made once per sweep at the head (undefs dominate everything)
static uint32_t undef_of(fwd_t *f, svsl_ir_edit_t *ed, svsl_type_id_t type) {
	for (int32_t k = 0; k < f->undefs.count; k++)
		if (svsl_ir_edit_inst(ed, f->undefs.items[k])->type == type) return f->undefs.items[k];
	uint32_t id = svsl_ir_edit_insert(ed, 0, (svsl_ir_inst_t){ .op = svsl_ir_undef, .type = type,
	                                                           .args = { 0, 0, 0, SVSL_IR_NONE } }, NULL, 0);
	svsl_array_push(f->scratch, &f->undefs, id);
	return id;
}

static void bump(fwd_t *f, uint32_t r) {
	if (f->frames.count)
		svsl_array_push(f->scratch, &f->log, (fwd_undo_t){ .id = r, .gen = f->root_gen[r], .value = SVSL_IR_NONE });
	f->root_gen[r] = ++f->gen_next;
}

static bool entry_live(const fwd_t *f, const svsl_ir_scope_t *scope, uint32_t p, uint32_t r) {
	return svsl_ir_scope_live(scope, f->e_depth[p], f->e_serial[p]) && f->e_gen[p] == f->root_gen[r];
}

static void record(fwd_t *f, const svsl_ir_scope_t *scope, uint32_t p, uint32_t r, uint32_t value) {
	if (f->frames.count && f->e_serial[p])
		svsl_array_push(f->scratch, &f->log, (fwd_undo_t){ .id = p, .gen = f->e_gen[p], .value = f->value[p],
		                                                   .serial = f->e_serial[p], .depth = f->e_depth[p] });
	f->value[p]    = value;
	f->e_gen[p]    = f->root_gen[r];
	f->e_depth[p]  = scope->depth;
	f->e_serial[p] = scope->serial[scope->depth];
}

// Control-flow markers: open/close the undo frames, and invalidate what a loop's
// back-edge can bring around.
static void forward_marker(fwd_t *f, const svsl_ir_cf_t *cf, uint32_t i, svsl_ir_op_ op) {
	const svsl_ir_func_t *fn = f->fn;
	switch (op) {
	case svsl_ir_if:
		svsl_array_push(f->scratch, &f->frames, (uint32_t)f->log.count);
		svsl_array_push(f->scratch, &f->frames, SVSL_IR_NONE);
		break;
	case svsl_ir_else: { // the else arm sees memory as it was before the if
		uint32_t start = f->frames.items[f->frames.count - 2];
		f->frames.items[f->frames.count - 1] = (uint32_t)f->log.count;
		for (int32_t k = f->log.count - 1; k >= (int32_t)start; k--) {
			const fwd_undo_t *u = &f->log.items[k];
			if (u->value == SVSL_IR_NONE) { f->root_gen[u->id] = u->gen; continue; }
			f->value[u->id]    = u->value;
			f->e_gen[u->id]    = u->gen;
			f->e_serial[u->id] = u->serial;
			f->e_depth[u->id]  = u->depth;
		}
		break;
	}
	case svsl_ir_end_if: { // past the merge, either arm's stores may have happened
		uint32_t start    = f->frames.items[f->frames.count - 2];
		uint32_t then_end = f->frames.items[f->frames.count - 1];
		f->frames.count -= 2;
		if (then_end != SVSL_IR_NONE)
			for (uint32_t k = start; k < then_end; k++)
				if (f->log.items[k].value == SVSL_IR_NONE) bump(f, f->log.items[k].id);
		if (!f->frames.count) f->log.count = 0; // nothing left to undo into
		break;
	}
	case svsl_ir_loop: // the back-edge: every store in the loop may precede its top
		for (uint32_t k = i + 1; k < cf->end[i]; k++) {
			const svsl_ir_inst_t *in = &fn->insts.items[k];
			if (in->op != svsl_ir_store && in->op != svsl_ir_atomic) continue;
			uint32_t r = svsl_ir_root_ptr(fn, in->args[0]);
			if (svsl_ir_forwardable_root(fn, f->prog, r)) bump(f, r);
		}
		break;
	default:
		break;
	}
}

void svsl_ir_forward(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn    = ed->fn;
	svsl_arena_t   *arena = ed->scratch;
	int32_t         count = fn->insts.count;
	if (count == 0) return;

	size_t size = (size_t)count * sizeof(uint32_t);
	fwd_t  f    = { .fn = fn, .prog = prog, .scratch = arena,
	                .value    = svsl_arena_alloc(arena, size), .e_gen   = svsl_arena_alloc(arena, size),
	                .e_serial = svsl_arena_alloc(arena, size), .e_depth = svsl_arena_alloc(arena, size),
	                .root_gen = svsl_arena_alloc(arena, size) };
	svsl_ir_cf_t    cf;
	svsl_ir_scope_t scope;
	svsl_ir_cf_build(arena, fn, &cf);
	svsl_ir_scope_begin(&scope, arena, fn);

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_     op   = (svsl_ir_op_)inst->op;

		if (svsl_ir_ends_run(op)) {
			forward_marker(&f, &cf, (uint32_t)i, op);
			svsl_ir_scope_step(&scope, op);
			continue;
		}

		switch (op) {
		case svsl_ir_var: // declared: nothing stored yet...
			// ...unless a loop around it repeats: its storage is hoisted, so the
			// last trip's value carries over (as glslang and DXC keep it)
			if (!svsl_ir_cf_in_repeating_loop(&cf, fn, (uint32_t)i))
				record(&f, &scope, (uint32_t)i, (uint32_t)i, UNDEF_VALUE);
			break;
		case svsl_ir_store: {
			uint32_t p = inst->args[0], r = svsl_ir_root_ptr(fn, p);
			if (!svsl_ir_forwardable_root(fn, prog, r)) break;
			bump(&f, r);                                       // invalidate this root's members
			record(&f, &scope, p, r, svsl_ir_edit_resolve(ed, inst->args[1]));
			break;
		}
		case svsl_ir_load: {
			uint32_t p = inst->args[0], r = svsl_ir_root_ptr(fn, p);
			if (!svsl_ir_forwardable_root(fn, prog, r)) break;
			if (entry_live(&f, &scope, p, r) && f.value[p] != UNDEF_VALUE) {
				svsl_ir_edit_replace(ed, (uint32_t)i, svsl_ir_edit_resolve(ed, f.value[p])); // store->load or load->load
				break;
			}
			if (entry_live(&f, &scope, r, r) && f.value[r] == UNDEF_VALUE) { // nothing stored to any of it yet
				svsl_ir_edit_replace(ed, (uint32_t)i, undef_of(&f, ed, inst->type));
				break;
			}
			// single-index load off a var whose whole value is live and dominates
			if (p != r && svsl_ir_is_invocation_local(fn, r) && entry_live(&f, &scope, r, r)) {
				const svsl_ir_inst_t *ch = &fn->insts.items[p];
				if (ch->op == svsl_ir_chain && ch->args[0] == r && ch->aux_count == 1) {
					uint32_t              idx_id = fn->aux.items[ch->aux];
					const svsl_ir_inst_t *idx    = &fn->insts.items[idx_id];
					uint32_t              v_id   = svsl_ir_edit_resolve(ed, f.value[r]);
					const svsl_ir_inst_t *val    = svsl_ir_edit_inst(ed, v_id); // may be this sweep's undef
					const svsl_type_t    *vt     = svsl_type_get(&prog->types, val->type);
					if (idx->op == svsl_ir_const) {
						// (a vector built from wider parts doesn't hold member k at aux[k])
						if (val->op == svsl_ir_construct && idx->args[0] < val->aux_count &&
						    (vt->kind != svsl_type_vector || (int32_t)val->aux_count == vt->count)) {
							// constant member of a live construct -> reuse that
							// component value directly (no new instruction)
							uint32_t comp = fn->aux.items[val->aux + idx->args[0]];
							if (svsl_ir_edit_inst(ed, comp)->type == inst->type) {
								svsl_ir_edit_replace(ed, (uint32_t)i, svsl_ir_edit_resolve(ed, comp));
								break;
							}
						}
						// constant member of any other live composite -> static
						// extract, no spill: the var/store/chain die (DCE), emit
						// uses OpCompositeExtract straight off the value
						if (vt->kind != svsl_type_scalar) {
							inst->op        = svsl_ir_extract;
							inst->args[0]   = v_id;
							inst->args[1]   = idx->args[0];
							inst->aux_count = 0;
							svsl_ir_edit_touch(ed);
							break;
						}
					} else if (vt->kind == svsl_type_vector) {
						// dynamic component of a live vector -> extract-dynamic, no spill:
						// the var/store/chain die (DCE), emit uses OpVectorExtractDynamic
						inst->op        = svsl_ir_extract_dynamic;
						inst->args[0]   = v_id;
						inst->args[1]   = svsl_ir_edit_resolve(ed, idx_id);
						inst->aux_count = 0;
						svsl_ir_edit_touch(ed);
						break;
					}
				}
			}
			record(&f, &scope, p, r, (uint32_t)i); // later loads reuse this one
			break;
		}
		case svsl_ir_atomic: {                                 // atomic on a local var writes it
			uint32_t r = svsl_ir_root_ptr(fn, inst->args[0]);
			if (svsl_ir_forwardable_root(fn, prog, r)) bump(&f, r);
			break;
		}
		default:
			break;
		}
	}
}

// Overwriting-store elimination: a store to an exact local pointer whose value
// is overwritten by a later store to the *same* pointer, with no load of that
// pointer's root and no control-flow boundary in between, is unobservable. The
// never-read pass below can't catch this - the root is read elsewhere (e.g. a
// whole-struct `return o` loads every member), so it keeps all member stores;
// but `o.color = a; o.color.rgb *= b;` writes the same member twice with the
// first never observed. Exact-pointer comparison is sound because CSE has
// already merged duplicate address computations to one id, and any intervening
// load of the same root (a whole-var load included) conservatively invalidates.
//
// Stamps instead of rescans: last[P] remembers P's previous store with the run
// and its root's read-stamp at that time; a later store to P kills it only if
// both still match (same run, no read of the root since). O(1) per op.
static void dse_overwriting(svsl_ir_edit_t *ed) {
	svsl_ir_func_t *fn    = ed->fn;
	svsl_arena_t   *arena = ed->scratch;
	int32_t         count = fn->insts.count;
	size_t          size  = (size_t)count * sizeof(uint32_t);
	uint32_t *last      = svsl_arena_alloc(arena, size); // last[P]: previous store to P, + 1 (0 = none)
	uint32_t *last_run  = svsl_arena_alloc(arena, size); // the run it was made in
	uint32_t *last_read = svsl_arena_alloc(arena, size); // read_stamp[root(P)] when it was made
	uint32_t *read_stamp = svsl_arena_alloc(arena, size); // per root: bumped by every read of it
	uint32_t  run = 1;

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_     op   = (svsl_ir_op_)inst->op;

		if (svsl_ir_ends_run(op)) { run++; continue; } // region boundary: a store before it may be observed
		if (op == svsl_ir_store) {
			uint32_t p = inst->args[0], r = svsl_ir_root_ptr(fn, p);
			if (!svsl_ir_is_invocation_local(fn, r)) continue; // writable globals may be observed
			if (last[p] && last_run[p] == run && last_read[p] == read_stamp[r])
				svsl_ir_edit_kill(ed, last[p] - 1);          // prior store to p, unread -> dead
			last[p]      = (uint32_t)i + 1;
			last_run[p]  = run;
			last_read[p] = read_stamp[r];
		} else if (op == svsl_ir_load || op == svsl_ir_atomic) {
			read_stamp[svsl_ir_root_ptr(fn, inst->args[0])]++; // a read of the root observes any member
		}
	}
}

// Dead-store elimination: a store to a local var that is never read anywhere in
// the function is unobservable. A var is "read" only via a load or atomic (a
// bare chain isn't a read); once forwarding has removed a temp's only load, all
// its stores fall here and DCE then reclaims the var itself. Also runs the
// overwriting-store pass (dead writes that a later write to the same pointer
// shadows), which the never-read test structurally cannot see.
void svsl_ir_dse(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)prog; (void)level;
	svsl_ir_func_t *fn    = ed->fn;
	int32_t         count = fn->insts.count;
	if (count == 0) return;

	dse_overwriting(ed);

	uint8_t *read = svsl_arena_alloc(ed->scratch, (size_t)count); // zeroed; read[var] = observed

	for (int32_t i = 0; i < count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_store || op == svsl_ir_chain || op == svsl_ir_var || op == svsl_ir_nop)
			continue; // stores only write; chain/var are not reads on their own

		uint32_t mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++)
			if (mask & (1u << a)) {
				uint32_t r = svsl_ir_root_ptr(fn, inst->args[a]);
				if (svsl_ir_is_invocation_local(fn, r)) read[r] = 1;
			}
		if (svsl_ir_aux_holds_values(inst))
			for (uint32_t k = 0; k < inst->aux_count; k++) {
				uint32_t r = svsl_ir_root_ptr(fn, fn->aux.items[inst->aux + k]);
				if (svsl_ir_is_invocation_local(fn, r)) read[r] = 1;
			}
	}

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_store) continue;
		uint32_t r = svsl_ir_root_ptr(fn, inst->args[0]);
		// a never-read local, or an undefined value (any memory: leaving the old
		// contents is one of the values undefined allows)
		if ((svsl_ir_is_invocation_local(fn, r) && !read[r]) || fn->insts.items[inst->args[1]].op == svsl_ir_undef)
			svsl_ir_edit_kill(ed, (uint32_t)i);
	}
}
