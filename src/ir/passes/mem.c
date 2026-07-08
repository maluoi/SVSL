// Memory passes: store-to-load forwarding, redundant-load elimination, and
// dead-store elimination. Together they collapse the var/store/load traffic
// that full inlining and out/inout copies generate, plus repeated reads of the
// same local, parameter, or read-only global (cbuffer/pushconstant) location.
//
// Forwarding tracks the value currently at a pointer, keyed by exact pointer id,
// only for *forwardable roots*: function-local vars, entry parameters, and
// read-only globals (uniform/pushconstant buffers, const globals, read-only
// structured buffers). Those are the storages we can reason about soundly:
//   * distinct vars/params never alias (private storage, no escaping pointers);
//   * read-only globals are never written, so their loads are always reusable.
// A per-root generation, bumped on every store to a root, invalidates only the
// entries that share that root (same storage object → may alias; different root
// → cannot).
//
// Forwarding across control flow (dominance). A value carried in a local carries
// past a branch only when it provably dominates the use and cannot be clobbered
// on any path between. Rather than build a dominance tree, we exploit two facts
// of this structured, forward-only-reference IR:
//   * anything established at region-depth 0 (the single top-level path)
//     dominates every later program point, and by the IR's dominance validity a
//     depth-0 store's value operand is itself defined at depth 0 (values that
//     escape a nested region must do so through a var), so it dominates too;
//   * a root that is never stored inside a conditional region (depth > 0) has one
//     unambiguous value at every point — no phi is ever needed to merge it.
// So an entry is *durable* — survives control-flow markers — exactly when its
// root is never conditionally stored AND the entry was created at depth 0. Every
// other entry flushes at each marker, as before (its value may not dominate a
// sibling/outer region). This forwards single-assignment locals and read-only
// globals into branch/loop bodies with no merge machinery. See §4 / §5b.
//
// Writable globals (storage buffers, images, groupshared) are left untouched —
// reasoning about aliasing across duplicate global pointers would need pointer
// canonicalization; the win there is small.

#include "../ir.h"
#include "../ir_operands.h"

// Walk chain bases down to the underlying var/param/ptr/global.
static uint32_t root_ptr(const svsl_ir_func_t *fn, uint32_t p) {
	while (p < (uint32_t)fn->insts.count && fn->insts.items[p].op == svsl_ir_chain)
		p = fn->insts.items[p].args[0];
	return p;
}

static bool is_local_var(const svsl_ir_func_t *fn, uint32_t p) {
	return p < (uint32_t)fn->insts.count && fn->insts.items[p].op == svsl_ir_var;
}

// Region-nesting change contributed by a marker op (0 for else/case/terminators).
static int32_t depth_delta(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_if: case svsl_ir_loop: case svsl_ir_switch:         return +1;
	case svsl_ir_end_if: case svsl_ir_end_loop: case svsl_ir_end_switch: return -1;
	default:                                                          return 0;
	}
}

// A root through which loads/stores can be tracked soundly (see file header).
static bool forwardable_root(const svsl_ir_func_t *fn, const svsl_program_t *prog, uint32_t r) {
	const svsl_ir_inst_t *in = &fn->insts.items[r];
	if (in->op == svsl_ir_var || in->op == svsl_ir_param) return true;
	if (in->op != svsl_ir_ptr) return false;
	switch ((svsl_ref_)in->args[0]) {
	case svsl_ref_buffer_member: {
		const svsl_buffer_t *b = &prog->buffers.items[in->args[1]];
		return b->kind == svsl_block_uniform || b->kind == svsl_block_pushconstant;
	}
	case svsl_ref_const_global:
		return true;
	case svsl_ref_resource:
		return prog->resources.items[in->args[1]].kind == svsl_res_structured; // read-only
	default:
		return false; // storage buffers, images, groupshared, builtins, …
	}
}

static uint32_t resolve(const uint32_t *remap, uint32_t id) {
	while (remap[id] != id) id = remap[id];
	return id;
}

bool svsl_ir_forward(svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_program_t *prog) {
	int32_t count = fn->insts.count;
	if (count == 0) return false;

	// per-pointer entry: table[P] holds the value at *P; e_gen[P] snapshots
	// root_gen[root(P)] (bumped on each store to a root, so stale after a
	// sibling-member store); e_cf[P] the CF generation, 0 = empty. An entry is
	// valid iff its root generation still matches and either it is durable or the
	// CF generation matches. e_dur[P] marks entries whose value dominates all
	// later uses (non-conditional root, established at depth 0) — those survive
	// control-flow markers.
	uint32_t *table    = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t));
	uint32_t *e_gen    = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t));
	uint32_t *e_cf     = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t)); // zeroed = empty
	uint32_t *root_gen = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t)); // zeroed
	uint8_t  *e_dur    = svsl_arena_alloc(arena, (size_t)count);                    // zeroed
	uint8_t  *cond     = svsl_arena_alloc(arena, (size_t)count);                    // zeroed
	uint32_t *remap    = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t));
	for (int32_t i = 0; i < count; i++) remap[i] = (uint32_t)i;

	// pre-scan: a root written inside any conditional region (depth > 0) has an
	// ambiguous value at the region's merge, so its entries can never be durable.
	for (int32_t i = 0, depth = 0; i < count; i++) {
		svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[i].op;
		depth += depth_delta(op);
		if (op != svsl_ir_store && op != svsl_ir_atomic) continue;
		if (depth <= 0) continue;
		uint32_t r = root_ptr(fn, fn->insts.items[i].args[0]);
		if (forwardable_root(fn, prog, r)) cond[r] = 1;
	}

	uint32_t cf      = 1;
	int32_t  depth   = 0;
	bool     changed = false;

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_     op   = (svsl_ir_op_)inst->op;

		if (svsl_ir_ends_run(op)) { depth += depth_delta(op); cf++; continue; } // flush non-durable

		switch (op) {
		case svsl_ir_store: {
			uint32_t p = inst->args[0], r = root_ptr(fn, p);
			if (!forwardable_root(fn, prog, r)) break;
			root_gen[r]++;                                    // invalidate this root's members
			table[p] = resolve(remap, inst->args[1]);
			e_gen[p] = root_gen[r];
			e_cf[p]  = cf;
			e_dur[p] = (!cond[r] && depth == 0);              // value dominates all later uses
			break;
		}
		case svsl_ir_load: {
			uint32_t p = inst->args[0], r = root_ptr(fn, p);
			if (!forwardable_root(fn, prog, r)) break;
			if (e_cf[p] != 0 && e_gen[p] == root_gen[r] && (e_dur[p] || e_cf[p] == cf)) {
				remap[i] = resolve(remap, table[p]);         // store→load or load→load
				changed  = true;
				break;
			}
			// single-index load off a var whose whole value is live and dominates
			if (p != r && is_local_var(fn, r) &&
			    e_cf[r] != 0 && e_gen[r] == root_gen[r] && (e_dur[r] || e_cf[r] == cf)) {
				const svsl_ir_inst_t *ch = &fn->insts.items[p];
				if (ch->op == svsl_ir_chain && ch->args[0] == r && ch->aux_count == 1) {
					uint32_t              idx_id = fn->aux.items[ch->aux];
					const svsl_ir_inst_t *idx    = &fn->insts.items[idx_id];
					uint32_t              v_id   = resolve(remap, table[r]);
					const svsl_ir_inst_t *val    = &fn->insts.items[v_id];
					if (idx->op == svsl_ir_const) {
						if (val->op == svsl_ir_construct && idx->args[0] < val->aux_count) {
							// constant member of a live construct → reuse that
							// component value directly (no new instruction)
							uint32_t comp = fn->aux.items[val->aux + idx->args[0]];
							if (fn->insts.items[comp].type == inst->type) {
								remap[i] = resolve(remap, comp);
								changed  = true;
								break;
							}
						}
						// constant member of any other live composite → static
						// extract, no spill: the var/store/chain die (DCE), emit
						// uses OpCompositeExtract straight off the value
						svsl_type_kind_ vk = svsl_type_get(&prog->types, val->type)->kind;
						if (vk != svsl_type_scalar) {
							inst->op        = svsl_ir_extract;
							inst->args[0]   = v_id;
							inst->args[1]   = idx->args[0];
							inst->aux_count = 0;
							changed         = true;
							break;
						}
					} else if (svsl_type_get(&prog->types, val->type)->kind == svsl_type_vector) {
						// dynamic component of a live vector → extract-dynamic, no spill:
						// the var/store/chain die (DCE), emit uses OpVectorExtractDynamic
						inst->op        = svsl_ir_extract_dynamic;
						inst->args[0]   = v_id;
						inst->args[1]   = resolve(remap, idx_id);
						inst->aux_count = 0;
						changed         = true;
						break;
					}
				}
			}
			table[p] = (uint32_t)i; e_gen[p] = root_gen[r]; e_cf[p] = cf; // record this load
			e_dur[p] = (!cond[r] && depth == 0);
			break;
		}
		case svsl_ir_atomic: {                                // atomic on a local var writes it
			uint32_t r = root_ptr(fn, inst->args[0]);
			if (forwardable_root(fn, prog, r)) root_gen[r]++;
			break;
		}
		default:
			break;
		}
	}

	if (!changed) return false;

	// Apply the remap: forwarded loads become nops, everyone else has its value
	// operands redirected to the resolved definition.
	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (remap[i] != (uint32_t)i) {                 // a forwarded load
			inst->op        = svsl_ir_nop;
			inst->type      = SVSL_TYPE_NONE;
			inst->aux_count = 0;
			continue;
		}
		svsl_ir_remap_operands(fn, (uint32_t)i, remap, (uint32_t)count);
	}
	return true;
}

// Overwriting-store elimination: a store to an exact local pointer whose value
// is overwritten by a later store to the *same* pointer, with no load of that
// pointer's root and no control-flow boundary in between, is unobservable. The
// never-read pass below can't catch this — the root is read elsewhere (e.g. a
// whole-struct `return o` loads every member), so it keeps all member stores;
// but `o.color = a; o.color.rgb *= b;` writes the same member twice with the
// first never observed. Exact-pointer comparison is sound because CSE has
// already merged duplicate address computations to one id, and any intervening
// load of the same root (a whole-var load included) conservatively invalidates.
static bool dse_overwriting(svsl_arena_t *arena, svsl_ir_func_t *fn) {
	int32_t   count   = fn->insts.count;
	uint32_t *last    = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t)); // last[P] = store idx
	uint32_t *pend    = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t)); // pointers awaiting reuse
	uint8_t  *pending = svsl_arena_alloc(arena, (size_t)count);                    // zeroed; in `pend`
	for (int32_t i = 0; i < count; i++) last[i] = SVSL_IR_NONE;

	bool    changed = false;
	int32_t np      = 0;
	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_     op   = (svsl_ir_op_)inst->op;

		if (svsl_ir_ends_run(op)) {                     // region boundary: a store before it may
			for (int32_t k = 0; k < np; k++)            // be observed on another path — flush
				{ last[pend[k]] = SVSL_IR_NONE; pending[pend[k]] = 0; }
			np = 0;
			continue;
		}
		if (op == svsl_ir_store) {
			uint32_t p = inst->args[0];
			if (!is_local_var(fn, root_ptr(fn, p))) continue; // writable globals may be observed
			if (last[p] != SVSL_IR_NONE) {                    // prior store to p, unread → dead
				svsl_ir_inst_t *dead = &fn->insts.items[last[p]];
				dead->op = svsl_ir_nop; dead->type = SVSL_TYPE_NONE; dead->aux_count = 0;
				changed = true;
			}
			last[p] = (uint32_t)i;
			if (!pending[p]) { pending[p] = 1; pend[np++] = p; }
		} else if (op == svsl_ir_load || op == svsl_ir_atomic) {
			uint32_t r = root_ptr(fn, inst->args[0]);       // a read of the root observes any member
			for (int32_t k = 0; k < np; k++)
				if (root_ptr(fn, pend[k]) == r) last[pend[k]] = SVSL_IR_NONE;
		}
	}
	return changed;
}

// Dead-store elimination: a store to a local var that is never read anywhere in
// the function is unobservable. A var is "read" only via a load or atomic (a
// bare chain isn't a read); once forwarding has removed a temp's only load, all
// its stores fall here and DCE then reclaims the var itself. Also runs the
// overwriting-store pass (dead writes that a later write to the same pointer
// shadows), which the never-read test structurally cannot see.
bool svsl_ir_dse(svsl_arena_t *arena, svsl_ir_func_t *fn) {
	int32_t count = fn->insts.count;
	if (count == 0) return false;

	bool changed = dse_overwriting(arena, fn);

	uint8_t *read = svsl_arena_alloc(arena, (size_t)count); // zeroed; read[var] = observed

	for (int32_t i = 0; i < count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
		if (op == svsl_ir_store || op == svsl_ir_chain || op == svsl_ir_var || op == svsl_ir_nop)
			continue; // stores only write; chain/var are not reads on their own

		uint32_t mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++)
			if (mask & (1u << a)) {
				uint32_t r = root_ptr(fn, inst->args[a]);
				if (is_local_var(fn, r)) read[r] = 1;
			}
		if (svsl_ir_aux_holds_values(inst))
			for (uint32_t k = 0; k < inst->aux_count; k++) {
				uint32_t r = root_ptr(fn, fn->aux.items[inst->aux + k]);
				if (is_local_var(fn, r)) read[r] = 1;
			}
	}

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_store) continue;
		uint32_t r = root_ptr(fn, inst->args[0]);
		if (is_local_var(fn, r) && !read[r]) {
			inst->op        = svsl_ir_nop;
			inst->type      = SVSL_TYPE_NONE;
			inst->aux_count = 0;
			changed         = true;
		}
	}
	return changed;
}
