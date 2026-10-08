// See ir_edit.h.

#include "ir_edit.h"
#include "ir_operands.h"

#include <assert.h>
#include <string.h>

void svsl_ir_edit_begin(svsl_ir_edit_t *ed, svsl_arena_t *arena, svsl_arena_t *scratch,
                        svsl_ir_func_t *fn) {
	uint32_t base = (uint32_t)fn->insts.count;
	*ed = (svsl_ir_edit_t){ .fn = fn, .arena = arena, .scratch = scratch, .base = base, .lowest = base };
	ed->repl = svsl_arena_alloc(scratch, (size_t)(base > 0 ? base : 1) * sizeof(uint32_t)); // zeroed: all kept
}

void svsl_ir_edit_resolve_operands(svsl_ir_edit_t *ed, uint32_t id) {
	svsl_ir_inst_t *inst  = svsl_ir_edit_inst(ed, id);
	uint32_t        mask  = svsl_ir_value_arg_mask(inst);
	uint32_t        limit = ed->base + (uint32_t)ed->added.count;
	for (int32_t a = 0; a < 4; a++) {
		if (!(mask & (1u << a)) || inst->args[a] >= limit) continue;
		uint32_t r = svsl_ir_edit_resolve(ed, inst->args[a]);
		if (r != SVSL_IR_NONE) inst->args[a] = r;
	}
	if (!svsl_ir_aux_holds_values(inst)) return;
	for (uint32_t k = 0; k < inst->aux_count; k++) {
		uint32_t *slot = &ed->fn->aux.items[inst->aux + k];
		if (*slot >= limit) continue;
		uint32_t r = svsl_ir_edit_resolve(ed, *slot);
		if (r != SVSL_IR_NONE) *slot = r;
	}
}

void svsl_ir_edit_replace(svsl_ir_edit_t *ed, uint32_t id, uint32_t with) {
	if (ed->repl[id] == 0) svsl_array_push(ed->scratch, &ed->dead, id);
	ed->repl[id] = with + 1;
	ed->changed  = true;
	ed->replaced = true;
	if (id < ed->lowest) ed->lowest = id;
}

void svsl_ir_edit_kill(svsl_ir_edit_t *ed, uint32_t id) {
	if (ed->repl[id] == 0) svsl_array_push(ed->scratch, &ed->dead, id);
	ed->repl[id] = SVSL_IR_NONE;
	ed->changed  = true;
	if (id < ed->lowest) ed->lowest = id;
}

void svsl_ir_edit_touch(svsl_ir_edit_t *ed) {
	ed->changed = true;
}

uint32_t svsl_ir_edit_insert(svsl_ir_edit_t *ed, uint32_t before, svsl_ir_inst_t inst,
                             const uint32_t *opt_aux, uint32_t aux_count) {
	svsl_ir_func_t *fn = ed->fn;
	if (opt_aux && aux_count && opt_aux >= fn->aux.items && opt_aux < fn->aux.items + fn->aux.count) {
		// copying an existing operand list: the pushes below may move the pool
		uint32_t *copy = svsl_arena_alloc(ed->scratch, (size_t)aux_count * sizeof(uint32_t));
		memcpy(copy, opt_aux, (size_t)aux_count * sizeof(uint32_t));
		opt_aux = copy;
	}
	inst.aux       = (uint32_t)fn->aux.count;
	inst.aux_count = opt_aux ? aux_count : 0;
	for (uint32_t k = 0; k < inst.aux_count; k++)
		svsl_array_push(ed->arena, &fn->aux, opt_aux[k]);
	svsl_array_push(ed->scratch, &ed->added, inst);
	svsl_array_push(ed->scratch, &ed->anchor, before);
	ed->changed = true;
	return ed->base + (uint32_t)ed->added.count - 1;
}

static uint32_t const_hash(svsl_type_id_t type, uint64_t bits) {
	uint64_t h = (bits ^ ((uint64_t)(uint32_t)type << 40)) * 0x9E3779B97F4A7C15ull;
	return (uint32_t)(h >> 32);
}

uint32_t svsl_ir_edit_const(svsl_ir_edit_t *ed, svsl_type_id_t type, uint64_t bits) {
	if (ed->const_cap == 0 || (uint32_t)ed->added.count * 2 >= ed->const_cap) {
		// (re)build the table over every constant inserted so far
		uint32_t cap = ed->const_cap ? ed->const_cap * 2 : 64;
		while (cap < (uint32_t)ed->added.count * 4) cap *= 2;
		ed->const_slot = svsl_arena_alloc(ed->scratch, (size_t)cap * sizeof(uint32_t)); // zeroed = empty
		ed->const_cap  = cap;
		for (int32_t k = 0; k < ed->added.count; k++) {
			const svsl_ir_inst_t *c = &ed->added.items[k];
			if (c->op != svsl_ir_const || ed->anchor.items[k] != 0) continue;
			uint64_t cb = (uint64_t)c->args[0] | ((uint64_t)c->args[1] << 32);
			for (uint32_t s = const_hash(c->type, cb) & (cap - 1);; s = (s + 1) & (cap - 1))
				if (!ed->const_slot[s]) { ed->const_slot[s] = (uint32_t)k + 1; break; }
		}
	}
	uint32_t mask = ed->const_cap - 1;
	uint32_t s    = const_hash(type, bits) & mask;
	for (;; s = (s + 1) & mask) {
		uint32_t k = ed->const_slot[s];
		if (!k) break;
		const svsl_ir_inst_t *c = &ed->added.items[k - 1];
		if (c->type == type && c->args[0] == (uint32_t)bits && c->args[1] == (uint32_t)(bits >> 32))
			return ed->base + k - 1;
	}
	uint32_t id = svsl_ir_edit_insert(ed, 0, (svsl_ir_inst_t){
		.op = svsl_ir_const, .type = type,
		.args = { (uint32_t)bits, (uint32_t)(bits >> 32), 0, SVSL_IR_NONE } }, NULL, 0);
	ed->const_slot[s] = id - ed->base + 1;
	return id;
}

// Where an operand ends up: `final_of` maps every old/provisional id through
// its replacements to its committed position (NONE when it died).
static uint32_t final_id(const uint32_t *final_of, uint32_t total, uint32_t v) {
	return v < total ? final_of[v] : v; // >= total: not an id (defensive)
}

// The in-place commit: no inserts, so positions don't change. Killed and
// replaced instructions become nops; survivors' operands are redirected.
// Nothing below `lowest` can name a changed id, so the walk starts there - and
// when no operand needs redirecting, only the dead ids are visited.
static void nop(svsl_ir_inst_t *inst) {
	inst->op        = svsl_ir_nop;
	inst->type      = SVSL_TYPE_NONE;
	inst->aux_count = 0;
}

static void commit_in_place(svsl_ir_edit_t *ed) {
	svsl_ir_func_t *fn = ed->fn;
	if (!ed->replaced || ed->resolved) {
		for (int32_t k = 0; k < ed->dead.count; k++) nop(&fn->insts.items[ed->dead.items[k]]);
		return;
	}
	for (uint32_t i = ed->lowest; i < ed->base; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (ed->repl[i] != 0) { nop(inst); continue; }
		if (inst->op == svsl_ir_nop) continue;
		uint32_t mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++) {
			if (!(mask & (1u << a)) || inst->args[a] >= ed->base) continue;
			inst->args[a] = svsl_ir_edit_resolve(ed, inst->args[a]);
			assert(inst->args[a] < i && "IR edit: operand must be defined earlier and survive");
		}
		if (!svsl_ir_aux_holds_values(inst)) continue;
		for (uint32_t k = 0; k < inst->aux_count; k++) {
			uint32_t *slot = &fn->aux.items[inst->aux + k];
			if (*slot >= ed->base) continue;
			*slot = svsl_ir_edit_resolve(ed, *slot);
			assert(*slot < i && "IR edit: operand must be defined earlier and survive");
		}
	}
}

// Copy one instruction to its final slot: operands renumbered, aux re-packed
// at *ref_aux_at in the new pool.
static void place(const uint32_t *final_of, uint32_t total,
                  const svsl_ir_inst_t *src, uint32_t at, svsl_ir_inst_t *out,
                  const uint32_t *old_aux, uint32_t *new_aux, uint32_t *ref_aux_at) {
	svsl_ir_inst_t c    = *src;
	uint32_t       mask = svsl_ir_value_arg_mask(&c);
	for (int32_t a = 0; a < 4; a++) {
		if (!(mask & (1u << a))) continue;
		c.args[a] = final_id(final_of, total, c.args[a]);
		assert(c.args[a] < at && "IR edit: operand must be defined earlier and survive");
	}
	bool values = svsl_ir_aux_holds_values(&c);
	for (uint32_t k = 0; k < c.aux_count; k++) {
		uint32_t v = old_aux[c.aux + k];
		if (values) {
			v = final_id(final_of, total, v);
			assert(v < at && "IR edit: operand must be defined earlier and survive");
		}
		new_aux[*ref_aux_at + k] = v;
	}
	c.aux        = *ref_aux_at;
	*ref_aux_at += c.aux_count;
	out[at] = c;
}

// Copies count items into *ref_items, growing it (with headroom: unrolling keeps
// growing a function) only when it's too small.
static void copy_back(svsl_arena_t *arena, void **ref_items, int32_t *ref_capacity, const void *src,
                      uint32_t count, size_t item_size) {
	if ((uint32_t)*ref_capacity < count) {
		uint32_t capacity = count + count / 2 + 16;
		*ref_items        = svsl_arena_alloc_raw(arena, (size_t)capacity * item_size);
		*ref_capacity     = (int32_t)capacity;
	}
	if (count) memcpy(*ref_items, src, (size_t)count * item_size);
}

bool svsl_ir_edit_commit(svsl_ir_edit_t *ed, bool compact) {
	svsl_ir_func_t *fn    = ed->fn;
	uint32_t        base  = ed->base;
	uint32_t        nadd  = (uint32_t)ed->added.count;
	uint32_t        total = base + nadd;
	if (nadd == 0 && !compact) {
		if (ed->changed) commit_in_place(ed);
		return ed->changed;
	}

	// bucket the inserts by anchor, stable (counting sort; anchor base = append)
	uint32_t *start = svsl_arena_alloc(ed->scratch, (size_t)(base + 2) * sizeof(uint32_t));
	uint32_t *order = svsl_arena_alloc_raw(ed->scratch, (size_t)(nadd > 0 ? nadd : 1) * sizeof(uint32_t));
	for (uint32_t k = 0; k < nadd; k++) start[ed->anchor.items[k] + 1]++;
	for (uint32_t i = 0; i <= base; i++) start[i + 1] += start[i];
	{
		uint32_t *fill = svsl_arena_alloc(ed->scratch, (size_t)(base + 1) * sizeof(uint32_t));
		for (uint32_t k = 0; k < nadd; k++) {
			uint32_t a = ed->anchor.items[k];
			order[start[a] + fill[a]++] = k;
		}
	}

	// final positions: inserts first, then the anchor itself if it survives
	uint32_t *newpos = svsl_arena_alloc_raw(ed->scratch, (size_t)(total > 0 ? total : 1) * sizeof(uint32_t));
	uint32_t  count  = 0;
	for (uint32_t i = 0; i <= base; i++) {
		for (uint32_t j = start[i]; j < start[i + 1]; j++) newpos[base + order[j]] = count++;
		if (i == base) break;
		bool live = ed->repl[i] == 0 && fn->insts.items[i].op != svsl_ir_nop;
		newpos[i] = live ? count++ : SVSL_IR_NONE;
	}
	uint32_t *final_of = svsl_arena_alloc_raw(ed->scratch, (size_t)(total > 0 ? total : 1) * sizeof(uint32_t));
	for (uint32_t v = 0; v < total; v++) {
		uint32_t r = svsl_ir_edit_resolve(ed, v);
		final_of[v] = r == SVSL_IR_NONE ? SVSL_IR_NONE : newpos[r];
	}

	// Without inserts the dense rebuild runs in place (every slot moves down or
	// stays); with inserts it builds in scratch and copies back. The aux pool is
	// re-packed either way, dropping dead operand lists. Copying back reuses the
	// function's storage, so a commit doesn't touch fresh memory unless the
	// function outgrew it (page faults, not copies, are what large shaders pay).
	uint32_t aux_total = 0;
	for (uint32_t k = 0; k < nadd; k++) aux_total += ed->added.items[k].aux_count;
	for (uint32_t i = 0; i < base; i++)
		if (newpos[i] != SVSL_IR_NONE) aux_total += fn->insts.items[i].aux_count;
	uint32_t       *new_aux = svsl_arena_alloc_raw(ed->scratch, (size_t)(aux_total > 0 ? aux_total : 1) * sizeof(uint32_t));
	const uint32_t *old_aux = fn->aux.items;
	uint32_t        aux_at  = 0;
	svsl_ir_inst_t *out     = nadd ? svsl_arena_alloc_raw(ed->scratch, (size_t)(count > 0 ? count : 1) * sizeof(svsl_ir_inst_t))
	                               : fn->insts.items;
	for (uint32_t i = 0; i <= base; i++) {
		for (uint32_t j = start[i]; j < start[i + 1]; j++) {
			uint32_t k = order[j];
			place(final_of, total, &ed->added.items[k], newpos[base + k], out, old_aux, new_aux, &aux_at);
		}
		if (i == base) break;
		if (newpos[i] == SVSL_IR_NONE) continue;
		svsl_ir_inst_t src = fn->insts.items[i]; // copy first: in place, the slot may be its own target
		place(final_of, total, &src, newpos[i], out, old_aux, new_aux, &aux_at);
	}

	if (nadd) copy_back(ed->arena, (void **)&fn->insts.items, &fn->insts.capacity, out, count, sizeof(svsl_ir_inst_t));
	fn->insts.count = (int32_t)count;
	copy_back(ed->arena, (void **)&fn->aux.items, &fn->aux.capacity, new_aux, aux_total, sizeof(uint32_t));
	fn->aux.count = (int32_t)aux_total;
	return ed->changed;
}
