// Scalar replacement of aggregates (LLVM's SROA): a function-local array or
// struct whose every use names one element through a constant index splits
// into one variable per element. Forwarding then promotes those to values -
// registers instead of the scratch memory an addressable array lives in on
// Adreno. Unrolling (unroll.c) is what turns loop-indexed arrays into
// constant-indexed ones; arrays that stay dynamically indexed are left alone.
// Entry parameters are struct locals filled from the stage inputs (ir_build's
// lower_entry_io), so this is also what reduces them to the input loads.
//
// Accepted uses of the variable V:
//   * chain(V, [k, rest...]) with k constant and in range: becomes V_k, or
//     chain(V_k, [rest...]) - nested aggregates split one level per round;
//   * store V, x (an initializer, a copy): one store of extract(x, k) per element;
//   * load V (a copy out): a construct of the element loads.
// Anything else (a dynamic first index, an atomic on V itself) keeps V whole.

#include "passes.h"
#include "../ir_operands.h"

#define SROA_MAX_ELEMENTS 256

// element count of an array or struct type (0 for anything else), and element k's type
static int32_t aggregate_count(const svsl_types_t *types, svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(types, type);
	return t->kind == svsl_type_array  ? t->array_count
	     : t->kind == svsl_type_struct ? types->structs.items[t->struct_index].members.count : 0;
}
static svsl_type_id_t element_type(const svsl_types_t *types, svsl_type_id_t type, int32_t k) {
	const svsl_type_t *t = svsl_type_get(types, type);
	return t->kind == svsl_type_array ? t->elem : types->structs.items[t->struct_index].members.items[k].type;
}

void svsl_ir_sroa(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t     *fn    = ed->fn;
	const svsl_types_t *types = &prog->types;
	uint32_t            count = (uint32_t)fn->insts.count;

	bool any = false; // most functions have no local aggregates: skip the scans
	for (uint32_t i = 0; i < count && !any; i++)
		any = fn->insts.items[i].op == svsl_ir_var && aggregate_count(types, fn->insts.items[i].type) > 0;
	if (!any) return;

	// which vars are splittable: everything that names them must be an accepted use
	uint8_t *ok = svsl_arena_alloc(ed->scratch, (size_t)(count > 0 ? count : 1)); // zeroed
	for (uint32_t i = 0; i < count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != svsl_ir_var) continue;
		int32_t n = aggregate_count(types, in->type);
		ok[i] = n > 0 && n <= SROA_MAX_ELEMENTS;
	}
	for (uint32_t i = 0; i < count; i++) {
		const svsl_ir_inst_t *in   = &fn->insts.items[i];
		uint32_t              mask = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++) {
			if (!(mask & (1u << a))) continue;
			uint32_t v = in->args[a];
			if (v >= count || !ok[v]) continue;
			bool accepted = false;
			if (in->op == svsl_ir_chain && a == 0 && in->aux_count > 0) {
				const svsl_ir_inst_t *k = &fn->insts.items[fn->aux.items[in->aux]];
				accepted = k->op == svsl_ir_const &&
				           k->args[1] == 0 && k->args[0] < (uint32_t)aggregate_count(types, fn->insts.items[v].type);
			} else if (in->op == svsl_ir_store && a == 0) {
				accepted = in->args[1] != v;
			} else if (in->op == svsl_ir_load) {
				accepted = true;
			}
			if (!accepted) ok[v] = 0;
		}
		if (svsl_ir_aux_holds_values(in)) // V inside an operand list (an index, an asm operand...)
			for (uint32_t k = 0; k < in->aux_count; k++) {
				uint32_t v = fn->aux.items[in->aux + k];
				if (v < count) ok[v] = 0;
			}
	}

	// the element variables, declared where V was
	uint32_t *first = svsl_arena_alloc(ed->scratch, (size_t)(count > 0 ? count : 1) * sizeof(uint32_t));
	for (uint32_t i = 0; i < count; i++) {
		if (!ok[i]) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		int32_t               n  = aggregate_count(types, in->type);
		for (int32_t k = 0; k < n; k++) {
			uint32_t id = svsl_ir_edit_insert(ed, i, (svsl_ir_inst_t){
				.op = svsl_ir_var, .type = element_type(types, in->type, k),
				.args = { 0, 0, 0, SVSL_IR_NONE }, .loc = in->loc, .name = in->name }, NULL, 0);
			if (k == 0) first[i] = id; // provisional ids of one var's elements are consecutive
		}
		svsl_ir_edit_kill(ed, i);
	}

	// rewrite the uses onto them
	for (uint32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op == svsl_ir_chain && in->args[0] < count && ok[in->args[0]]) {
			uint32_t v    = in->args[0];
			uint32_t elem = first[v] + fn->insts.items[fn->aux.items[in->aux]].args[0];
			if (in->aux_count == 1) {
				svsl_ir_edit_replace(ed, i, elem);   // the element itself
			} else {
				in->args[0] = elem;                 // the rest of the path, from the element
				in->aux++;
				in->aux_count--;
				svsl_ir_edit_touch(ed);
			}
		} else if (in->op == svsl_ir_store && in->args[0] < count && ok[in->args[0]]) {
			uint32_t       v = in->args[0], x = in->args[1];
			svsl_type_id_t t = fn->insts.items[v].type;
			for (int32_t k = 0; k < aggregate_count(types, t); k++) {
				uint32_t part = svsl_ir_edit_insert(ed, i, (svsl_ir_inst_t){
					.op = svsl_ir_extract, .type = element_type(types, t, k),
					.args = { x, (uint32_t)k, 0, SVSL_IR_NONE }, .loc = in->loc }, NULL, 0);
				svsl_ir_edit_insert(ed, i, (svsl_ir_inst_t){ .op = svsl_ir_store, .type = SVSL_TYPE_NONE,
					.args = { first[v] + (uint32_t)k, part, 0, SVSL_IR_NONE }, .loc = in->loc }, NULL, 0);
			}
			svsl_ir_edit_kill(ed, i);
		} else if (in->op == svsl_ir_load && in->args[0] < count && ok[in->args[0]]) {
			uint32_t       v = in->args[0];
			svsl_type_id_t t = fn->insts.items[v].type;
			int32_t        n = aggregate_count(types, t);
			uint32_t      *parts = svsl_arena_alloc(ed->scratch, (size_t)n * sizeof(uint32_t));
			for (int32_t k = 0; k < n; k++)
				parts[k] = svsl_ir_edit_insert(ed, i, (svsl_ir_inst_t){
					.op = svsl_ir_load, .type = element_type(types, t, k),
					.args = { first[v] + (uint32_t)k, 0, 0, SVSL_IR_NONE }, .loc = in->loc }, NULL, 0);
			svsl_ir_edit_replace(ed, i, svsl_ir_edit_insert(ed, i, (svsl_ir_inst_t){
				.op = svsl_ir_construct, .type = t, .args = { 0, 0, 0, SVSL_IR_NONE }, .loc = in->loc },
				parts, (uint32_t)n));
		}
	}
}
