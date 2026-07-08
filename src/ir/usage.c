#include "usage.h"

#include "../util/array.h"

// ORs `stage` into a resource's slot plus its block-form buffer twin and any fused
// sampler. Both callers pass stage-mask arrays; the per-function pass sets a single
// "used" bit, so a non-zero slot means used either way.
static void mark_resource(const svsl_program_t *prog, uint8_t *res_stages, uint8_t *buf_stages,
                          uint32_t index, uint8_t stage) {
	if (index >= (uint32_t)prog->resources.count) return;
	res_stages[index] |= stage;
	const svsl_resource_t *res = &prog->resources.items[index];
	if (res->buffer_index >= 0) // block-form storage buffers live in both tables
		buf_stages[res->buffer_index] |= stage;
	// a texture's fused sampler counts as used with it
	if (res->kind == svsl_res_texture && res->sampler_slot >= 0)
		for (int32_t k = 0; k < prog->resources.count; k++) {
			const svsl_resource_t *smp = &prog->resources.items[k];
			if (smp->kind == svsl_res_sampler && smp->bind.slot == res->sampler_slot &&
			    smp->bind.space == res->bind.space)
				res_stages[k] |= stage;
		}
}

// member access reaches a buffer directly; block-form storage buffers must
// light up their resource-table twin too, or reflection drops the binding
static void mark_buffer(const svsl_program_t *prog, uint8_t *res_stages, uint8_t *buf_stages,
                        uint32_t index, uint8_t stage) {
	if (index >= (uint32_t)prog->buffers.count) return;
	buf_stages[index] |= stage;
	for (int32_t k = 0; k < prog->resources.count; k++)
		if (prog->resources.items[k].buffer_index == (int32_t)index)
			res_stages[k] |= stage;
}

// the shared resource/buffer walk: every ptr/tex/image op that names a binding
// ORs `stage` into it. The one place the two usage passes must stay in sync.
static void mark_func_resources(const svsl_program_t *prog, const svsl_ir_func_t *fn,
                                uint8_t *res_stages, uint8_t *buf_stages, uint8_t stage) {
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		switch ((svsl_ir_op_)inst->op) {
		case svsl_ir_ptr:
			if (inst->args[0] == svsl_ref_buffer_member)
				mark_buffer(prog, res_stages, buf_stages, inst->args[1], stage);
			else if (inst->args[0] == svsl_ref_resource)
				mark_resource(prog, res_stages, buf_stages, inst->args[1], stage);
			break;
		case svsl_ir_tex:
			mark_resource(prog, res_stages, buf_stages, inst->args[0], stage);
			if (inst->args[1] != SVSL_IR_NONE)
				mark_resource(prog, res_stages, buf_stages, inst->args[1], stage);
			break;
		case svsl_ir_image_load:
		case svsl_ir_image_store:
		case svsl_ir_image_atomic:
			mark_resource(prog, res_stages, buf_stages, inst->args[0], stage);
			break;
		default:
			break;
		}
	}
}

void svsl_ir_func_globals(const svsl_program_t *prog, const svsl_ir_func_t *fn,
                          uint8_t *out_buffer_used, uint8_t *out_resource_used) {
	for (int32_t i = 0; i < prog->buffers.count; i++)   out_buffer_used[i]   = 0;
	for (int32_t i = 0; i < prog->resources.count; i++) out_resource_used[i] = 0;
	mark_func_resources(prog, fn, out_resource_used, out_buffer_used, 1); // any non-zero = used
}

// maps a vertex-entry (param, member) pair to its index in prog->vertex_inputs,
// which was collected in flattened declaration order excluding builtins
static int32_t vs_input_index(const svsl_program_t *prog, const svsl_entry_t *entry,
                              int32_t param, int32_t member) {
	const svsl_func_info_t *info = svsl_program_func_info(prog, entry->func);
	if (!info) return -1;

	int32_t index = 0;
	for (int32_t p = 0; p < entry->func->param_count; p++) {
		const svsl_type_t *pt = svsl_type_get(&prog->types, info->param_types[p]);
		if (pt->kind == svsl_type_struct) {
			const svsl_struct_info_t *si = &prog->types.structs.items[pt->struct_index];
			for (int32_t m = 0; m < si->members.count; m++) {
				bool counted = index < prog->vertex_inputs.count &&
				               svsl_str_eq(prog->vertex_inputs.items[index].name,
				                           si->members.items[m].name);
				if (p == param && m == member) return counted ? index : -1;
				if (counted) index++;
			}
		} else {
			bool counted = index < prog->vertex_inputs.count &&
			               svsl_str_eq(prog->vertex_inputs.items[index].name,
			                           entry->func->params[p]->name);
			if (p == param && member < 0) return counted ? index : -1;
			if (counted) index++;
		}
	}
	return -1;
}

void svsl_ir_analyze_usage(svsl_arena_t *arena, const svsl_program_t *prog,
                           const svsl_ir_module_t *module, svsl_usage_t *out_usage) {
	out_usage->buffer_stages   = svsl_arena_alloc(arena, (size_t)(prog->buffers.count > 0 ? prog->buffers.count : 1));
	out_usage->resource_stages = svsl_arena_alloc(arena, (size_t)(prog->resources.count > 0 ? prog->resources.count : 1));
	out_usage->spec_stages     = svsl_arena_alloc(arena, (size_t)(prog->spec_consts.count > 0 ? prog->spec_consts.count : 1));
	out_usage->vs_input_used   = svsl_arena_alloc(arena, (size_t)(prog->vertex_inputs.count > 0 ? prog->vertex_inputs.count : 1));

	for (int32_t f = 0; f < module->func_count; f++) {
		const svsl_ir_func_t *fn    = &module->funcs[f];
		uint8_t               stage = (uint8_t)fn->entry->stage;
		bool                  is_vs = fn->entry->stage == svsl_stage_vertex;

		mark_func_resources(prog, fn, out_usage->resource_stages, out_usage->buffer_stages, stage);

		for (int32_t i = 0; i < fn->insts.count; i++) {
			const svsl_ir_inst_t *inst = &fn->insts.items[i];
			switch ((svsl_ir_op_)inst->op) {
			case svsl_ir_spec_const:
				if (inst->args[0] < (uint32_t)prog->spec_consts.count)
					out_usage->spec_stages[inst->args[0]] |= stage;
				break;
			case svsl_ir_chain:
				if (is_vs) { // member access into an entry input struct
					const svsl_ir_inst_t *base = &fn->insts.items[inst->args[0]];
					if (base->op == svsl_ir_param && inst->aux_count > 0) {
						const svsl_ir_inst_t *idx = &fn->insts.items[fn->aux.items[inst->aux]];
						if (idx->op == svsl_ir_const) {
							int32_t vi = vs_input_index(prog, fn->entry,
							                            (int32_t)base->args[0], (int32_t)idx->args[0]);
							if (vi >= 0) out_usage->vs_input_used[vi] = 1;
						}
					}
				}
				break;
			case svsl_ir_load:
				if (is_vs) { // whole-struct read marks every member of that param
					const svsl_ir_inst_t *base = &fn->insts.items[inst->args[0]];
					if (base->op == svsl_ir_param) {
						const svsl_type_t *pt = svsl_type_get(&prog->types, base->type);
						if (pt->kind != svsl_type_struct) {
							int32_t vi = vs_input_index(prog, fn->entry, (int32_t)base->args[0], -1);
							if (vi >= 0) out_usage->vs_input_used[vi] = 1;
						} else {
							const svsl_struct_info_t *si = &prog->types.structs.items[pt->struct_index];
							for (int32_t m = 0; m < si->members.count; m++) {
								int32_t vi = vs_input_index(prog, fn->entry, (int32_t)base->args[0], m);
								if (vi >= 0) out_usage->vs_input_used[vi] = 1;
							}
						}
					}
				}
				break;
			default:
				break;
			}
		}
	}
}
