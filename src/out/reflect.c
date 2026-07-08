#include "reflect.h"

#include "../util/array.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef svsl_array_t(char) text_buf_t;

typedef struct printer_t {
	svsl_arena_t *arena;
	text_buf_t    out;
} printer_t;

static void put(printer_t *p, const char *fmt, ...) {
	char    tmp[512];
	va_list args;
	va_start(args, fmt);
	int32_t len = vsnprintf(tmp, sizeof(tmp), fmt, args);
	va_end(args);
	if (len > (int32_t)sizeof(tmp) - 1) len = (int32_t)sizeof(tmp) - 1;
	for (int32_t i = 0; i < len; i++)
		svsl_array_push(p->arena, &p->out, tmp[i]);
}

// prints a member default like "1, 0.5, 0, 1" from the defaults blob
static void put_default(printer_t *p, const svsl_program_t *prog,
                        const svsl_buf_member_t *member, const uint8_t *defaults) {
	const svsl_type_t *t     = svsl_type_get(&prog->types, member->type);
	int32_t            count = t->kind == svsl_type_vector ? t->count :
	                           t->kind == svsl_type_scalar ? 1 : 0;
	if (count == 0) return;

	put(p, " = ");
	for (int32_t i = 0; i < count; i++) {
		if (i) put(p, ", ");
		const uint8_t *at = defaults + member->offset + (uint32_t)i * 4;
		if (t->scalar == svsl_scalar_int32 || t->scalar == svsl_scalar_uint32 || t->scalar == svsl_scalar_bool) {
			int32_t v;
			memcpy(&v, at, 4);
			put(p, "%d", v);
		} else {
			float v;
			memcpy(&v, at, 4);
			put(p, "%g", v);
		}
	}
}

const char *svsl_reflect_print(svsl_arena_t *arena, const svsl_program_t *prog) {
	printer_t p = { .arena = arena };

	put(&p, "shader %.*s\n", prog->name.len, prog->name.ptr);
	if (prog->wave_size) put(&p, "wave_size %d\n", prog->wave_size);

	for (int32_t i = 0; i < prog->entries.count; i++) {
		const svsl_entry_t *e = &prog->entries.items[i];
		const char *stage = e->stage == svsl_stage_vertex ? "vertex" :
		                    e->stage == svsl_stage_pixel  ? "pixel" : "compute";
		put(&p, "entry %-8s %.*s", stage, e->name.len, e->name.ptr);
		if (e->stage == svsl_stage_compute)
			put(&p, " [%d, %d, %d]", e->workgroup[0], e->workgroup[1], e->workgroup[2]);
		put(&p, "\n");
	}

	for (int32_t i = 0; i < prog->buffers.count; i++) {
		const svsl_buffer_t *buf = &prog->buffers.items[i];
		if (buf->kind == svsl_block_storagebuffer) continue; // shown as a resource
		if (buf->kind == svsl_block_pushconstant)
			put(&p, "push %.*s - %u bytes\n", buf->name.len, buf->name.ptr, buf->size);
		else
			put(&p, "buffer %.*s : b%d space%d - %u bytes%s\n", buf->name.len, buf->name.ptr,
			    buf->bind.slot, buf->bind.space, buf->size, buf->defaults ? " (has defaults)" : "");
		for (int32_t m = 0; m < buf->members.count; m++) {
			const svsl_buf_member_t *member = &buf->members.items[m];
			put(&p, "  %-16.*s +%-6u %5ub  %s", member->name.len, member->name.ptr,
			    member->offset, member->size, svsl_type_name(&prog->types, member->type));
			if (member->extra.len) put(&p, " [%.*s]", member->extra.len, member->extra.ptr);
			if (buf->defaults) put_default(&p, prog, member, buf->defaults);
			put(&p, "\n");
		}
	}

	for (int32_t i = 0; i < prog->resources.count; i++) {
		const svsl_resource_t *res  = &prog->resources.items[i];
		static const char     *kinds[] = { "texture", "sampler", "buffer_ro", "buffer_rw", "image", "subpass", "tileimage" };
		if (res->kind == svsl_res_sampler && res->sampler_slot < 0) {
			// paired samplers vanish into their texture; standalone ones still print
			bool paired = false;
			for (int32_t k = 0; k < prog->resources.count; k++)
				if (prog->resources.items[k].kind == svsl_res_texture &&
				    prog->resources.items[k].sampler_slot == res->bind.slot &&
				    prog->resources.items[k].bind.space == res->bind.space) paired = true;
			if (paired) continue;
		}
		put(&p, "resource %-10s %-16.*s : %c%d space%d", kinds[res->kind],
		    res->name.len, res->name.ptr,
		    res->bind.cls ? res->bind.cls : '#', res->bind.slot, res->bind.space);
		if (res->element_size) put(&p, "  %ub/elem", res->element_size);
		if (res->value.len)    put(&p, "  = %.*s", res->value.len, res->value.ptr);
		if (res->tags.len)     put(&p, "  [%.*s]", res->tags.len, res->tags.ptr);
		put(&p, "\n");
	}

	for (int32_t i = 0; i < prog->vertex_inputs.count; i++) {
		const svsl_vertex_input_t *in = &prog->vertex_inputs.items[i];
		put(&p, "input %-10s : %.*s\n", svsl_type_name(&prog->types, in->type),
		    in->semantic.len, in->semantic.ptr);
	}

	for (int32_t i = 0; i < prog->spec_consts.count; i++) {
		const svsl_spec_const_t *sc = &prog->spec_consts.items[i];
		put(&p, "spec %-16.*s : id %u  %s  default 0x%08x\n", sc->name.len, sc->name.ptr,
		    sc->id, svsl_type_name(&prog->types, sc->type), sc->default_bits);
	}

	svsl_array_push(arena, &p.out, '\0');
	return p.out.items;
}
