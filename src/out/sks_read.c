// SKS container reader (<svsl/svsl.h> svsl_sks_parse). Byte layout mirrors the
// writer in sks_write.c exactly. Everything the parsed view points at is copied
// into an arena the returned handle owns, so the caller's input buffer can go
// away immediately. Bounds-checked throughout; malformed input returns NULL.

#include <svsl/svsl.h>

#include "../util/arena.h"

#include <stdlib.h>
#include <string.h>

typedef struct sks_impl_t {
	svsl_arena_t    arena;
	svsl_sks_file_t file;
} sks_impl_t;

typedef struct rd_t {
	const uint8_t *data;
	int32_t        size;
	int32_t        pos;
	svsl_arena_t  *arena;
	bool           bad;
} rd_t;

// advances past `n` bytes and returns a pointer to them, or NULL (and marks bad)
// if that would run off the end
static const uint8_t *rd_take(rd_t *r, int32_t n) {
	if (r->bad || n < 0 || n > r->size - r->pos) { r->bad = true; return NULL; }
	const uint8_t *p = r->data + r->pos;
	r->pos += n;
	return p;
}
static uint8_t  rd_u8 (rd_t *r) { const uint8_t *p = rd_take(r, 1); return p ? p[0] : 0; }
static uint16_t rd_u16(rd_t *r) { const uint8_t *p = rd_take(r, 2); uint16_t v = 0; if (p) memcpy(&v, p, 2); return v; }
static uint32_t rd_u32(rd_t *r) { const uint8_t *p = rd_take(r, 4); uint32_t v = 0; if (p) memcpy(&v, p, 4); return v; }
static int32_t  rd_i32(rd_t *r) { const uint8_t *p = rd_take(r, 4); int32_t  v = 0; if (p) memcpy(&v, p, 4); return v; }
static uint64_t rd_u64(rd_t *r) { const uint8_t *p = rd_take(r, 8); uint64_t v = 0; if (p) memcpy(&v, p, 8); return v; }

// a fixed-size zero-padded field → an arena-owned NUL-terminated string
static const char *rd_str(rd_t *r, int32_t field) {
	const uint8_t *p = rd_take(r, field);
	if (!p) return "";
	int32_t len = 0;
	while (len < field && p[len]) len++;
	return svsl_arena_strndup(r->arena, (const char *)p, (size_t)len);
}

// a byte span copied into the arena (defaults / SPIR-V payloads)
static const uint8_t *rd_blob(rd_t *r, int32_t n) {
	const uint8_t *p = rd_take(r, n);
	if (!p || n <= 0) return NULL;
	uint8_t *copy = svsl_arena_alloc(r->arena, (size_t)n);
	memcpy(copy, p, (size_t)n);
	return copy;
}

// a count is trustworthy only if each element needs at least one byte, so it can
// never exceed the bytes left; reject anything larger before we allocate for it
static bool count_ok(const rd_t *r, int32_t count) {
	return count >= 0 && count <= r->size - r->pos;
}

#define ALLOC_N(T, n) ((T *)svsl_arena_alloc(r.arena, sizeof(T) * (size_t)((n) > 0 ? (n) : 1)))

svsl_sks_file_t *svsl_sks_parse(const void *bytes, int32_t size) {
	if (!bytes || size < 8) return NULL;

	sks_impl_t *impl = calloc(1, sizeof(sks_impl_t));
	if (!impl) return NULL;
	rd_t             r    = { .data = bytes, .size = size, .arena = &impl->arena };
	svsl_sks_file_t *file = &impl->file;

	const uint8_t *tag = rd_take(&r, 8);
	if (!tag || memcmp(tag, "SKSHADER", 8) != 0) goto fail;

	file->version         = rd_u16(&r);
	int32_t stage_count   = (int32_t)rd_u32(&r);
	file->name            = rd_str(&r, 256);
	int32_t buffer_count  = (int32_t)rd_u32(&r);
	int32_t resource_count= (int32_t)rd_u32(&r);
	int32_t vin_count     = rd_i32(&r);
	int32_t spec_count    = (int32_t)rd_u32(&r);
	file->features        = rd_u64(&r);
	rd_u64(&r);            // reserved features word
	rd_take(&r, 24);       // op counts (vertex/pixel total/tex/flow)
	file->wave_size       = rd_u32(&r);
	file->tile_apron[0]   = rd_u32(&r); // v11: //--apron
	file->tile_apron[1]   = rd_u32(&r);
	if (r.bad || file->version != SVSL_SKS_VERSION) goto fail;
	if (!count_ok(&r, buffer_count) || !count_ok(&r, resource_count) ||
	    !count_ok(&r, vin_count) || !count_ok(&r, spec_count) || !count_ok(&r, stage_count))
		goto fail;

	// --- buffers ---
	svsl_sks_buffer_t *buffers = ALLOC_N(svsl_sks_buffer_t, buffer_count);
	for (int32_t b = 0; b < buffer_count; b++) {
		svsl_sks_buffer_t *buf = &buffers[b];
		buf->name          = rd_str(&r, 32);
		buf->space         = rd_u8(&r);
		buf->slot          = rd_u16(&r);
		buf->stage_bits    = rd_u8(&r);
		buf->register_type = rd_u8(&r);
		buf->size          = rd_u32(&r);
		int32_t var_count  = (int32_t)rd_u32(&r);
		buf->defaults_size = rd_u32(&r);
		buf->defaults      = rd_blob(&r, (int32_t)buf->defaults_size);
		if (r.bad || !count_ok(&r, var_count)) goto fail;
		svsl_sks_var_t *vars = ALLOC_N(svsl_sks_var_t, var_count);
		for (int32_t v = 0; v < var_count; v++) {
			vars[v].name       = rd_str(&r, 32);
			vars[v].extra      = rd_str(&r, 64);
			vars[v].type_name  = rd_str(&r, 32);
			vars[v].offset     = rd_u32(&r);
			vars[v].size       = rd_u32(&r);
			vars[v].type       = rd_u16(&r);
			vars[v].type_count = rd_u16(&r);
		}
		buf->vars      = vars;
		buf->var_count = var_count;
		if (r.bad) goto fail;
	}

	// --- vertex inputs ---
	svsl_sks_vertex_input_t *vins = ALLOC_N(svsl_sks_vertex_input_t, vin_count);
	for (int32_t v = 0; v < vin_count; v++) {
		vins[v].format   = rd_i32(&r);
		vins[v].count    = rd_u8(&r);
		vins[v].semantic = rd_i32(&r);
		vins[v].slot     = rd_u8(&r);
		vins[v].location = rd_u8(&r);
	}

	// --- resources ---
	svsl_sks_resource_t *resources = ALLOC_N(svsl_sks_resource_t, resource_count);
	for (int32_t i = 0; i < resource_count; i++) {
		svsl_sks_resource_t *res = &resources[i];
		res->name          = rd_str(&r, 32);
		res->value         = rd_str(&r, 64);
		res->tags          = rd_str(&r, 64);
		res->slot          = rd_u16(&r);
		res->stage_bits    = rd_u8(&r);
		res->register_type = rd_u8(&r);
		res->element_size  = rd_u32(&r);
		res->shape         = rd_u8(&r);
		res->image_format  = rd_u8(&r);
		rd_u16(&r); // reserved
	}

	// --- specialization constants ---
	svsl_sks_spec_t *specs = ALLOC_N(svsl_sks_spec_t, spec_count);
	for (int32_t i = 0; i < spec_count; i++) {
		specs[i].name         = rd_str(&r, 32);
		specs[i].id           = rd_u32(&r);
		specs[i].default_bits = rd_u32(&r);
		specs[i].type         = rd_u16(&r);
		specs[i].stage_bits   = rd_u8(&r);
	}

	// --- stages (per-entry SPIR-V) ---
	svsl_sks_stage_t *stages = ALLOC_N(svsl_sks_stage_t, stage_count);
	for (int32_t i = 0; i < stage_count; i++) {
		rd_i32(&r);                                   // language tag (skr_shader_lang_spirv)
		stages[i].stage     = (svsl_stage_)rd_i32(&r);
		stages[i].wave_size = rd_u32(&r);
		int32_t spirv_bytes = (int32_t)rd_u32(&r);
		if (r.bad || spirv_bytes < 0 || (spirv_bytes & 3) || spirv_bytes > r.size - r.pos) goto fail;
		stages[i].spirv            = (const uint32_t *)rd_blob(&r, spirv_bytes);
		stages[i].spirv_word_count = spirv_bytes / 4;
	}
	if (r.bad) goto fail;

	file->buffers            = buffers;       file->buffer_count       = buffer_count;
	file->vertex_inputs      = vins;          file->vertex_input_count = vin_count;
	file->resources          = resources;     file->resource_count     = resource_count;
	file->spec_consts        = specs;         file->spec_count         = spec_count;
	file->stages             = stages;        file->stage_count        = stage_count;
	file->_impl              = impl;
	return file;

fail:
	svsl_arena_free(&impl->arena);
	free(impl);
	return NULL;
}

void svsl_sks_free(svsl_sks_file_t *file) {
	if (!file) return;
	sks_impl_t *impl = file->_impl;
	svsl_arena_free(&impl->arena);
	free(impl);
}
