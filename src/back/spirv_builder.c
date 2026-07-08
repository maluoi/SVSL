#include "spirv_builder.h"

#include <assert.h>
#include <string.h>

// SPIR-V module header version word: major<<16 | minor<<8. Targets 1.3 (the
// baseline for the features SVSL emits: subgroups, 8/16-bit storage).
#define SVSL_SPIRV_VERSION 0x00010300u

void svsl_spv_init(svsl_spv_t *spv, svsl_arena_t *arena) {
	*spv = (svsl_spv_t){ .arena = arena, .next_id = 1 };
}

uint32_t svsl_spv_id(svsl_spv_t *spv) {
	return spv->next_id++;
}

void svsl_spv_inst(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op,
                   const uint32_t *operands, uint32_t operand_count) {
	svsl_array_push(spv->arena, stream, ((operand_count + 1) << SpvWordCountShift) | (uint32_t)op);
	for (uint32_t i = 0; i < operand_count; i++)
		svsl_array_push(spv->arena, stream, operands[i]);
}

void svsl_spv_inst1(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a) {
	svsl_spv_inst(spv, stream, op, (uint32_t[]){ a }, 1);
}
void svsl_spv_inst2(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b) {
	svsl_spv_inst(spv, stream, op, (uint32_t[]){ a, b }, 2);
}
void svsl_spv_inst3(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b, uint32_t c) {
	svsl_spv_inst(spv, stream, op, (uint32_t[]){ a, b, c }, 3);
}
void svsl_spv_inst4(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
	svsl_spv_inst(spv, stream, op, (uint32_t[]){ a, b, c, d }, 4);
}

void svsl_spv_inst_str(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op,
                       const uint32_t *operands, uint32_t operand_count, svsl_str_t text) {
	uint32_t str_words = ((uint32_t)text.len + 1 + 3) / 4; // incl. NUL, padded
	svsl_array_push(spv->arena, stream,
	                ((operand_count + str_words + 1) << SpvWordCountShift) | (uint32_t)op);
	for (uint32_t i = 0; i < operand_count; i++)
		svsl_array_push(spv->arena, stream, operands[i]);
	for (uint32_t w = 0; w < str_words; w++) {
		uint32_t word = 0;
		for (uint32_t k = 0; k < 4; k++) {
			uint32_t index = w * 4 + k;
			if (index < (uint32_t)text.len)
				word |= (uint32_t)(uint8_t)text.ptr[index] << (k * 8);
		}
		svsl_array_push(spv->arena, stream, word);
	}
}

void svsl_spv_cap(svsl_spv_t *spv, SpvCapability cap) {
	for (int32_t i = 0; i < spv->cap_list.count; i++)
		if (spv->cap_list.items[i] == (uint32_t)cap) return;
	svsl_array_push(spv->arena, &spv->cap_list, (uint32_t)cap);
	svsl_spv_inst1(spv, &spv->caps, SpvOpCapability, (uint32_t)cap);
}

void svsl_spv_extension(svsl_spv_t *spv, const char *name) {
	// dedupe: scan the extension stream for an identical OpExtension instruction
	svsl_str_t text = svsl_str(name);
	uint32_t str_words = ((uint32_t)text.len + 1 + 3) / 4;
	uint32_t total     = str_words + 1;
	for (int32_t i = 0; i + (int32_t)total <= spv->extensions.count; i++) {
		if (spv->extensions.items[i] != ((total << SpvWordCountShift) | SpvOpExtension)) continue;
		bool same = true;
		for (uint32_t w = 0; w < str_words && same; w++) {
			uint32_t word = 0;
			for (uint32_t k = 0; k < 4; k++) {
				uint32_t index = w * 4 + k;
				if (index < (uint32_t)text.len) word |= (uint32_t)(uint8_t)text.ptr[index] << (k * 8);
			}
			if (spv->extensions.items[i + 1 + w] != word) same = false;
		}
		if (same) return;
	}
	svsl_spv_inst_str(spv, &spv->extensions, SpvOpExtension, NULL, 0, text);
}

uint32_t svsl_spv_type(svsl_spv_t *spv, SpvOp op, const uint32_t *operands, uint32_t count) {
	// Cache key holds 7 operands; every cached type (vectors, pointers, arrays,
	// small structs, function types) fits. Wide types (general OpTypeStruct) are
	// emitted uncached, so they never reach here.
	assert(count <= 7 && "type-cache key holds only 7 operands");
	svsl_spv_type_key_t key = { .op = (uint16_t)op };
	uint32_t *slots = &key.a;
	for (uint32_t i = 0; i < count && i < 7; i++) slots[i] = operands[i];

	for (int32_t i = 0; i < spv->type_cache.count; i++) {
		const svsl_spv_type_key_t *entry = &spv->type_cache.items[i];
		if (entry->op == key.op && entry->a == key.a && entry->b == key.b &&
		    entry->c == key.c && entry->d == key.d && entry->e == key.e &&
		    entry->f == key.f && entry->g == key.g)
			return entry->id;
	}
	uint32_t id = svsl_spv_id(spv);
	key.id = id;
	svsl_array_push(spv->arena, &spv->type_cache, key);

	uint32_t words[9] = { id };
	for (uint32_t i = 0; i < count && i < 8; i++) words[1 + i] = operands[i];
	svsl_spv_inst(spv, &spv->types, op, words, count + 1);
	return id;
}

uint32_t svsl_spv_const(svsl_spv_t *spv, uint32_t type_id, uint64_t bits, bool wide, bool is_bool) {
	svsl_spv_type_key_t key = {
		.op = (uint16_t)(is_bool ? (bits ? SpvOpConstantTrue : SpvOpConstantFalse) : SpvOpConstant),
		.a  = type_id,
		.b  = (uint32_t)bits,
		.c  = wide ? (uint32_t)(bits >> 32) : 0,
		.d  = wide ? 1u : 0u };
	for (int32_t i = 0; i < spv->const_cache.count; i++) {
		const svsl_spv_type_key_t *entry = &spv->const_cache.items[i];
		if (entry->op == key.op && entry->a == key.a && entry->b == key.b &&
		    entry->c == key.c && entry->d == key.d)
			return entry->id;
	}
	uint32_t id = svsl_spv_id(spv);
	key.id = id;
	svsl_array_push(spv->arena, &spv->const_cache, key);

	if (is_bool) {
		svsl_spv_inst2(spv, &spv->types, bits ? SpvOpConstantTrue : SpvOpConstantFalse, type_id, id);
	} else if (wide) {
		svsl_spv_inst4(spv, &spv->types, SpvOpConstant, type_id, id,
		               (uint32_t)bits, (uint32_t)(bits >> 32));
	} else {
		svsl_spv_inst3(spv, &spv->types, SpvOpConstant, type_id, id, (uint32_t)bits);
	}
	return id;
}

const uint32_t *svsl_spv_finalize(svsl_spv_t *spv, int32_t *out_word_count) {
	svsl_spv_stream_t out = {0};
	svsl_array_push(spv->arena, &out, SpvMagicNumber);
	svsl_array_push(spv->arena, &out, SVSL_SPIRV_VERSION);
	svsl_array_push(spv->arena, &out, 0);               // generator: none registered yet
	svsl_array_push(spv->arena, &out, spv->next_id);    // bound
	svsl_array_push(spv->arena, &out, 0);               // schema

	const svsl_spv_stream_t *sections[] = {
		&spv->caps, &spv->extensions, &spv->imports, &spv->memory, &spv->entries,
		&spv->exec_modes, &spv->debug, &spv->decor, &spv->types, &spv->funcs,
	};
	for (int32_t s = 0; s < (int32_t)(sizeof(sections) / sizeof(sections[0])); s++)
		for (int32_t i = 0; i < sections[s]->count; i++)
			svsl_array_push(spv->arena, &out, sections[s]->items[i]);

	*out_word_count = out.count;
	return out.items;
}
