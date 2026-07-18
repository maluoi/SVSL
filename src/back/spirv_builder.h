// SPIR-V word builder: id allocation, per-section instruction streams
// concatenated at finalize, and type/constant caches. All opcodes and enums
// come from vendor/spirv.h — never hand-written constants.

#pragma once

#include "../util/arena.h"
#include "../util/array.h"
#include "../util/str.h"
#include "../../vendor/spirv.h"

#include <stdint.h>

typedef svsl_array_t(uint32_t) svsl_spv_stream_t;

typedef struct svsl_spv_type_key_t {
	uint16_t op;      // SpvOpTypeXxx
	uint32_t a, b, c, d, e, f, g;
	uint32_t id;
} svsl_spv_type_key_t;

typedef struct svsl_spv_t {
	svsl_arena_t *arena;
	uint32_t      next_id;
	// module header version word (major<<16 | minor<<8); starts at the 1.3
	// baseline, raised by features that need more (QCOM image processing → 1.4).
	// 1.4+ changes the entry-point interface rule — the emitter handles that.
	uint32_t      version;

	svsl_spv_stream_t caps;        // OpCapability
	svsl_spv_stream_t extensions;  // OpExtension
	svsl_spv_stream_t imports;     // OpExtInstImport
	svsl_spv_stream_t memory;      // OpMemoryModel
	svsl_spv_stream_t entries;     // OpEntryPoint
	svsl_spv_stream_t exec_modes;  // OpExecutionMode
	svsl_spv_stream_t debug;       // OpName / OpMemberName
	svsl_spv_stream_t decor;       // Op(Member)Decorate
	svsl_spv_stream_t types;       // types, constants, global variables
	svsl_spv_stream_t funcs;       // function bodies

	uint32_t glsl450;              // ext-inst import id

	svsl_array_t(svsl_spv_type_key_t) type_cache;  // types + pointer types
	svsl_array_t(svsl_spv_type_key_t) const_cache; // (type, bits) → id
	svsl_array_t(uint32_t)            cap_list;    // deduped capabilities
} svsl_spv_t;

void     svsl_spv_init(svsl_spv_t *spv, svsl_arena_t *arena);
uint32_t svsl_spv_id  (svsl_spv_t *spv);

// instruction emission: operands as an array
void svsl_spv_inst(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op,
                   const uint32_t *operands, uint32_t operand_count);
// convenience for short instructions
void svsl_spv_inst1(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a);
void svsl_spv_inst2(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b);
void svsl_spv_inst3(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b, uint32_t c);
void svsl_spv_inst4(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op, uint32_t a, uint32_t b, uint32_t c, uint32_t d);

// instruction with a trailing string literal (OpName, OpEntryPoint, …):
// fixed operands first, then the packed string
void svsl_spv_inst_str(svsl_spv_t *spv, svsl_spv_stream_t *stream, SpvOp op,
                       const uint32_t *operands, uint32_t operand_count, svsl_str_t text);

void svsl_spv_cap(svsl_spv_t *spv, SpvCapability cap);          // deduped
void svsl_spv_extension(svsl_spv_t *spv, const char *name);     // deduped by content
void svsl_spv_require_version(svsl_spv_t *spv, uint32_t version); // raises, never lowers

// cached type lookup: emits OpType* on first request. Operands beyond the key
// are not supported — struct types are emitted uncached by the caller.
uint32_t svsl_spv_type(svsl_spv_t *spv, SpvOp op, const uint32_t *operands, uint32_t count);

// cached scalar constant (bits sized by the type's width; 64-bit uses two words)
uint32_t svsl_spv_const(svsl_spv_t *spv, uint32_t type_id, uint64_t bits, bool wide, bool is_bool);

// final module: header + all streams, arena-owned
const uint32_t *svsl_spv_finalize(svsl_spv_t *spv, int32_t *out_word_count);
