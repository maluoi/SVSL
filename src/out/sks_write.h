// SKS container writer (StereoKit's SKSHADER format, version 9). One version
// at a time, per the standing SKS policy — the version field lets runtimes
// refuse old files, it is not a compatibility mechanism. Byte layout follows
// sksc.cpp::sksc_build_file / sksc_file.c exactly (the authoritative pair).

#pragma once

#include "../back/emit_spirv.h"
#include "../ir/ir.h"
#include "../sema/sema.h"

typedef struct svsl_sks_blob_t {
	const uint8_t *bytes;
	int32_t        size;
} svsl_sks_blob_t;

// blobs: one SPIR-V module per module->funcs entry, same order
void svsl_sks_write(svsl_arena_t *arena, const svsl_program_t *prog,
                    const svsl_ir_module_t *module, const svsl_spirv_blob_t *blobs,
                    svsl_sks_blob_t *out_blob);
