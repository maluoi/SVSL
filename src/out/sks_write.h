// SKS container writer (StereoKit's SKSHADER format). One version at a time,
// per the standing SKS policy — the version field lets runtimes refuse old
// files, it is not a compatibility mechanism. Byte layout follows
// sksc.cpp::sksc_build_file / sksc_file.c exactly (the authoritative pair).

#pragma once

#include "../back/emit_spirv.h"
#include "../back/emit_wgsl.h"
#include "../ir/ir.h"
#include "../sema/sema.h"

typedef struct svsl_sks_blob_t {
	const uint8_t *bytes;
	int32_t        size;
} svsl_sks_blob_t;

typedef struct svsl_sks_options_t {
	uint32_t targets;          // svsl_target_ bits; 0 = SPIR-V only
	bool     keep_debug_names; // keep OpName/OpMemberName in SPIR-V stages
	bool     no_smolv;         // store SPIR-V stages raw rather than SMOL-V encoded
} svsl_sks_options_t;

// blobs: one SPIR-V module per module->funcs entry, same order — always
// required (metadata derives from them); opts->targets decides which languages
// the container carries (skshaderc -t style). opt_wgsl adds one WGSL stage
// record per entry whose blob has text, plus the v12 standalone sampler records
// those stages bind.
void svsl_sks_write(svsl_arena_t *arena, const svsl_program_t *prog,
                    const svsl_ir_module_t *module, const svsl_spirv_blob_t *blobs,
                    const svsl_wgsl_blob_t *opt_wgsl, const svsl_sks_options_t *opts,
                    svsl_sks_blob_t *out_blob);
