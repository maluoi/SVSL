// WGSL emission: one IR entry function → one WGSL module (text), for the
// StereoKit WebGPU backend. Only compiled when SVSL_ENABLE_WGSL is on; the
// emitter is pure C text output — the core stays dependency-free.
//
// Output contract (sk_renderer's wgpu backend, sksc_file.h v12):
// - All bindings @group(0); slot = register + shift: b+0, t+100, u+200,
//   input attachments +300, standalone samplers +400.
// - Textures and samplers never fuse; each texture's paired sampler binds
//   separately and is reported in `samplers` so the runtime can apply the
//   texture's sampler settings (Vulkan combined-sampler semantics).
// - SV_ViewID reads become the pipeline-overridable constant sk_view_index
//   (@id 999, default 0); the runtime renders one pass per view.
// - A stage using features browser WebGPU can't express is *skipped*: the
//   blob's text stays NULL, a warning diagnostic says why, and the SKS simply
//   carries no WGSL for that stage (it still serves Vulkan).

#pragma once

#include "../ir/ir.h"
#include "../sema/sema.h"
#include "../diag.h"

#include <stdint.h>

typedef struct svsl_wgsl_sampler_t {
	svsl_str_t name;        // paired: "<texture>_sampler"; standalone: its own name
	uint16_t   slot;        // s register + 400
	uint16_t   paired_slot; // texture resource bind slot, 0xFFFF = unpaired
} svsl_wgsl_sampler_t;

typedef struct svsl_wgsl_blob_t {
	const char *text;   // NUL-terminated WGSL; NULL when the stage was skipped
	int32_t     length; // excluding the NUL
	const svsl_wgsl_sampler_t *samplers; // split samplers this stage binds
	int32_t                    sampler_count;
} svsl_wgsl_blob_t;

// Emits WGSL for one entry point. Returns false only on hard errors (an error
// diagnostic was added); an inexpressible stage returns true with text == NULL
// and a warning diagnostic naming the offending feature and location.
bool svsl_wgsl_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                    const svsl_ir_func_t *fn, svsl_wgsl_blob_t *out_blob,
                    svsl_diag_list_t *ref_diags);
