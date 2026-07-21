# SVSL Design Decisions

The *why* behind SVSL's shape. Code style and ground rules are in `CLAUDE.md`; the
language surface is `docs/LANGUAGE_SPEC.md`; the optimizer is `docs/OPTIMIZATION_PLAN.md`;
future work is `docs/BACKLOG.md`. This document is the standing rationale — it outlives the
implementation plan the project was built from.

## Lessons from the prototype

`~/projects/spirv_sl` was a working ~14k-LOC prototype of this concept. It proved the
approach (hand-emitted SPIR-V, arena allocation, data-driven tables, pixel-diff testing
against skshaderc) and documented its own failures. SVSL is a rewrite, not a refactor; each
prototype failure is addressed *structurally* here:

| Prototype failure | Cause | Structural fix in SVSL |
|---|---|---|
| Textures/samplers as function params broke codegen ("Id is 0") | Direct AST→SPIR-V; opaque types can't be loaded/passed as values | IR stage with mandatory full inlining — opaque params resolve to the caller's global resource at inline time |
| 64× over-fetch: loaded a whole struct, then extracted one member | No access-chain folding | IR access-chain folding: member/index chains fold into one pointer chain |
| Spec promised mesh/RT/coop-matrix/pointers; impl covered a fraction | Aspirational spec | The spec documents implemented features only; everything else is deferred (below) or backlog |
| Half-built preprocessor inside the parser (`#elif` unsupported) | Preprocessing mixed into parsing | Separate preprocessor pass with a line map |
| Two entry-point conventions; doc/code binding-offset divergence | Organic growth | One binding model (below), verified by SPIR-V diff against skshaderc |
| Mixed ownership (arena + malloc'd output blobs) | No ownership rule | Everything owned by the result arena; one `svsl_result_free` |
| `spirv_version`/`debug_info` options silently ignored | — | Options are honored or rejected with an error, never silently ignored |
| ~33 dead/scaffolding functions | "For later" code | No dead scaffolding (CLAUDE.md) |

## Decision log

| Decision | Choice | Rationale |
|---|---|---|
| v1 scope | HLSL core + float16/layouts/pushconstant/multiview/spec-constants/demote/interp + subgroups/atomics/barriers + early-Z toolkit/`[wave_size]`/`SubpassInput` | Honest spec; the corpus needs most of it anyway; the additions are decoration-cheap except SubpassInput (runtime support landed in sk_renderer) |
| Names over numbers | Bare `specialization` auto-ids; `register()` optional with deterministic auto-binding; locations by declaration order | Reflection is name-based, so numbers are an interop detail; auto-assignment never collides with explicit values |
| Deferred | pointers/bindless, mesh/task, RT, coop-matrix, interlock | Big surface; the prototype proved the cost of promising them early |
| Output | SKS container v12, plus `.spv`/headers; one version at a time, no back-compat | StereoKit is the target consumer; the version field exists to refuse old files, not to branch on |
| Codegen | AST → small SSA-ish IR → SPIR-V, mandatory full inlining | Structurally fixes opaque-params and access-chain bugs; matches reference-compiler behavior |
| Dependencies | Core lib + CLI: libc only; vendored `spirv.h`; `spirv-val` shelled out; app deps FetchContent, dev-only | Dependency-free/light core is a hard requirement |
| Compat level | Corpus compiles (light porting acceptable); legacy forms porting-warn under `-Wporting` (opt-in, off by default) | User-selected; hints are opt-in so a clean corpus build stays quiet, and someone modernizing turns them on deliberately |
| Entry points | Name-based (`vs`/`ps`/`cs`) and attributes both first-class | Name-based is the StereoKit convention; attributes are the native form |
| Binding scheme | `b`+0, `t`/`s`+100, `u`+200 (skshaderc-compatible); `register(bind, set)` for explicit | Must match the runtime; the prototype's doc/code divergence is resolved in favor of skshaderc's actual behavior |
| half vs float16 | `half` = RelaxedPrecision f32 (min16float semantics, portable, degrades gracefully); `float16` = true 16-bit opt-in; `--half=strict16` to re-type | Desktop fp16 support is spotty; SK relies on graceful degradation; explicit-width types give a natural home for strict fp16 |
| `vk::` escapes | All `[[vk::*]]` parse and work; each concept also has native SVSL syntax; unknown `vk::` attrs are errors | `vk::` attributes mark exactly the concepts HLSL lacks — SVSL's reason to exist |
| No JSON | No JSON input or output anywhere; reflection = C API, container, or text tables | User preference |
| SPIR-V target | 1.3 / Vulkan 1.1, fixed; extras via extensions | Matches skshaderc; a version option nothing needs would be dead surface |
| Preprocessor | Real separate pass with a line map | The corpus requires it; the prototype's parser-embedded half-preprocessor was a known smell |
| Optimization | Fixed in-IR pipeline iterated to a fixpoint: fold, peephole, store→load forwarding, dead-store, CSE, DCE — value-preserving at `-O1` (default, oracle-covered), float algebra opt-in at `-O2`. `-O0/1/2` flag. No pass manager. See `docs/OPTIMIZATION_PLAN.md`. | Clean output without linking SPIRV-Tools; `-O1` cut corpus SPIR-V ~14% / live IR ~34%, pixel-identical to skshaderc; heavy opt still `spirv-opt`'d externally |
| Correctness bar | Outputs only: pixels and buffer bits, bit-exact for compute | SPIR-V text differences are legal encodings of the same program; comparing them creates false failures and hides real ones |
| glslang divergences | Keep HLSL/DXC-correct behavior; note the divergence in the check shader | Verified glslang bugs: WavePrefixSum emits inclusive (should be exclusive), InterlockedCompareStore emits nothing, `ldexp` with float exponent emits invalid SPIR-V, unknown `SV_*` on a vertex input silently becomes a located attribute that its own reflection then drops (meta ≠ SPIR-V — the bug class SKS v10 exists to kill; SVSL rejects it like DXC) |
| Structured-buffer element layout | Object-form elements default to **C layout where C layout is free**: pack1 rules, but layouts needing `scalarBlockLayout` are compile errors unless `pack1` is written explicitly, which permits them and records SKS feature bit 16. Layout keywords prefix declarations (any form) with standards-name aliases `scalar`/`relaxed`/`std140`/`std430`. | The product goal is C-struct interop: `element_size == sizeof`, offsets match plain C. The error-vs-infer split is about consent, mirroring `float16`: typing a feature opts into its device requirement, but a bare declaration names no layout and must not silently narrow device support (the failure would surface as pipeline-creation errors on other people's hardware). glslang was rejected as the reference here — its HLSL mode emits DX-packed offsets with a std430-rounded stride (a TODO'd inconsistency in `updateMemberOffset`, which even excludes `$Global` by name), matching neither C nor std430, so C arrays break from element 1. The structs the default refuses are exactly those std430 silently corrupted against C. |
| QCOM extensions (image_processing/2, tile_shading) | First-class syntax, every name `QCOM`-postfixed (`SampleWeightedQCOM`, `[tile_shading_rate_qcom]`, `tile_offset_qcom()`). Verification is compile + spirv-val + emitted-word unit tests (`tests/test_qcom.c`) — no desktop implementation exists, and skshaderc cannot compile any of it | A driving reason for SVSL and a differentiator: no HLSL frontend (DXC, glslang-HLSL, Slang) exposes these natively; only GLSL does. The postfix keeps clean names free for a future EXT promotion |
| QCOM image-processing surface | Texture methods with **one shared sampler** per op; decorations (`WeightTextureQCOM`, `BlockMatchTextureQCOM`, `BlockMatchSamplerQCOM`) inferred from use, and decorated resources are **exclusive to their op family** (compile error on mixing). These modules emit SPIR-V 1.4 — the extension's floor — with the all-globals interface 1.4 mandates; every other module stays 1.3 | The sampler constraints (unnormalized, clamp-only, `IMAGE_PROCESSING` create flag) are identical for both textures of a block match, so split sampler args would only add footguns. Exclusivity mirrors the Vulkan binding rules — no descriptor satisfies a mixed-use resource. spirv-val traces the decorations through *direct* OpLoads, so fused texture+sampler pairs load the combined variable (both decorations land on it, glslang's combined-sampler2D model), and a window-op texture fused with a *different* sampler is unexpressable → compile error |
| Tile attachments | `[tile_attachment]` attribute on `Texture2D`/`RWTexture2D`, not a new type name | The type *is* a 2D texture/image in every way the type system cares about; only the storage class (`TileAttachmentQCOM`) and bind semantics differ. GLSL made the same call (`tile_attachmentQCOM` layout qualifier). Distinct from `TileImage` (EXT), which has no descriptor at all |
| Tile shading rate | `[tile_shading_rate_qcom(x,y,z)]` **replaces** `[numthreads]`; declaring both is an error, and the attribute alone marks a compute entry | SPIR-V forbids TileShadingRateQCOM alongside LocalSize — the implementation derives the workgroup shape from the rate (area-based dispatch) |
| Apron size | Program-level reserved meta key `//--apron = W[, H]` (parsed + validated now, serialized in the SKS v11 bump), warned when nothing tile-shaped consumes it | The apron is per-render-pass state (`VkRenderPassTileShadingCreateInfoQCOM::tileApronSize`) with **no shader-side representation** in SPIR-V or any shading language — a per-attachment spelling would promise granularity the API doesn't have. Carrying it as shader metadata is *more* than GLSL offers (there, apps hardcode it in C++); shaders read the active value via `tile_apron_size_qcom()` |
| spirv_asm requirements | `OpCapability <int>;` / `OpExtension "name";` inside a block are routed to the module's declaration streams (deduped); string-literal operands added | The escape hatch stays closed under composition — a block states its own prerequisites in SPIR-V's own spelling. Rejected: `//--require` (the `//--` channel is material metadata read by renderers, and it would be the first meta key to change codegen) and a `require`/`#require` keyword (new surface duplicating a spelling SPIR-V already has) |

## SKS v9 additions

v9 adds three things to the v8 container, each motivated by a gap SVSL hit building against
v8. The authoritative byte layout is sk_renderer's `sksc.cpp::sksc_build_file` /
`sksc_file.c::_sksc_load_meta`; `src/out/sks_write.c` follows it exactly. Standing policy:
**one version at a time** — the version field lets runtimes refuse old files, it is not a
compatibility mechanism, so a version bump means recompiling shaders.

- **Device-feature mask** (`uint64 features` + reserved) in the header, derived by the
  compiler from the SPIR-V it actually emitted (capabilities/extensions, plus one
  format-implying opcode). Each bit is a device-level feature question the runtime settles
  before pipeline creation without parsing the blob; unknown capabilities set bit 63.
  Bit 16 (`scalarBlockLayout`) is set from sema rather than the blob — scalar block
  layout has no SPIR-V capability, so the mask is its only machine-readable signal.
- **Per-resource shape/format bytes**: a texture shape byte (dim/arrayed/MS/comparison) and
  a storage-image format byte, for bind-time validation instead of guessing from names.
- **Per-stage `wave_size`**: so multiple compute entries can pin subgroup size independently.

## SKS v10 additions

v10 adds one byte per vertex-input record: `uint8 location`, the input's SPIR-V location
(first of the span for arrays/matrices), serialized after `semantic_slot`. It exists so
sk_renderer can wire mesh components to shader inputs by semantic at pipeline creation
(`../sk_renderer/docs/PLAN_attribute_remap.md`) instead of assuming meta array index ==
location — an assumption v9 silently broke whenever an unused input was dropped from the
meta while its location stayed consumed in the SPIR-V.

Invariant, and where the data flows: **the vertex-inputs block mirrors the vs module's
input interface exactly** — an entry is written iff its OpVariable survived emission, at
the location the emitter actually decorated. The emitter records this per input while
decorating (`svsl_spirv_blob_t.vs_input_locations`, name-checked against
`prog->vertex_inputs` so a walk divergence is a compile error, not wrong metadata); the
SKS writer only consumes it. Location math is never re-derived downstream, and the old
IR-side vertex-input usage scan was deleted with it. Unused inputs are stripped from the
SPIR-V (matching glslang) but still consume their location, so recorded locations have
visible gaps — `checks/check_vertex_locations.hlsl` pins this and the explicit
`[location(N)]` path against the reference compiler.

## SKS v11 additions

v11 carries the QCOM extension reflection (lockstep change with sk_renderer's
`sksc_file.h`/`sksc.cpp`/`sksc_file.c`):

- **`uint32 tile_apron[2]`** in the meta block, after `wave_size` — the `//--apron`
  value, applied by the renderer as `VkRenderPassTileShadingCreateInfoQCOM::tileApronSize`.
- **Four `skr_register_` values** so the runtime picks the right descriptor type:
  `sample_weight` (VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM), `block_match`
  (BLOCK_MATCH_IMAGE_QCOM), and `tile_sampled`/`tile_storage` (ordinary
  combined-sampler/storage-image descriptors whose variables live in
  `TileAttachmentQCOM` storage). The classification comes from the emitters
  (`svsl_spirv_blob_t.qcom_res_use`, the same array that drives decorations and
  exclusivity), never re-derived — mirroring the v10 vertex-locations pattern.
- **Resource shape bit 6**: this sampler serves QCOM image-processing ops and must be
  created with `VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM`. Set on standalone sampler
  records, or on the texture record when the sampler is fused (like the comparison
  bit 5).

Feature bits 17–19 (image_processing, image_processing2, tile_shading) predate the bump
— they fit the existing v9 mask — but v11 is where runtimes gain the reflection to act
on them.
