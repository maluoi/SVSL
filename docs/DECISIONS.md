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
| Output | SKS container v9, plus `.spv`/headers; one version at a time, no back-compat | StereoKit is the target consumer; the version field exists to refuse old files, not to branch on |
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
| glslang divergences | Keep HLSL/DXC-correct behavior; note the divergence in the check shader | Verified glslang bugs: WavePrefixSum emits inclusive (should be exclusive), InterlockedCompareStore emits nothing, `ldexp` with float exponent emits invalid SPIR-V |
| Structured-buffer element layout | Object-form elements default to **C layout where C layout is free**: pack1 rules, but layouts needing `scalarBlockLayout` are compile errors unless `pack1` is written explicitly, which permits them and records SKS feature bit 16. Layout keywords prefix declarations (any form) with standards-name aliases `scalar`/`relaxed`/`std140`/`std430`. | The product goal is C-struct interop: `element_size == sizeof`, offsets match plain C. The error-vs-infer split is about consent, mirroring `float16`: typing a feature opts into its device requirement, but a bare declaration names no layout and must not silently narrow device support (the failure would surface as pipeline-creation errors on other people's hardware). glslang was rejected as the reference here — its HLSL mode emits DX-packed offsets with a std430-rounded stride (a TODO'd inconsistency in `updateMemberOffset`, which even excludes `$Global` by name), matching neither C nor std430, so C arrays break from element 1. The structs the default refuses are exactly those std430 silently corrupted against C. |

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
