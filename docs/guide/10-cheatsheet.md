# Appendix: Quick Reference & Porting Cheat-Sheet

A one-page condensation of the mappings scattered through the guide. Every row here is
covered in depth in the chapter it links to; this page is for quick lookup while porting or
writing. Everything below compiles **today**.

---

## Native ↔ HLSL spellings

Both columns compile to identical SPIR-V. The native form is recommended; the HLSL form keeps
existing shaders working. Under `-Wporting`, the rows marked † earn a suppressible `porting`
hint; all others are accepted silently. ([full detail →](09-hlsl-compatibility.md))

### Types

| Native | HLSL alias | |
|--------|-----------|--|
| `half`, `half2/3/4` | `min16float`, `min16float2/3/4` † | relaxed-precision float - **not** `float16` ([Types §3](02-types-and-values.md#3-half-vs-float16--the-one-thing-that-is-not-hlsl)) |
| `Sampler` | `SamplerState` † | |
| `SamplerComparison` | `SamplerComparisonState` † | |
| `Image2D<T[, fmt]>` (etc.) | `RWTexture2D<T>` † + `[[vk::image_format("fmt")]]` † | storage image ([Resources §2](05-resources-and-buffers.md#2-storage-images)) |
| `storagebuffer [readonly] N { T x[]; }` | `StructuredBuffer<T>` / `RWStructuredBuffer<T>` / `Buffer<T>` † | block vs. subscript access |

### Declaration keywords (accepted silently, even under `-Wporting`)

| Native | HLSL alias |
|--------|-----------|
| `uniform` | `cbuffer` |
| `workgroup` | `groupshared` |
| `flat` | `nointerpolation` |
| `[compute(x,y,z)]` | `[numthreads(x,y,z)]` |
| `[early_depth_stencil]` | `[earlydepthstencil]` |
| declaration-order locations / `[location(N)]` | legacy semantics (`NORMAL0`, `TEXCOORD2`, `COLOR1`, …) |

### `[[vk::*]]` escapes → native (all †)

| `[[vk::*]]` | Native |
|-------------|--------|
| `[[vk::binding(b, s)]]` | `register(b, s)` - direct `(binding, set)` |
| `[[vk::push_constant]]` | `pushconstant N { }` |
| `[[vk::constant_id(N)]]` | `[specialization(N)]`, or bare `specialization const` (auto-id) |
| `[[vk::image_format("rgba8")]]` | `Image2D<T, rgba8>` |
| `[[vk::input_attachment_index(N)]]` | `SubpassInput<T, N>`, or `SubpassInput<T>` (auto-index) |
| `[[vk::location(N)]]` | `[location(N)]` |
| `[[vk::offset(N)]]` | `[offset(N)]` |

An **unknown** `[[vk::*]]` is a hard error; a recognized one in the wrong place is a warning
and is ignored.

### Intrinsic aliases (all †)

`Wave*` / `Quad*` → `subgroup_*` / `quad_*` ([§7.4](06-intrinsics.md#74-hlsl-wave--quad-aliases)) ·
`Interlocked*` → `atomic_*` ([§8](06-intrinsics.md#8-atomics)) ·
`GroupMemoryBarrier*` → `workgroup_barrier()` etc. ([§9](06-intrinsics.md#9-barriers)) ·
`subpass.SubpassLoad(...)` → `subpass.Load(...)`.

---

## At-a-glance rules

**Default entry names:** `vs` / `ps` / `cs` (configurable via `-vs` / `-ps` / `-cs`). A stage
attribute (`[vertex]`, `[fragment]`/`[pixel]`, `[compute]`) works too.
([Stages §1](04-stages-and-interfaces.md#1-entry-points))

**Buffer layout defaults:**

| Declaration | Default | Allowed |
|-------------|---------|---------|
| `uniform` / `cbuffer` | `pack16` | `pack16` only |
| `storagebuffer` | `std430` | any layout keyword |
| `pushconstant` | `std430` | any layout keyword |
| `StructuredBuffer<T>` family | C layout (`pack1`, refusing layouts that need a device feature) | any layout keyword |

Keywords prefix the declaration: `pack16 storagebuffer Data { ... };`. Aliases:
`pack1` = `scalar` (C-struct tight; `scalarBlockLayout` only when a vector straddles) ·
`pack8` = `relaxed` · `pack16` = `std140` · `std430` = the core storage default.
([Resources §5](05-resources-and-buffers.md#5-memory-layout-pack1-pack8-pack16-std430))

**Binding offsets (prefix-letter `register()`):** `b` → +0 · `t`/`s` → +100 · `u` → +200. The
direct `register(binding, set)` form applies **no** offset. `register()` is optional - omit it
and SVSL auto-assigns the lowest free slot, never colliding with explicit ones.
([Resources §7](05-resources-and-buffers.md#7-bindings-and-registers))

**Porting hints:** off by default; enable with `-Wporting`. Only the † rows above warn;
`cbuffer` / `groupshared` / `nointerpolation` / legacy semantics stay silent.
([HLSL Compatibility §1](09-hlsl-compatibility.md#1-one-language-two-spellings))

**System values:** full table in
[Stages §3](04-stages-and-interfaces.md#3-semantics-system-values-vs-numbered-io). A misplaced
`SV_*` degrades silently to a numbered varying.

**Deliberate glslang divergences:** `WavePrefixSum` is exclusive (HLSL semantics) ·
`InterlockedCompareStore` actually compiles.
([§4](09-hlsl-compatibility.md#4-deliberate-divergences-from-glslang))
