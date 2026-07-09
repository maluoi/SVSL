# HLSL Compatibility

SVSL is designed so the C-like subset of HLSL - specifically StereoKit's shader corpus -
compiles with little or no editing. This chapter is the porting reference: the dialect model,
the complete list of legacy spellings and their native replacements, the `[[vk::*]]` escape
mapping, the diagnostics you'll see, and the handful of places SVSL deliberately behaves
differently from the reference compiler (glslang).

---

## 1. One language, two spellings

SVSL is a single language with, for many concepts, both an **HLSL spelling** and a
**native SVSL spelling**. Both compile to the same SPIR-V. The native spelling is the
recommended form; the HLSL spelling is there so your existing shaders keep working.

**When does the compiler nudge you toward the native form?** Only when you ask it to.
Porting hints are **opt-in**: off by default, enabled with `-Wporting`.

- **Off by default.** Every legacy spelling - `cbuffer`, `groupshared`, `min16float`,
  `SamplerState`, `RWTexture2D`, `StructuredBuffer`, the `Wave*` / `Interlocked*` /
  `GroupMemoryBarrier*` intrinsics, and the `[[vk::*]]` escapes - compiles with **no
  diagnostic**. They are true aliases, not deprecations.
- **With `-Wporting`,** each legacy form that has a native SVSL spelling earns a suppressible
  **`porting`** hint: the `[[vk::*]]` escapes (§3), the legacy resource and scalar type
  spellings (`min16float`, `SamplerState`, `SamplerComparisonState`, `StructuredBuffer`,
  `RWStructuredBuffer`, `RWTexture*`), and the HLSL intrinsic aliases (`Wave*`,
  `Interlocked*`, `GroupMemoryBarrier*`). A hint never fires on a construct with no native
  alternative.
- **Not yet hinted,** even under `-Wporting`: the declaration keywords `cbuffer` → `uniform`,
  `groupshared` → `workgroup`, `nointerpolation` → `flat`, and legacy semantics. These
  compile silently for now (tracked in `docs/BACKLOG.md`).

So porting is quiet by default; turn on `-Wporting` when you want to modernize a shader
deliberately, using the tables below.

---

## 2. Legacy spelling → native spelling

Every row here compiles today. Under `-Wporting`, the type, intrinsic, and `[[vk::*]]` rows
emit a hint; the keyword rows (`cbuffer`, `groupshared`, `nointerpolation`) do not yet.

| HLSL spelling | SVSL-native | Notes |
|---------------|-------------|-------|
| `cbuffer N : register(b0) { }` | `uniform N : register(b0) { }` | read-only constant buffer |
| `StructuredBuffer<T>` | `readonly storagebuffer N { T x[]; }` | subscript vs. block access |
| `RWStructuredBuffer<T>` | `storagebuffer N { T x[]; }` | |
| `Buffer<T>` | `readonly storagebuffer` block | |
| `groupshared` | `workgroup` | compute shared memory |
| `SamplerState` | `Sampler` | |
| `SamplerComparisonState` | `SamplerComparison` | |
| `RWTexture2D<T>` (etc.) | `Image2D<T[, format]>` | storage image; format is a template arg |
| `min16float`, `min16float2/3/4` | `half`, `half2/3/4` | relaxed-precision float |
| `nointerpolation` | `flat` | |
| `[numthreads(x,y,z)]` | `[compute(x,y,z)]` | compute workgroup size |
| `Wave*` / `Quad*` intrinsics | `subgroup_*` / `quad_*` | see [Intrinsics §7.4](06-intrinsics.md#74-hlsl-wave--quad-aliases) |
| `Interlocked*` | `atomic_*` | out-param vs. return value |
| `GroupMemoryBarrierWithGroupSync`, … | `workgroup_barrier()`, … | see [Intrinsics §9](06-intrinsics.md#9-barriers) |
| `NORMAL0`, `TEXCOORD2`, `COLOR1`, … | declaration-order locations, or `[location(N)]` | semantics kept for vertex-input reflection |

---

## 3. `[[vk::*]]` escapes → native syntax

Each `[[vk::*]]` attribute exists in HLSL because HLSL has no keyword for the Vulkan concept.
SVSL gives every one a real keyword. The escape still parses and works, and under `-Wporting`
earns a `porting` hint pointing at the native spelling.

| `[[vk::*]]` escape | SVSL-native form |
|--------------------|------------------|
| `[[vk::binding(b, s)]]` | `register(b, s)` - the direct `(binding, set)` form |
| `[[vk::push_constant]] cbuffer` | `pushconstant N { }` |
| `[[vk::constant_id(N)]] const T x` | `[specialization(N)] const T x`, or bare `specialization const` (auto-id) |
| `[[vk::image_format("rgba8")]] RWTexture2D<T>` | `Image2D<T, rgba8>` |
| `[[vk::input_attachment_index(N)]] SubpassInput<T>` | `SubpassInput<T, N>`, or `SubpassInput<T>` (auto-index) |
| `[[vk::location(N)]]` | `[location(N)]` |
| `[[vk::offset(N)]]` | `[offset(N)]` |

Two related behaviours:

- **An unknown `[[vk::*]]` is a hard error**, naming the attribute: `unknown attribute
  '[[vk::something]]'`. It is never silently skipped - an escape you thought was doing
  something won't quietly do nothing.
- **A recognized `[[vk::*]]` used in the wrong place is a `warning`**, not an error, and the
  attribute is ignored: `attribute 'vk::binding' does not apply here and is ignored`.

---

## 4. Deliberate divergences from glslang

The reference compiler for the visual tests is glslang (via skshaderc). SVSL matches its
output almost everywhere - but in a few spots glslang is simply *wrong*, and SVSL chooses
correctness over bug-for-bug compatibility:

- **`WavePrefixSum` is an exclusive scan.** HLSL and DXC define it as exclusive; glslang
  emits an inclusive scan. SVSL follows the HLSL semantics. If you were relying on glslang's
  behaviour, your results will differ (and be correct).
- **`InterlockedCompareStore` actually compiles.** glslang silently drops it, emitting
  nothing. SVSL implements it as a compare-exchange.

Both are noted inline where the feature is documented ([Intrinsics
§7.4](06-intrinsics.md#74-hlsl-wave--quad-aliases), [§8](06-intrinsics.md#8-atomics)).

---

## 5. Things that are not HLSL (or behave differently)

A short list of gotchas for someone arriving from HLSL/DXC:

- **`half` ≠ `float16`.** `half` is relaxed-precision float32 (portable, HLSL's `min16float`
  semantics); `float16` is true 16-bit and requires capability support. See
  [Types §3](02-types-and-values.md#3-half-vs-float16--the-one-thing-that-is-not-hlsl).
- **A texture and same-slot sampler fuse into one binding.** There is no standalone sampler
  in the output. Pairing is by slot number, not the `_s` name.
- **Matrices are column-major, and `*` between matrices is component-wise.** Use `mul()` for
  matrix and matrix/vector products, exactly as in HLSL - but don't expect `a * b` to be a
  matrix product. Indexing `m[i]` gives **row** *i* (HLSL-style), not a column.
- **A misplaced `SV_*` semantic degrades silently to a numbered varying** rather than
  erroring. Check the [system-value
  table](04-stages-and-interfaces.md#3-semantics-system-values-vs-numbered-io) if a built-in
  isn't behaving like one.
- **Recursion is a compile error** - every call is inlined, and SPIR-V forbids recursion.
- **Resources can't be local variables** - declare them as globals and pass them into helper
  functions as parameters (which *is* allowed, and inlines away).
- **`uniform` buffers must use `pack16`.** Storage buffers, push constants, and the
  object-form `StructuredBuffer` family accept any layout keyword; structured-buffer
  elements default to C layout ([Resources §5](05-resources-and-buffers.md#5-memory-layout-pack1-pack8-pack16-std430)).
- **Legacy D3D9 intrinsics `lit`, `dst`, `msad4` are rejected** with an error - compute the
  terms directly.
- **Bare globals with initializers aren't private variables** - they become `$Global`
  members and material defaults. Use `static` for a genuine private global.

---

## 6. Diagnostics

Every diagnostic carries a file, line, and column that survives preprocessing and includes.
Sema recovers after an error and keeps going, so you get a *batch* of diagnostics, not just
the first failure. There are four severities:

| Severity | Meaning |
|----------|---------|
| `error` | illegal - no output is produced |
| `warning` | legal but suspicious (e.g. a `[[vk::*]]` attribute ignored in the wrong context) |
| `porting` | a recognized `[[vk::*]]` escape has a native spelling; suppressible |
| `info` | a note (e.g. an ignored `#pragma`, or a loop attribute that couldn't be applied) |

The porting story, restated once more because it's the compatibility model in a sentence:
**only `[[vk::*]]` attributes are flagged; all other legacy HLSL spellings compile
silently.**

---

## 7. A porting checklist

To bring an HLSL shader across:

1. **Compile it.** Most StereoKit-style shaders compile unchanged. Fix any real `error`s
   first - they're genuine incompatibilities, not style.
2. **Resolve `[[vk::*]]` porting hints** by switching to the native spellings in §3, if you
   want them gone. Optional - the escapes keep working.
3. **Check for the divergences in §5** - particularly `half`/`float16`, matrix `*` vs `mul`,
   and any `SV_*` semantic that isn't behaving like a built-in.
4. **Modernize at your leisure** using §2 - `cbuffer` → `uniform`, `Wave*` → `subgroup_*`,
   and so on. Nothing forces this; do it when it improves the shader.

---

*This is the end of the language guide. For the terse normative rules, see
[`../LANGUAGE_SPEC.md`](../LANGUAGE_SPEC.md); for the compiler and output formats, see the
tooling documentation.*
