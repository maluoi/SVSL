# SVSL Language Specification

**Version 1.0**

> Status: every section has been verified against the implementation and its tests.
> This spec covers exactly the v1 scope — nothing aspirational. Deferred features
> (pointers, bindless, mesh/task, ray tracing, …) are intentionally absent; they live
> in docs/DECISIONS.md and docs/BACKLOG.md.

SVSL (SPIR-V Shading Language) is a shader language that maps directly to SPIR-V. It is
backwards compatible with the C-like subset of HLSL (StereoKit's shader corpus compiles
as-is, with porting hints), while providing native syntax for the modern SPIR-V concepts
that HLSL only reaches through `[[vk::*]]` escapes.

---

## 1. Design Principles

1. **Direct SPIR-V mapping** — language constructs correspond clearly to SPIR-V concepts.
2. **HLSL familiarity** — C-like HLSL code works with no or minimal changes; legacy forms
   emit suppressible `porting` hints under `-Wporting` (opt-in, off by default), never errors.
3. **vk:: escapes become language** — every `[[vk::*]]` attribute concept has a native
   SVSL spelling. The escapes still parse and work.
4. **Capability inference** — required SPIR-V capabilities/extensions are derived from
   what the code actually uses.
5. **Names over numbers** — reflection is name-based, so explicit numbers (binding slots,
   specialization ids) are optional interop tools, not requirements. Where a number is
   omitted, SVSL auto-assigns deterministically (declaration order, lowest free value,
   never colliding with explicit ones).
6. **Honest surface** — everything in this document is implemented and tested.

---

## 2. Lexical Structure

### Comments

```c
// single-line
/* multi-line */
```

`//--` at the start of a comment introduces a **metadata annotation** (§12).

### Identifiers

```
identifier = [a-zA-Z_][a-zA-Z0-9_]*
```

### Literals

```c
42        // int32          42u   // uint32
42L       // int64          42uL  // uint64
0xFF      // int32 (hex)    0b101 // int32 (binary)   0777 // int32 (octal, C rules)
3.14      // float32        3.14f // float32
3.14h     // half           3.14lf// float64
true, false
```

---

## 3. Types

### 3.1 Scalar types

| Type | Meaning |
|---|---|
| `bool` | boolean (stored as a 32-bit uint 0/1 in buffers; see below) |
| `int8` `int16` `int32` `int64` | signed integers, exact width |
| `uint8` `uint16` `uint32` `uint64` | unsigned integers, exact width |
| `float16` | **exactly** 16-bit float (§3.2) |
| `float32` `float64` | 32/64-bit floats |
| `half` | **at least** 16-bit float (§3.2) |
| `int` = `int32`, `uint` = `uint32`, `float` = `float32`, `double` = `float64`, `min16float` = `half` | aliases |

Non-32-bit widths add SPIR-V capabilities automatically (`Int8`, `Int16`, `Int64`,
`Float16`, `Float64`, plus 8/16-bit storage capabilities when used in buffers).

`OpTypeBool` is not allowed in externally visible memory, so `bool` members of
cbuffers, storage buffers, and push constants are stored as `uint` 0/1 and convert
on access (`!= 0` on load, select 1/0 on store), matching glslang. Reflection
reports them as uint-typed. Specialization-constant bools stay real booleans.

### 3.2 `half` vs `float16`

These are distinct types with different portability contracts:

- **`half`** means *at least 16-bit precision*. It compiles to a float32 decorated with
  `RelaxedPrecision`: hardware with fp16 support may compute at reduced precision, all
  other hardware runs it as float32. It is always **4 bytes** in buffers, so memory
  layouts never depend on device support. This is the portable default and what HLSL's
  `min16float` maps to. The `h` literal suffix produces a `half`.
- **`float16`** means *exactly 16 bits, always*. Requires the `Float16` capability (and
  16-bit storage capabilities when used in buffers); **2 bytes** in buffers. If the
  target environment cannot express it, compilation fails with a clear diagnostic. Use it
  when you control the hardware and want the bandwidth/perf win.

The compile option `--half=strict16` re-types every `half` as `float16`.

### 3.3 Vectors and matrices

```c
float2 uv;      half4  color;     int3 coords;    bool4 mask;
float4x4 viewProj;   float3x3 normalMat;   float3x4 skin;  // <rows>x<cols>
```

All scalar types take `2/3/4` vector suffixes and `RxC` matrix suffixes. Matrices are
column-major. Matrix element access: `m._m00`–`._m33` (0-based) and `._11`–`._44`
(1-based), plus `m[col]` returning a column vector. `(float3x3)m` truncation casts.

### 3.4 Arrays and structs

```c
float4   positions[64];
float4x4 bones[MAX_BONES];     // any constant expression, incl. specialization constants
float    data[];               // runtime-sized; last member of a storagebuffer only

static const float3 grad[] = { float3(0,0,0), float3(1,1,1) }; // size inferred: float3[2]

struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
};
```

An unsized array declared with an initializer list (const global or local) takes its
count from the list. Only the outermost dimension may be unsized this way.

### 3.4.1 Packed bit fields

A struct member may carry a `: width` suffix, which makes the whole struct **packed**:
its members become dense bit fields laid out LSB-first inside a sequence of backing
`uint32` words. This is the shader-side mirror of C bit fields, for the compute-shader
idiom of hand-packing many small values into a few words.

```c
struct Packed {
	uint  a : 4;       // raw unsigned, 4 bits
	int   b : 6;       // raw signed — sign-extended on read
	bool  f : 1;       // 1-bit flag
	float c : un10;    // unorm: [0,1] mapped across 10 bits
	half  d : sn6;     // snorm: [-1,1] mapped across 6 bits
	half  h : 16;      // raw 16-bit half bit pattern
	uint  e;           // plain member → its natural width (32 bits), its own word
};
```

The type on the **left** of the colon is the *resolve type* — the type you read and
write in shader code. The token on the **right** is the storage width and format:

| right side | meaning | resolve types |
| --- | --- | --- |
| `N` | raw: the low `N` bits of the integer, or a whole 16/32-bit float/half pattern | int/uint (any size), `bool` (`:1`), `half`/`float16` (`:16`), `float` (`:32`) |
| `unN` | unorm — a float in `[0,1]` quantized to `N` bits (`N` ≤ 24) | `float`, `half` |
| `snN` | snorm — a float in `[-1,1]` quantized to `N` bits (`N` ≤ 24) | `float`, `half` |

Layout rules, matching C:

- Fields pack densely from the LSB. A field that would straddle a 32-bit boundary is
  bumped to the start of the next backing word (**no field crosses a word**).
- A member with **no** `: width` occupies its type's natural width (`uint8` → 8 bits,
  `half` → 16, `uint`/`float` → 32), so packed and whole members mix freely. A member
  wider than 32 bits needs an explicit width.
- Writes **truncate** — `float c : un10 = 0.75` stores `round-toward-zero(0.75 · 1023)`.
  No clamping or rounding is inserted; clamp yourself if a value may exceed the range.
- The struct's storage is exactly its backing words (`Packed` above is 3 × `uint32` =
  12 bytes), so it round-trips with a matching CPU-side layout in a `StructuredBuffer`.
  Because the backing words are plain scalars, packed structs are alignment-neutral
  under every layout keyword — dense 4-byte words under the C-packed default, `pack1`,
  and `std430` alike, never needing a device feature; only `pack16` rounds the element
  stride up to 16.

### 3.4.2 Enums

An `enum` declares named integer constants and, when named, an integer **type alias**:

```c
enum : int16 { One, Two } option = One;   // anonymous, with an inline variable
enum Mode : uint { ModeA = 1, ModeB = 4, ModeC };  // ModeC = 5
enum { Red, Green, Blue };                // just three global constants
```

- Constant names are **global** (the HLSL unscoped-`enum` flavor), not enum-scoped:
  reference `One`, `ModeA`, `Red` directly.
- Values follow C: they start at `0` and increment by one unless an item gives an
  explicit constant-integer initializer, after which counting continues from that value.
- The underlying type after `:` must be an integer scalar; it defaults to `int` when
  omitted. A named enum is an **alias** for that type — `Mode m = ModeB;` declares an
  ordinary `uint`. There is no distinct enum type and no nominal type checking: an enum
  value is its underlying integer, converts freely to and from it, and can be compared,
  switched on, or used anywhere an integer is expected.
- An anonymous `enum { … } v;` declares `v` with the underlying type (here `int`); the
  form exists to introduce constants (and optionally a variable) without naming a type.
- Because an enum is just an integer, an enum type is a valid
  [packed bit-field](#341-packed-bit-fields) resolve type — `Facing dir : 2;` stores the
  constant in a raw 2-bit field.

An `enum { … }` may also be written **inline in any type position** — a function
parameter, a struct member, or a local variable — not only as a standalone declaration:

```c
uint doThing(enum { Thing1, Thing2 } which) { ... }   // parameter
struct Widget { enum : uint { Hidden, Shown } vis : 1;  uint id : 31; };
void f() { enum Dir { Up, Down } d = Down; ... }       // local
```

Wherever it appears, an inline enum's constants still register **globally** (matching the
unscoped-name rule), so two inline enums that share a constant name in the same translation
unit collide. The type position itself just resolves to the underlying integer.

Enum constants are compile-time constants: they fold into literals at use sites and are
valid in constant contexts (array sizes, `switch` case labels, specialization defaults).

### 3.5 Resource types

**Sampled textures** (SPIR-V `OpTypeImage`, Sampled=1) — read through a `Sampler`:

```c
Texture1D<float4>  Texture2D<float4>  Texture3D<float4>  TextureCube<float4>
Texture1DArray<T>  Texture2DArray<T>  TextureCubeArray<T>
Texture2DMS<T>                        // multisampled; read with Load(coord, sample)
```
The element type defaults to `<float4>`. `Sampler` and `SamplerComparison` are the
sampler types (`SamplerState`/`SamplerComparisonState` = HLSL aliases). `Texture2DMS`
also accepts HLSL's sample-count template argument (`Texture2DMS<float4, 4>`) as a
source-level annotation — the pipeline's sample count is what matters, matching DXC.
Multisampled textures cannot be `Sample`d; fetch a specific sample with `Load`.

**Storage images** (Sampled=2) — direct load/store, no sampler. The optional second
argument is the **native spelling of `[[vk::image_format]]`**:

```c
Image2D<float4>         img;            // format inferred/unknown
Image2D<float4, rgba8>  img8;           // explicit format
Image3D<T,F>  ImageCube<T,F>  Image1DArray<T,F>  Image2DArray<T,F>  ImageCubeArray<T,F>
```
Format names follow SPIR-V image formats, lowercase: `rgba32f rgba16f rg32f r32f rgba8
rgba8_snorm r11g11b10f rgba16 rgba32i rgba32u r32i r32u ...`. `RWTexture*<T>` = HLSL
aliases of `Image*<T>`.

**Subpass inputs** (input attachments) — tile-local reads of a previous render-pass
attachment at the current fragment position; no sampler, no bandwidth cost on tilers.
Fragment stage only. Native spelling of `[[vk::input_attachment_index(N)]]`:

```c
SubpassInput<float4>   src;             // attachment index auto-assigned
SubpassInput<float4,1> src;             // explicit input_attachment_index 1
SubpassInputMS<float4> srcMS;           // multisampled attachment
float4 prev = src.Load();               // reads at the current fragment position
float4 s2   = srcMS.Load(2);            // one sample of an MS attachment
```
`SubpassLoad` is the DXC-compatible alias of `Load`. Subpass inputs have no
dimensions to query — `GetDimensions` on one is a compile error.

**Tile images** (SPV_EXT_shader_tile_image) — like subpass inputs, but reading the
*current* render pass's color attachments mid-pass, without a pipeline barrier or
input-attachment descriptor. Fragment stage only. The second argument is the color
attachment location (default 0):

```c
TileImage<float4>    color0;            // color attachment location 0
TileImage<float4, 1> color1;            // color attachment location 1
float4 behind = color0.Load();          // framebuffer-local read
float  d = tile_depth();                // current depth attachment value
uint32 s = tile_stencil();              //   ... and stencil
```
Tile images are tile memory, not descriptors: `register()` or a binding on one is a
compile error.

**Tile attachments** (VK_QCOM_tile_shading) — attachments processed per-tile while the
framebuffer content sits in on-die tile memory. Unlike tile images they keep ordinary
set/binding descriptors; only their storage class changes (`TileAttachmentQCOM`).
Declared with the `[tile_attachment]` attribute on a `Texture2D` or `RWTexture2D`
(2D, single-layer, non-multisampled only). Fragment and compute stages only:

```c
[tile_attachment] Texture2D<float4>   lastFrame;  // sampled tile reads
[tile_attachment] RWTexture2D<float4> color;      // storage read/write

uint2 off   = tile_offset_qcom();      // framebuffer coords of the tile's top-left texel
uint3 dim   = tile_dimension_qcom();   // tile size in pixels; z = layer count
uint2 apron = tile_apron_size_qcom();  // active apron size (set by the render pass)
```

Per-tile compute dispatch uses `[tile_shading_rate_qcom(x, y, z)]` in place of
`[numthreads]` (§6), and fragment entries may opt out of rasterization-order reads
with `[non_coherent_tile_reads_qcom]`. The render pass apron size itself has no
shader-side representation — it is renderer state, carried by the `//--apron`
metadata key (§12).

**Buffers** are declared as blocks (§4) or the HLSL-alias object forms
`StructuredBuffer<T>` / `RWStructuredBuffer<T>` / `Buffer<T>`.

---

## 4. Buffers, Storage, and Layout

### 4.1 Buffer blocks

```c
// Uniform buffer — read-only, pack16 layout required (and implied)
uniform SceneData : register(b0, space0) {
	float4x4 viewProj;
	float3   cameraPos;
	float    time;
};

// Storage buffer — std430 default; runtime-sized array allowed as last member
readonly storagebuffer Vertices : register(t0, space1) {
	Vertex vertices[];
};

// Push constants — native spelling of [[vk::push_constant]]; no descriptor
pushconstant Draw {
	float4x4 model;
	uint32   materialId;
};
```

Members are accessed directly by name (`viewProj`, `vertices[i]`) — the block name is a
grouping, not a namespace. `cbuffer N : register(b#) { }` is the HLSL alias of `uniform`.

### 4.2 Layout keywords

A layout keyword prefixes the declaration, like any other modifier — on blocks and
on the object-form buffers alike:

```c
pack16 storagebuffer Particles : register(u0) { particle_t items[]; };
pack16 RWStructuredBuffer<particle_t> particles : register(u1);
scalar readonly storagebuffer Raw { float data[]; };   // standards-name alias
```

Most modes have both an SVSL `packN` spelling and a standards-name alias; `std430` is
spelled by its standards name only. Any spelling is accepted everywhere a layout keyword is:

| Keyword | Alias | Rule | Vulkan requirement |
|---|---|---|---|
| `pack1` | `scalar` | members align to their scalar size; `float3` = 12 bytes, 4-aligned; matches C struct layout | core when no vector straddles a 16-byte boundary, else the `scalarBlockLayout` device feature |
| `pack8` | `relaxed` | vectors align to min(natural, 8) | same rule as `pack1` |
| `pack16` | `std140` | std140 — `float3`/`float4` 16-aligned, array elements 16-stride | core |
| `std430` | — | vectors natural-aligned, no 16-stride array rounding | core |

The aliases are context-sensitive, not reserved words — `scalar` and `relaxed` stay
usable as identifiers.

Defaults, by declaration form:

- `uniform` blocks are `pack16` only (constant-buffer hardware rules).
- `storagebuffer` / `pushconstant` blocks default to **std430**.
- `StructuredBuffer<T>` / `RWStructuredBuffer<T>` / `Buffer<T>` elements default to
  **C layout where C layout is free** — a plain C struct with the same members
  matches byte-for-byte, and reflected `element_size` equals the C `sizeof`.
  "C layout" is not a fifth mode: it **is** `pack1`/`scalar`. The default differs
  from writing `pack1` only in policy. Some valid pack1 layouts (a vector
  improperly straddling a 16-byte boundary, or a stride off its std430 alignment)
  are only loadable on devices with the `scalarBlockLayout` feature — and a bare
  declaration names no layout, so it must not silently acquire a device
  dependency the author never chose. The default therefore refuses those layouts
  with a compile error at the declaration; writing `pack1` supplies the consent,
  emits the same layout, and records the requirement in the SKS feature mask
  (bit 16) — the same contract as `float16`, where typing the feature is what
  opts into its device requirement.
  One C-matching caveat: SVSL `bool` is stored as a 32-bit uint (§3.1) — pair it
  with `uint32_t`/`int32_t` on the CPU side, never C `bool`.

Violations are compile errors. `[offset(N)]` on a member (native spelling of
`[[vk::offset]]`) sets an explicit byte offset; it may only increase offsets.

How the same struct lands under each mode:

```c
struct particle_t {
	uint   id;    // C: offset 0
	float2 pos;   // C: offset 4
	float  scale; // C: offset 12, sizeof 16
};
```

| Mode | `id` | `pos` | `scale` | array stride |
|---|---|---|---|---|
| `pack1` / `scalar` | 0 | 4 | 12 | 16 |
| `pack8` / `relaxed` | 0 | 8 | 16 | 24 |
| `pack16` / `std140` | 0 | 8 | 16 | 32 |
| `std430` | 0 | 8 | 16 | 24 |

Only `pack1` is byte-identical to the natural C layout — under the other modes a
matching C struct needs explicit padding members. `pack8` happens to match std430
here because `float2` already aligns to 8; the two differ on `float3`/`float4`
members, which std430 aligns to 16 and `pack8` to 8. `particle_t` never straddles,
so the C-packed default carries no device requirement; a struct like
`{ float a; float4 b; }` does straddle (`b` at offset 4) and is refused by the
default with a fix-it — reorder, pad, or opt in with `pack1`.

### 4.3 Access and variable storage

Access modifiers on buffer blocks and images: `readonly`, `writeonly`, `coherent`,
`volatile`.

```c
static const float PI = 3.14159;    // compile-time constant
static float3 accum;                // private per-invocation global
workgroup float4 tile[8][8];        // shared memory ('groupshared' = alias)
```

Bare globals with initializers form the implicit `$Global` uniform buffer (singular, matching
glslang/skshaderc), and their initializers become material defaults in reflection (§12):

```c
float4 color     = {1,1,1,1};
float  metallic  = 0;
```

---

## 5. Bindings

```c
Texture2D tex : register(t0);           // HLSL style: prefix letter + slot
Texture2D tex : register(t0, space1);   //   ... with descriptor set
Texture2D tex : register(0, 1);         // SVSL direct style: (binding, set) — no offsets
```

With prefix letters, bindings are offset to avoid cross-type collisions (matching
skshaderc): `b` → +0, `t`/`s` → +100, `u` → +200. A `Texture2D` at `t0` and a `Sampler`
at `s0` pair into one combined binding; the `name`/`name_s` naming convention is
conventional, pairing is by slot. The direct `(binding, set)` form applies no offsets —
it is the native spelling of `[[vk::binding(b, s)]]`.

**`register()` is optional.** Runtimes bind by name through reflection, so most shaders
don't care which slot a resource lands in. Resources without a `register()` are
auto-assigned after all explicit registers are placed: declaration order, lowest free
slot in the resource's register class (`b`/`t`/`s`/`u`), set 0. Assignment is
deterministic — the same source always produces the same slots — and never collides
with explicit registers (so StereoKit's reserved slots from `stereokit.hlsli` stay
untouched). A texture and its paired sampler are assigned together.

```c
Texture2D diffuse;                      // auto: first free t slot
Sampler   diffuse_s;                    // auto: paired with diffuse
uniform Params { float4 color; };       // auto: first free b slot
```

---

## 6. Entry Points, Stages, and Attributes

An entry point is marked by a stage attribute, **or** matched by name (defaults `vs`,
`ps`, `cs`; configurable). One compiled module is produced per entry point. In `.sks`
output the module's `OpEntryPoint` is renamed to the canonical stage name (`vs`/`ps`/
`cs`) whatever the source function is called — sk_renderer binds entry points by those
fixed names. Raw `.spv` output keeps the source name.

```c
[vertex]              VSOut  my_vs(VSIn input) { ... }
[fragment]            float4 my_ps(VSOut input) : SV_Target { ... }   // [pixel] = alias
[compute(8, 8, 1)]    void   my_cs() { ... }                          // workgroup size
```

`[numthreads(x,y,z)]` is the HLSL alias for compute sizes and combines with name-based
entry detection. The workgroup size may reference specialization constants, but such a
reference is baked to the constant's default (literal `LocalSize`), not kept specializable.

### Entry-point attributes

```c
[wave_size(32)]                   // required subgroup size: power of two, 4–128
[compute(64, 1, 1)]               //   (reflection-only; applied via VK_EXT_subgroup_size_control; `//--wave_size` = alias)
void my_cs() { ... }

[early_depth_stencil]             // EarlyFragmentTests execution mode: depth/stencil
[fragment]                        //   test runs before the shader; shader may not write depth
float4 my_ps(PSIn i) : SV_Target { ... }

[tile_shading_rate_qcom(2, 2, 1)] // VK_QCOM_tile_shading per-tile dispatch. REPLACES
void my_tile_cs() { ... }         //   [numthreads] (the implementation derives the
                                  //   workgroup shape); x/y must be powers of two

[non_coherent_tile_reads_qcom]    // NonCoherentTileAttachmentReadQCOM: tile attachment
float4 my_tile_ps() : SV_Target;  //   reads may ignore rasterization order (faster)
```

### Specialization constants

Runtime-settable constants (Vulkan specialization info), reflected by **name** into the
`.sks` spec-constant table with their defaults. 32-bit scalars and bool; usable in
constant expressions, array sizes, and compute workgroup sizes.

```c
specialization const uint32 TILE   = 16;     // id auto-assigned
specialization const bool   SHADOW = false;  // id auto-assigned
[specialization(7)] const float CUTOFF = 0.5;    // explicit id, for interop
```

The bare `specialization` keyword auto-assigns ids (declaration order, lowest free id,
never colliding with explicit ones) — since runtimes set these by name via reflection,
the id is usually an implementation detail. `[specialization(N)]` pins the id when an
external system expects a specific one; it is the native spelling of
`[[vk::constant_id(N)]]`.

Integer and bool expressions over specialization constants stay specializable:
`TILE * TILE`, `TILE / 2u + 1u`, or `TILE > 8u ? 1u : 0u` compile to
`OpSpecConstantOp`, so they are re-evaluated when the constant is set at pipeline
creation. Float math over spec constants is computed at runtime (SPIR-V does not
allow float `OpSpecConstantOp`), and storing through a named local variable also
drops to runtime — only direct expressions fold.

### Statement attributes

`[unroll]`, `[loop]`, `[branch]`, `[flatten]`, `[fastopt]` — accepted everywhere HLSL
accepts them and validated for placement (a statement attribute on a declaration warns).
The four control-flow hints are lowered to the matching SPIR-V control mask, carrying
author intent the backend cannot infer:

| attribute | construct | SPIR-V mask |
|---|---|---|
| `[flatten]` | `if` | `OpSelectionMerge … Flatten` (predicate rather than branch) |
| `[branch]`  | `if` | `OpSelectionMerge … DontFlatten` (keep the branch) |
| `[unroll]`  | `for`/`while`/`do` | `OpLoopMerge … Unroll` |
| `[loop]`    | `for`/`while`/`do` | `OpLoopMerge … DontUnroll` |

`[fastopt]` has no SPIR-V equivalent and remains advisory. `[unroll(n)]` (a fixed count)
is not yet distinguished from `[unroll]`.

### `precise`

`precise` on a local variable declaration forbids floating-point contraction (fused
multiply-add) in the expression that computes it, so the result matches the written
arithmetic bit-for-bit — for numerical stability or crack-free geometry between adjacent
triangles. Each contributing float arithmetic op is emitted with `OpDecorate …
NoContraction`. It applies to the initializing expression (`precise float x = a*b + c;`);
a contribution routed through a separate, non-`precise` local is not tracked across the
memory round-trip — mark that local `precise` too.

### `[[vk::*]]` escape summary

| Escape (parses; porting-warns under `-Wporting`) | Native form |
|---|---|
| `[[vk::binding(b, s)]]` | `register(b, s)` |
| `[[vk::push_constant]]` | `pushconstant { }` |
| `[[vk::constant_id(N)]]` | `[specialization(N)]`, or bare `specialization` (auto-id) |
| `[[vk::image_format("rgba8")]]` | `Image2D<float4, rgba8>` |
| `[[vk::input_attachment_index(N)]]` | `SubpassInput<T, N>`, or `SubpassInput<T>` (auto-index) |
| `[[vk::location(N)]]` | `[location(N)]` |
| `[[vk::offset(N)]]` | `[offset(N)]` |

Unknown `[[vk::*]]` attributes are compile errors naming the attribute.

---

## 7. Stage Interface

### 7.1 Locations

Struct members used for stage IO get locations by declaration order, skipping system
values. `[location(N)]` overrides. Legacy semantics (`NORMAL0`, `TEXCOORD2`, `COLOR0`…)
are accepted; they don't affect locations but are captured for vertex-input reflection.

```c
struct VSOut {
	float4        pos   : SV_Position;  // builtin — no location
	float2        uv;                   // location 0
	flat uint32   id;                   // location 1, no interpolation
	noperspective float2 screen;        // location 2
};
```

Interpolation modifiers (fragment inputs / vertex outputs): `flat` (`nointerpolation` =
alias), `noperspective`, `centroid`, `sample`.

The `invariant` qualifier (Invariant decoration) forces bit-identical results for the
same expression across entry points — put it on `SV_Position` when multiple passes
recompute the same position (depth prepass, shadow passes) to prevent z-fighting:

```c
struct VSOut {
	invariant float4 pos : SV_Position;
	...
};
```

### 7.2 System values

Case-insensitive (corpus uses both `SV_Position` and `SV_POSITION`).

| Semantic | SPIR-V builtin | Stage |
|---|---|---|
| `SV_Position` | Position / FragCoord | VS out / FS in |
| `SV_Target`, `SV_Target0..7` | fragment output location 0..7 | FS out |
| `SV_Depth` | FragDepth | FS out |
| `SV_DepthGreaterEqual` | FragDepth + `DepthGreater` execution mode | FS out |
| `SV_DepthLessEqual` | FragDepth + `DepthLess` execution mode | FS out |
| `SV_VertexID` | VertexIndex | VS in |
| `SV_InstanceID` | InstanceIndex | VS in |
| `SV_ViewID` | ViewIndex (+`MultiView` capability) | VS/FS in |
| `SV_IsFrontFace` | FrontFacing | FS in |
| `SV_SampleIndex` / `SV_Coverage` | SampleId / SampleMask | FS |
| `SV_PrimitiveID` | PrimitiveId | FS in |
| `SV_DispatchThreadID` | GlobalInvocationId | CS |
| `SV_GroupThreadID` | LocalInvocationId | CS |
| `SV_GroupID` | WorkgroupId | CS |
| `SV_GroupIndex` | LocalInvocationIndex | CS |

The conservative-depth variants promise the written depth is only ≥ (or ≤) the
rasterized depth, which lets the GPU keep early-Z culling enabled even though the shader
writes depth.

Multiview: `SV_ViewID` is how StereoKit's `sk_ids_t { uint inst : SV_InstanceID; uint
view : SV_ViewID; }` pattern works; using it emits the `MultiView` capability.

---

## 8. Expressions and Statements

C expression grammar: arithmetic `+ - * / %`, comparison, logical `&& || !`, bitwise
`& | ^ ~ << >>`, assignment ops, ternary, `++ --`. Vector ops are component-wise;
`mul()` is matrix/vector multiplication (all 14 HLSL overloads).

### Swizzles

Vector components are accessed by name with either set — `.xyzw` or `.rgba` (equivalent;
sets can't mix within one swizzle). A swizzle selects 1–4 components in any order:

```c
float4 c = float4(1, 0.5, 0.2, 1);
float3 rgb  = c.rgb;         float2 rg = c.rg;
float3 bgr  = c.bgr;         float4 rrrr = c.rrrr;   // repetition ok when reading
float  red  = c.r;           // single component yields a scalar
c.xw   = float2(0, 0);       // swizzle write (partial assignment)
c.rgb  = c.bgr;              // write via reordered swizzle
c.xx   = uv;                 // ERROR: duplicate components in a write
c.xg   = uv;                 // ERROR: mixed name sets
```

Swizzling a scalar is allowed for broadcast reads (`f.xxx`). Matrix element access uses
`._m00`–`._m33` / `._11`–`._44` (§3.3), one element at a time (no matrix swizzles).

Constructors: `float4(v3, 1)`, broadcast `float4(1)`, `float4x4(1)` identity, column-wise
matrix construction.

Statements: `if/else`, `for`, `while`, `do`, `switch/case/default`, `break`, `continue`,
`return`, and:

- `discard` — kill the fragment (`OpKill` semantics; also the `clip(x)` intrinsic).
- `demote` — demote to helper invocation (`DemoteToHelperInvocation` capability):
  derivatives keep working after it, unlike `discard`. `is_helper_invocation()` queries.

Functions support `in` (default), `out`, `inout` parameters. **Resource types (textures,
images, samplers) may be passed as function parameters** — calls are fully inlined during
compilation, so the resource resolves to its global declaration:

```c
float4 sample_fade(Texture2D t, Sampler s, float2 uv, float f) {
	return t.Sample(s, uv) * f;
}
```

Forward references between functions need no prototypes (two-pass parsing). Recursion is
a compile error (SPIR-V forbids it; full inlining requires it).

### Inline SPIR-V (`spirv_asm`)

`spirv_asm(T) { … }` is an expression yielding a value of type `T`, whose body is raw
SPIR-V spliced into the function. `spirv` is an accepted alias for the keyword. It is the
escape hatch to any opcode the language has no syntax for.

```c
float s = spirv_asm(float) {
    OpExtInst $$float %result glsl450 13 $angle;   // GLSL.std.450 Sin
};
```

Instructions are written in **binary operand order** (result-type and result-id operands
included; there is no `%r = Op` sugar) and terminated by `;`. Operands:

- `OpXxx` — the opcode mnemonic that begins each instruction (any name in the SPIR-V spec).
- `$$type` — an SVSL type → its SPIR-V type id.
- `$value` or `$(expr)` — an SVSL value → its SPIR-V result id.
- `%id` — a named local; first use allocates a fresh id. `%result` (required) carries the
  block's value.
- an integer literal — a raw 32-bit word (enum values and immediates are numeric).
- a `"string"` literal — packed as nul-terminated UTF-8 words.
- `glsl450` — the `GLSL.std.450` ext-instruction import id.

A block can state its own prerequisites: `OpCapability <int>;` and
`OpExtension "<name>";` instructions are routed to the module's capability/extension
declaration streams (deduplicated) instead of the function body, so a block using an
opcode the language has no syntax for is self-contained:

```c
uint2 t = spirv_asm(uint2) {
    OpCapability 5055;                    // ShaderClockKHR
    OpExtension "SPV_KHR_shader_clock";
    OpReadClockKHR $$uint2 %result $(3u); // Scope Subgroup
};
```

The block must define `%result`; its type `T` is asserted (verified downstream by
`spirv-val`). Operands are SSA values and types only — resources, pointers,
64-bit-literal operands, and named enumerants are not yet expressible. A `spirv_asm` block
is opaque to the optimizer: never merged, reordered, or eliminated. It is SVSL-dialect
(no glslang equivalent).

---

## 9. Resource Access

```c
// Sampled textures (with Sampler)
float4 c = tex.Sample         (smp, uv);
float4 c = tex.SampleLevel    (smp, uv, lod);
float4 c = tex.SampleBias     (smp, uv, bias);
float4 c = tex.SampleGrad     (smp, uv, ddx, ddy);
float  s = tex.SampleCmp          (cmpSmp, uv, ref);
float  s = tex.SampleCmpLevelZero (cmpSmp, uv, ref);
float4 g = tex.Gather / GatherRed / GatherGreen / GatherBlue / GatherAlpha / GatherCmp (...);
float4 t = tex.Load(int3(x, y, mip));      // unfiltered fetch, no sampler
float4 f = tex[int2(x, y)];                // fetch shorthand: Load at mip 0 (not Cube/MS)
float4 s = texMS.Load(int2(x, y), sample); // Texture2DMS: fetch one sample
float  l = tex.CalculateLevelOfDetail(smp, uv);            // clamped LOD query
float  u = tex.CalculateLevelOfDetailUnclamped(smp, uv);
uint2  d = tex.GetDimensions();

// GetDimensions also takes HLSL's out-parameter forms:
tex.GetDimensions(w, h);                   // uint or float outs
tex.GetDimensions(mip, w, h, mipCount);    // per-mip query
texMS.GetDimensions(w, h, sampleCount);    // multisampled: sample count, no mips
verts.GetDimensions(count, stride);        // structured buffers: element count + stride

// Storage images
float4 v = img.Load(int2(x, y));
img.Store(int2(x, y), v);
img[int2(x, y)] = v;                       // Store shorthand
uint32 old = imgU32.InterlockedAdd(coord, 1);   // image atomics (r32u/r32i formats)

// Buffers
Vertex v = vertices[i];
counts[i] += 1;

// QCOM image processing (VK_QCOM_image_processing / _processing2; Texture2D only)
float4 w = tex.SampleWeightedQCOM(smp, uv, weights);        // weights: Texture2DArray
float4 b = tex.BoxFilterQCOM     (smp, uv, boxSize);        // boxSize: float2, in texels
float4 e = target.BlockMatchSADQCOM(smp, targetCoord, reference, refCoord, blockSize);
float4 e = target.BlockMatchSSDQCOM(...);                   // coords/blockSize: uint2 texel coords
float4 e = target.BlockMatchWindowSADQCOM(...);  // + WindowSSD, GatherSAD, GatherSSD
```

The QCOM operations infer their SPIR-V decorations from use, like glslang: the
weights argument becomes a `WeightTextureQCOM`, block-match targets/references become
`BlockMatchTextureQCOM`, and Window-form samplers become `BlockMatchSamplerQCOM`.
Decorated resources are **exclusive to their op family** — the runtime binds them
through dedicated descriptor types and `VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM`
samplers, so mixing (e.g. `Sample` on a block-match texture, or one sampler used by
both Window and non-Window forms) is a compile error. A shared sampler serves both
sampled images of a block match. These methods raise the module to SPIR-V 1.4 (the
extension's floor); everything else stays 1.3.

---

## 10. Intrinsic Functions

### Math (component-wise unless noted)

```
sin cos tan asin acos atan atan2 sinh cosh tanh sincos(x, out s, out c)
pow exp exp2 log log2 log10 sqrt rsqrt rcp radians degrees
abs sign floor ceil trunc round frac fmod modf(x, out ip) ldexp
frexp(x, out exp) frexp_exp frexp_mant
min max clamp saturate lerp step smoothstep fma mad
length distance normalize dot cross reflect refract faceforward   // vector
transpose determinant inverse                                     // matrix
any all select isnan isinf
countbits reversebits firstbithigh firstbitlow
bitfield_extract(value, offset, bits) bitfield_insert(base, value, offset, bits) // integer bit fields
asfloat asuint asint f16tof32 f32tof16
pack_unorm4x8 unpack_unorm4x8 pack_snorm4x8 unpack_snorm4x8 pack_half2x16 unpack_half2x16
clip(x)          // discard if any component < 0
```

`frexp_exp`/`frexp_mant` are expression forms of `frexp` returning just the exponent
or just the mantissa — useful where HLSL's out-parameter form is awkward. The legacy
D3D9 helpers `lit`, `dst`, and `msad4` are recognized but rejected with a diagnostic
rather than implemented.

### Derivatives (fragment only)

```
ddx ddy ddx_coarse ddy_coarse ddx_fine ddy_fine fwidth
```

### Subgroup operations

Add `GroupNonUniform*` capabilities by use. HLSL `Wave*`/`Quad*` intrinsics are
porting-warned aliases (`WaveActiveSum` → `subgroup_add`, `WaveGetLaneIndex` →
`subgroup_lane_id`, `WaveReadLaneFirst` → `subgroup_broadcast_first`, …).

```c
// builtins (read-only variables)
uint32 subgroup_size;  uint32 subgroup_lane_id;  uint32 subgroup_id;  uint32 num_subgroups;

bool  subgroup_elect();
uint4 subgroup_ballot(bool v);          bool subgroup_all(bool v);
bool  subgroup_any(bool v);             bool subgroup_all_equal(T v);
uint32 subgroup_ballot_bit_count(bool v);            // lanes where v is true
uint32 subgroup_ballot_exclusive_bit_count(bool v);  //   ... at a lower lane than mine
T     subgroup_broadcast(T v, uint32 lane);   T subgroup_broadcast_first(T v);

T subgroup_add(T) subgroup_mul(T) subgroup_min(T) subgroup_max(T)
  subgroup_and(T) subgroup_or(T) subgroup_xor(T)                  // reductions
T subgroup_inclusive_add(T) subgroup_exclusive_add(T) ... (mul/min/max/and/or/xor)
T subgroup_clustered_add(T, uint32 clusterSize) ... (mul/min/max/and/or/xor)
T subgroup_shuffle(T, uint32 lane)  subgroup_shuffle_xor(T, uint32 mask)
  subgroup_shuffle_up(T, uint32 d)  subgroup_shuffle_down(T, uint32 d)

T quad_broadcast(T, uint32 lane)    quad_swap_horizontal(T)
  quad_swap_vertical(T)             quad_swap_diagonal(T)
```

A required subgroup size can be declared via metadata: `//--wave_size = 32`
(power of two, 4–128; lands in `.sks` reflection).

### Atomics

On `storagebuffer` members, `workgroup` variables, and storage images. HLSL
`Interlocked*` forms are porting-warned aliases.

```c
T atomic_add(ref T dest, T v)   atomic_sub  atomic_min  atomic_max
  atomic_and  atomic_or  atomic_xor  atomic_exchange
T atomic_compare_exchange(ref T dest, T compare, T v)
```

All return the prior value. Destinations are 32-bit integer scalars, plus `float32`
for `atomic_add`, `atomic_min`, `atomic_max`, and `atomic_exchange` — float exchange
is core SPIR-V; the others use SPV_EXT_shader_atomic_float and require the matching
device features (see §15). The remaining ops have no float form and say so in their
diagnostic.

**Memory scope is inferred from the destination's storage**: an atomic on a
`workgroup` variable synchronizes only within the workgroup (Workgroup scope);
buffer and image atomics are device-visible (Device scope). No syntax selects it.

**Memory order** is an optional trailing argument on the *native* forms —
`relaxed` (the default), `acquire`, `release`, `acq_rel`:

```c
atomic_exchange(lock, 1u, acquire);            // acquire the lock
atomic_exchange(lock, 0u, release);            // release it
atomic_add(counter, 1u);                       // relaxed (default)
```

Relaxed emits no memory-semantics bits (bit-identical to glslang); a stated order
emits its ordering bits combined with the destination's storage-class memory bit.
`seq_cst` is rejected — the Vulkan memory model has no sequential consistency, so
`acq_rel` is the strongest available. Order is native-only; the `Interlocked*`
aliases keep their HLSL out-parameter form.

### Barriers

```c
workgroup_barrier();            // execution + workgroup memory (GroupMemoryBarrierWithGroupSync)
workgroup_memory_barrier();     // workgroup memory only        (GroupMemoryBarrier)
device_memory_barrier();        // device memory only           (DeviceMemoryBarrier)
all_memory_barrier();           // all memory, no sync          (AllMemoryBarrier)
device_memory_barrier_sync();   // execution + device memory    (DeviceMemoryBarrierWithGroupSync)
all_memory_barrier_sync();      // execution + all memory       (AllMemoryBarrierWithGroupSync)
```

---

## 11. Preprocessor and Includes

A conventional C-style preprocessor runs before the language (diagnostics map back to
original file/line through it):

- `#include "file"` and `#include <file>` — resolved identically, via the embedder's
  include callback / `-i` search paths.
- `#define NAME value`, basic function-like `#define F(a, b) ...` (no variadics, no
  `#`/`##` operators), `#undef`.
- `#if #ifdef #ifndef #elif #else #endif`, `defined()`, full constant expressions.
- `#pragma once`.

`include "file"` (no `#`) is the language-level form; it resolves through the same
callback but is processed as a declaration, not textual paste, and is idempotent.

---

## 12. Material Metadata (`//--`)

Comment annotations attach reflection metadata consumed by StereoKit; format
`//--<param>[:<tag>] = <value>` (also inside `/* */`):

```c
//--name = sk/unlit               // unique shader name
//--diffuse = white               // default texture (white/black/gray/rough/flat/normal/…)
//--color:color = 1,1,1,1         // 'color' tag + default value
//--uv_scale: range(0, 2) = 0.5   // UI hint, stored in the member's `extra` field
//--wave_size = 32                // required subgroup size (alias of [wave_size(32)])
//--apron = 2                     // VK_QCOM_tile_shading render-pass apron, W or W, H
```

Bare-global initializers (§4.3) provide numeric defaults; `//--` values override and
extend them. Both are baked into the `.sks` container.

`name`, `wave_size`, and `apron` are the reserved program-level keys; every other key
must match a parameter or resource. `//--apron` is purely renderer-facing: the apron
has no shader-side representation (it is `VkRenderPassTileShadingCreateInfoQCOM::
tileApronSize`, set at render pass creation), so the shader that needs neighborhood
reads near tile edges declares the size it expects and the renderer applies it.
Shaders read the active size back via `tile_apron_size_qcom()` (§3.5). Setting it in
a shader with no tile-shading usage warns.

---

## 13. Diagnostics

Severities: `error`, `warning`, `porting`, `info`. `porting` marks legacy-HLSL spellings
with a native alternative (never fires when no alternative exists); it is **opt-in**
(`-Wporting` / `svsl_options_t.porting_hints`, off by default) and covers `[[vk::*]]`
escapes, legacy resource/scalar type spellings, and HLSL intrinsic aliases. It is a
distinct severity precisely so embedders can filter it — the API reports severity per
diagnostic, and svslc prints it as its own category. All diagnostics carry
file/line/column surviving preprocessing. Sema recovers after errors and reports
batches, not just the first failure.

## 14. Legacy HLSL Alias Summary

Every form below is accepted. Under `-Wporting` (§13), the **type spellings** (`min16float`,
`SamplerState`, `StructuredBuffer`/`RWStructuredBuffer`, `RWTexture*`), **intrinsic aliases**
(`Wave*`/`Quad*`, `Interlocked*`, `GroupMemoryBarrier*`), and **`[[vk::*]]`** escapes emit a
`porting` hint; the declaration keywords (`cbuffer`, `groupshared`, `nointerpolation`),
attribute aliases, and legacy semantics are accepted silently (their hints are backlog).

| HLSL (accepted) | SVSL-native |
|---|---|
| `cbuffer N : register(b0) { }` | `uniform N : register(b0, space0) { }` |
| `StructuredBuffer<T>` / `RWStructuredBuffer<T>` | `storagebuffer [readonly] N { T x[]; }` |
| `Buffer<T>` | `readonly storagebuffer` block |
| `groupshared` | `workgroup` |
| `SamplerState` / `SamplerComparisonState` | `Sampler` / `SamplerComparison` |
| `RWTexture2D<T>` (etc.) | `Image2D<T[, format]>` |
| `min16float*` | `half*` |
| `nointerpolation` | `flat` |
| `[numthreads(x,y,z)]` | `[compute(x,y,z)]` |
| `[earlydepthstencil]` | `[early_depth_stencil]` |
| `subpass.SubpassLoad(...)` | `subpass.Load(...)` |
| `Wave*` / `Quad*` intrinsics | `subgroup_*` / `quad_*` |
| `Interlocked*` | `atomic_*` |
| `GroupMemoryBarrierWithGroupSync` etc. | `workgroup_barrier()` etc. |
| `[[vk::*]]` attributes | see §6 table |
| legacy semantics (`NORMAL0`, `TEXCOORD#`, …) | declaration-order locations / `[location(N)]` |

## 15. Capability Inference

| Feature used | SPIR-V capability / extension |
|---|---|
| `float16` | Float16 (+ StorageBuffer16BitAccess / UniformAndStorageBuffer16BitAccess in buffers) |
| `half` | (none — RelaxedPrecision decoration only) |
| `int8`/`uint8`, `int16`/`uint16`, `int64`, `float64` | Int8, Int16, Int64, Float64 (+ storage caps) |
| `pack1`/`pack8` layout breaking core relaxed rules | *(no SPIR-V capability — VK_EXT_scalar_block_layout recorded as `.sks` feature-mask bit 16)* |
| `SV_ViewID` | MultiView (core in SPIR-V 1.3 — no extension) |
| `demote` / `is_helper_invocation` | DemoteToHelperInvocation |
| subgroup ops | GroupNonUniform + Vote/Ballot/Arithmetic/Shuffle/Clustered/Quad as used |
| storage images | StorageImageExtendedFormats etc. as needed by format |
| image atomics (`img.Interlocked*`) | (none extra — but the bound view's format must support STORAGE_IMAGE_ATOMIC) |
| float atomics | AtomicFloat32AddEXT + SPV_EXT_shader_atomic_float_add (add); AtomicFloat32MinMaxEXT + SPV_EXT_shader_atomic_float_min_max (min/max); exchange is core |
| `SubpassInput`, `SubpassInputMS` | InputAttachment |
| `TileImage`, `tile_depth`, `tile_stencil` | TileImageColorReadAccessEXT / DepthReadAccessEXT / StencilReadAccessEXT + SPV_EXT_shader_tile_image |
| `[wave_size(N)]` / `//--wave_size` | *(not in SPIR-V — carried in `.sks` reflection, applied via VK_EXT_subgroup_size_control at pipeline creation)* |
| `[early_depth_stencil]`, `SV_Depth*Equal`, `invariant` | (none — execution modes / decorations) |
| `SampleWeightedQCOM`, `BoxFilterQCOM`, `BlockMatchSAD/SSDQCOM` | TextureSampleWeightedQCOM / TextureBoxFilterQCOM / TextureBlockMatchQCOM + SPV_QCOM_image_processing (module → SPIR-V 1.4) |
| `BlockMatchWindow*` / `BlockMatchGather*` | + TextureBlockMatch2QCOM + SPV_QCOM_image_processing2 |
| `[tile_attachment]`, `tile_*_qcom()` builtins, `[tile_shading_rate_qcom]`, `[non_coherent_tile_reads_qcom]` | TileShadingQCOM + SPV_QCOM_tile_shading |
| `//--apron` | *(not in SPIR-V — renderer state: VkRenderPassTileShadingCreateInfoQCOM::tileApronSize)* |

Target environment: Vulkan 1.1 / SPIR-V 1.3, fixed — except QCOM image processing,
whose SPIR-V extension requires 1.4: those modules are emitted as SPIR-V 1.4 with the
full-interface entry point 1.4 mandates (runtimes need Vulkan 1.2+ or VK_KHR_spirv_1_4,
which every VK_QCOM_image_processing device has). The `.sks` container additionally
carries a 64-bit device-feature mask derived from the emitted SPIR-V, so runtimes can
answer "can this device run this shader?" before parsing the blob — see
docs/DECISIONS.md (SKS v9 additions). QCOM features map to bits 17–19.

---

*End of specification. Anything not documented here is not part of SVSL v1.*
