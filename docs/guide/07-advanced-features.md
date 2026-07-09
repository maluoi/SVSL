# Advanced Features

This chapter collects the modern-SPIR-V features that don't belong to a single earlier
chapter: specialization constants, the early-Z toolkit, and - the thing that ties the whole
language together - how capabilities and extensions are inferred rather than declared. The
subgroup, atomic, tile-image, subpass, MSAA, and storage-image features each have their home
in [Intrinsics](06-intrinsics.md) and [Resources](05-resources-and-buffers.md); this chapter
assumes you've met them.

---

## Specialization constants

A specialization constant is a value fixed at *pipeline creation* rather than compile time -
the runtime supplies it through Vulkan's specialization info. SVSL reflects each one **by
name** into the `.sks` container with its default, so the runtime sets it without caring
about numeric ids.

Declare one with `specialization const`. The bare form auto-assigns the constant id:

```c
specialization const uint32 TILE      = 16;
specialization const bool   HALF_RATE = false;
```

When an external system expects a *specific* id, pin it with `[specialization(N)]` - the
native spelling of `[[vk::constant_id(N)]]`:

```c
[specialization(7)] const float CUTOFF = 0.5;
```

Rules:

- **32-bit scalar or bool only.** A `float4` spec constant is rejected:
  `specialization constant 'X' must be a 32-bit scalar or bool`.
- **A constant default is required** - `specialization constant 'X' needs a constant
  default`.
- **Explicit ids must be unique** (`specialization id 7 already used`); auto-ids take the
  lowest free id that never collides with an explicit one.

Spec constants are usable wherever a constant expression is: **array sizes**, **compute
workgroup sizes**, and integer/bool arithmetic. Integer and boolean math over spec constants
stays *specializable* - it lowers to `OpSpecConstantOp`, so the runtime can still change the
inputs at pipeline creation:

```c
specialization const uint32 TILE = 16;

[numthreads(64, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint area   = TILE * TILE;                    // OpSpecConstantOp IMul - stays specializable
	uint stride = HALF_RATE ? TILE / 2u : TILE;   // Select over UDiv
	float4 tile[TILE];                            // spec constant as an array size
	...
}
```

> **One exception:** a spec constant used as a compute **workgroup size** is baked to its
> default value (emitted as a literal `LocalSize`), so it is *not* runtime-specializable -
> unlike an array size, which keeps the spec constant as its length. Use `[wave_size]` or a
> different mechanism if you need the launch dimensions to vary at pipeline creation.

---

## The early-Z toolkit

Three independent features that let you keep early depth testing while doing things that
normally force it off. All map to a single execution mode or decoration; there is no runtime
cost beyond the hardware behaviour they request.

### `[early_depth_stencil]`

Forces the depth/stencil test to run *before* the fragment shader (`EarlyFragmentTests`
execution mode). Use it when the shader has side effects (image or buffer writes) that must
not happen for occluded fragments:

```c
[early_depth_stencil]
float4 ps(psIn input) : SV_Target {
	...
}
```

### Conservative depth output

Writing `SV_Depth` normally disables early-Z (the hardware can't test a depth it hasn't
computed yet). The conservative variants promise the written depth only moves in one
direction, which lets the GPU keep early-Z on:

```c
struct psOut {
	float4 color : SV_Target;
	float  depth : SV_DepthGreaterEqual;   // promise: written depth >= rasterized depth
};

psOut ps(psIn input) {
	psOut o;
	o.color = shade(input);
	o.depth = input.pos.z + bump(input);   // only ever pushes depth away from the camera
	return o;
}
```

`SV_DepthGreaterEqual` emits `FragDepth` with the `DepthGreater` execution mode;
`SV_DepthLessEqual` emits `DepthLess`. Keep the promise - the hardware assumes it.

### `invariant`

Covered in [Stages](04-stages-and-interfaces.md#6-invariant): the `Invariant` decoration
forces bit-identical output across passes. Put it on `SV_Position` so a depth prepass and the
main pass compute exactly the same clip position and don't z-fight. It combines naturally
with the two features above.

These three are SVSL-dialect - glslang has no HLSL spelling for `[early_depth_stencil]`,
`invariant`, or the conservative-depth semantics.

---

## Precise arithmetic - `precise`

`invariant` pins a *stage output* bit-identical across pipelines; `precise` pins an
*intermediate value* bit-identical to the arithmetic you wrote. GPUs fuse `a*b + c` into a
single fused multiply-add by default - faster, but it rounds once instead of twice, so the
result differs in the last bit. Usually that is a win. Occasionally it is a bug: a Kahan
summation that depends on the rounding, an edge-equation that must match between two
triangles or they crack apart, a comparison that flips.

Mark the variable `precise` and the contraction is forbidden for the expression that
computes it - each contributing float op is emitted with `OpDecorate … NoContraction`:

```c
precise float edge = a*x + b*y + c;   // evaluated exactly as written, no fma fusion
```

It applies to the initializing expression. If a contribution is routed through a separate,
non-`precise` local, mark that one `precise` too - the guarantee does not cross the memory
round-trip on its own. `precise` is the HLSL spelling; it maps to SPIR-V `NoContraction`.

---

## Inline SPIR-V - `spirv_asm`

SVSL exposes a lot of modern SPIR-V directly, but it can't have a keyword for
every opcode in the spec. `spirv_asm` is the escape hatch: a block of raw SPIR-V
instructions, spliced into the function body, that can read your SVSL values and
produce one back. It is the "we own the SPIR-V" promise made literal - nothing in
the instruction set is out of reach, even before the language grows syntax for it.

It is an **expression**. The parenthesised type is the SVSL type of the value it
yields; the distinguished `%result` id carries that value out:

```c
// GLSL.std.450 Sin (ext instruction 13) on an SVSL value
float s = spirv_asm(float) {
    OpExtInst $$float %result glsl450 13 $angle;
};
```

Instructions are written in **binary operand order** - exactly the word sequence
the opcode takes, including the result-type and result-id operands. There is no
`%r = Op …` sugar; you write every operand explicitly, which keeps the assembler
fully general (it needs no per-opcode knowledge). Operands are:

| operand | meaning |
|---|---|
| `OpXxx` | the opcode mnemonic, starting each instruction (any name the SPIR-V spec defines) |
| `$$type` | an SVSL type → its SPIR-V *type* id (`$$float`, `$$float2`, `$$MyStruct`) |
| `$value` / `$(expr)` | an SVSL value → its SPIR-V *result* id; `$(…)` allows any expression |
| `%id` | a named local; first use allocates a fresh id, later uses reference it |
| `%result` | the local whose value becomes the block's result (required) |
| `42` | a literal 32-bit word - enum values and immediates are written numerically |
| `glsl450` | the `GLSL.std.450` ext-instruction import id (created on demand) |

An intermediate `%local` threads a value between instructions in the block:

```c
float2 v = spirv_asm(float2) {
    OpFNegate           $$float2 %neg    $uv;      // %neg = -uv
    OpVectorTimesScalar $$float2 %result %neg $s;  // result = %neg * s
};
```

A block may be a whole function body - it inlines like any other call:

```c
float fadd(float a, float b) {
    return spirv_asm(float) { OpFAdd $$float %result $a $b; };
}
```

Rules and current limits:

- The block **must define `%result`** - `spirv_asm block defines no %result id`
  otherwise. Its declared type is asserted, not checked against the opcode; the
  output is still run through `spirv-val`, so a malformed block fails loudly.
- Operands are **SSA values and types only** - you cannot name a global resource,
  pointer, or push a string/64-bit-literal operand yet. Pass data in as `$values`.
- **Enums are numeric.** There is no named-enumerant table yet, so masks, memory
  semantics, and ext-instruction numbers are written as integer literals (the
  `13` above is `GLSLstd450Sin`).
- A `spirv_asm` block is treated as **opaque and side-effecting**: the optimizer
  never merges, reorders, or deletes it, even if its result goes unused.

`spirv_asm` is SVSL-dialect - glslang/skshaderc has no equivalent, so shaders
that use it compile only through SVSL.

---

## Required subgroup size - `[wave_size(N)]`

Pin the hardware subgroup (wave) width for a compute entry point so subgroup ops partition
identically every run. The width is recorded in `.sks` reflection and applied as the required
subgroup size at pipeline creation (Vulkan's `VK_EXT_subgroup_size_control`) - it is not a
SPIR-V execution mode:

```c
[wave_size(32)]
[compute(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) { ... }
```

The metadata form `//--wave_size = 32` (see [Metadata](08-preprocessor-and-metadata.md)) is
the alias and additionally lands in `.sks` reflection. `N` must be a **power of two in
[4, 128]** - otherwise `wave_size must be a power of two in [4,128]`.

---

## Capability inference

You **never** write a capability or extension declaration. SVSL scans the types and
operations your shader actually uses and emits exactly the SPIR-V capabilities and extensions
they need - principle 4 of the language. This table is a reference for *what triggers what*,
not a set of knobs you turn.

| What you use | Capability / extension inferred |
|--------------|----------------------------------|
| `float16` | `Float16` (+ 16-bit storage caps in buffers) |
| `half` | *(none - `RelaxedPrecision` decoration only)* |
| `int8`/`uint8` | `Int8` (+ 8-bit storage caps, `SPV_KHR_8bit_storage`) |
| `int16`/`uint16` | `Int16` (+ 16-bit storage caps) |
| `int64` | `Int64` |
| `float64` | `Float64` |
| `pack1`/`pack8` layout breaking core relaxed rules | `VK_EXT_scalar_block_layout` (a Vulkan device feature - no SPIR-V capability; recorded as `.sks` feature-mask bit 16) |
| `SV_ViewID` | `MultiView` (core in SPIR-V 1.3 - no extension) |
| `demote` / `is_helper_invocation` | `DemoteToHelperInvocation` |
| subgroup ops | `GroupNonUniform` + `Vote`/`Ballot`/`Arithmetic`/`Shuffle`/`Clustered`/`Quad` as used |
| storage images | `StorageImageExtendedFormats` when the format needs it |
| `SubpassInput` | `InputAttachment` |
| `TileImage`, `tile_depth`/`tile_stencil` | `SPV_EXT_shader_tile_image` |
| float atomics | `SPV_EXT_shader_atomic_float_add` (add) / `SPV_EXT_shader_atomic_float_min_max` (min/max); exchange is core |
| `[wave_size(N)]` / `//--wave_size` | *(not a SPIR-V feature - carried in `.sks` reflection, applied via `VK_EXT_subgroup_size_control` at pipeline creation)* |
| `[early_depth_stencil]`, `SV_Depth*Equal`, `invariant` | *(none - execution modes / decorations)* |

### Target environment

The target is **Vulkan 1.1 / SPIR-V 1.3**, fixed - the same as the reference compiler
skshaderc. SVSL always emits SPIR-V 1.3; there is no target-version knob. Every feature in
this guide is expressible at 1.3, so [capability inference](#capability-inference) is the
whole story: you never pick a version, declare a capability, or request an extension, and
nothing silently downgrades.

The half/float16 split from [Types §3](02-types-and-values.md#3-half-vs-float16--the-one-thing-that-is-not-hlsl)
is the one place a type choice interacts with the target directly: `half` always works
(it's decorated float32), while `float16` will fail to compile against a target that can't
express true 16-bit floats.

---

Next: [Preprocessor and Metadata](08-preprocessor-and-metadata.md).
