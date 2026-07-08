# The SVSL Language Guide

SVSL (SPIR-V Shading Language) is a shader language that compiles an HLSL-compatible
source dialect directly to SPIR-V. It keeps the C-like HLSL you already know, and adds
first-class syntax for the modern SPIR-V features that HLSL can only reach through
`[[vk::*]]` escape-hatch attributes.

This guide is the programmer's manual for the **language**. It documents every construct
SVSL accepts, how to use it, and how it relates to HLSL. (The `svslc` command-line
compiler and the C API are documented separately.)

> **Two documents, two audiences.** [`../LANGUAGE_SPEC.md`](../LANGUAGE_SPEC.md) is the
> terse normative reference - the "what is legal" list. This guide is the explanatory
> companion - the "how and why," with worked examples for each feature. Where they ever
> disagree, the compiler is the source of truth.

---

## The one-paragraph mental model

You write shaders that look like HLSL. Vertex, fragment, and compute stages are ordinary
functions; resources are global declarations; the pieces you pass between stages are
`struct`s with semantics. SVSL compiles this straight to SPIR-V with no glslang or DXC in
the loop. Everywhere HLSL forces a `[[vk::*]]` attribute to express a Vulkan concept, SVSL
gives that concept a real keyword - but the HLSL spelling still compiles, with a
suppressible *porting* hint pointing you at the native form. Nothing you know is thrown
away; there is just more you can now say directly.

```c
//--name = example/unlit

float4 color = {1, 1, 1, 1}; // becomes a material default in reflection

//--diffuse = white
Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0); // texture + sampler fuse into one binding

struct vsIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; float4 col : COLOR0; };
struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float4 col : COLOR0; };

psIn vs(vsIn input) { // matched by name - "vs" is the default VS entry
	psIn o;
	o.pos = input.pos;
	o.uv  = input.uv;
	o.col = input.col * color;
	return o;
}

float4 ps(psIn input) : SV_TARGET {   // "ps" is the default FS entry
	return diffuse.Sample(diffuse_s, input.uv) * input.col;
}
```

---

## Five surprises if you come from HLSL/DXC

Most HLSL compiles unchanged, but a handful of behaviours differ from what a DXC user
expects. These are the ones worth knowing up front; each is expanded later.

1. **`half` ≠ `float16`.** `half` is portable relaxed-precision (compiles to `float32` +
   `RelaxedPrecision`, always 4 bytes); `float16` is true 16-bit and needs capability
   support. ([Types §3](02-types-and-values.md#3-half-vs-float16--the-one-thing-that-is-not-hlsl))
2. **Matrices: `m[i]` is a *row*, and `a * b` is component-wise.** Use `mul()` for matrix and
   matrix/vector products. ([Types §4](02-types-and-values.md#4-vectors-and-matrices))
3. **A misplaced `SV_*` semantic degrades silently** to a numbered varying - no error. A
   typo'd or wrong-stage system value just becomes an ordinary interpolant.
   ([Stages §3](04-stages-and-interfaces.md#3-semantics-system-values-vs-numbered-io))
4. **A texture and same-slot sampler fuse into one binding.** There is no standalone sampler
   in the output; pairing is by slot number, not the `_s` name.
   ([Resources §1](05-resources-and-buffers.md#1-sampled-textures-and-samplers))
5. **Resources can't be locals, and recursion is a compile error.** Every call is inlined, so
   textures/samplers pass as parameters (and resolve away) but a call cycle has no base case.
   ([Expressions §3](03-expressions-and-functions.md#3-functions))

---

## How to read this guide

The chapters are ordered so you can read straight through, but each stands alone as a
reference. If you already write HLSL, skim chapters 2–3 and spend your time in 4–7 and 9.

| # | Chapter | What it covers |
|---|---------|----------------|
| 1 | [The Tour](01-tour.md) | A hands-on walk from a first triangle shader to a compute shader. Start here. |
| 2 | [Types and Values](02-types-and-values.md) | Scalars, `half` vs `float16`, vectors, matrices, arrays, structs, literals, constructors, casts, swizzles |
| 3 | [Expressions and Functions](03-expressions-and-functions.md) | Operators, control flow, functions, resource-typed parameters, inlining, `discard` / `demote` |
| 4 | [Stages and Interfaces](04-stages-and-interfaces.md) | Entry points, stage attributes, system-value semantics, locations, interpolation, `invariant`, multiview |
| 5 | [Resources and Buffers](05-resources-and-buffers.md) | Textures, samplers, storage images, subpass/MS/tile inputs, buffer blocks, memory layout, bindings |
| 6 | [Intrinsic Functions](06-intrinsics.md) | The full intrinsic library: math, derivatives, subgroup ops, atomics, barriers, queries |
| 7 | [Advanced Features](07-advanced-features.md) | Specialization constants, the early-Z toolkit, `precise`, inline SPIR-V (`spirv_asm`), required subgroup size, capability inference |
| 8 | [Preprocessor and Metadata](08-preprocessor-and-metadata.md) | The preprocessor, `include`, and `//--` material metadata |
| 9 | [HLSL Compatibility](09-hlsl-compatibility.md) | The migration reference: dialect model, alias tables, `[[vk::*]]` escapes, porting hints, glslang divergences |
| A | [Quick Reference & Porting Cheat-Sheet](10-cheatsheet.md) | One-page condensation of every native↔HLSL mapping, layout default, and binding rule |

---

## The design principles behind the language

Six ideas explain nearly every decision in the chapters that follow. Keep them in mind and
the language stops having surprises.

1. **Direct SPIR-V mapping.** Every construct corresponds to something concrete in SPIR-V.
   When you reach for a feature, there is a specific opcode, decoration, or capability it
   produces - nothing is a leaky abstraction over the target.

2. **HLSL familiarity.** The C-like subset of HLSL compiles as-is. Legacy spellings compile
   *silently* - they never error, and the only ones that even warn are `[[vk::*]]` escape
   attributes, which get a suppressible `porting` hint (principle 3).

3. **`vk::` escapes become language.** Every `[[vk::*]]` attribute marks a concept HLSL
   lacks a word for. SVSL gives each one a real keyword. The escape still parses and works;
   the keyword is the recommended form. This is the language's reason to exist.

4. **Capability inference.** You never declare SPIR-V capabilities or extensions. They are
   derived from the types and operations you actually use - use `float16`, get `Float16`;
   sample a subgroup op, get the matching `GroupNonUniform*` capability.

5. **Names over numbers.** Reflection is name-based, so explicit numbers - binding slots,
   specialization ids, interpolation locations - are optional interop tools, not
   requirements. Omit them and SVSL assigns deterministically, never colliding with the
   explicit ones you *did* write.

6. **Honest surface.** This guide documents only what the compiler implements and the test
   corpus exercises. Features that are planned but absent live in `docs/BACKLOG.md`, never here.

---

## What is *not* in the language (yet)

So you don't go looking: physical storage-buffer pointers, bindless / `nonuniform`
descriptor indexing, mesh and task stages, ray tracing, cooperative matrices, and
fragment shader interlock are **deferred**. They are not part of the language today. If you
need them, they are tracked in the project's `docs/BACKLOG.md`.
