# The Tour

This chapter walks from nothing to a working compute shader, introducing the language's
shape as we go. Every example here is real SVSL that compiles. Later chapters are the
reference; this one is the map.

If you write HLSL, almost none of this will surprise you - which is the point. Read it for
the few places SVSL differs, and for where it offers something HLSL can't say. Where a
construct has a **native SVSL spelling**, this chapter shows it alongside the HLSL one so you
can see what the language adds - both compile to the same SPIR-V, and (with the sole
exception of `[[vk::*]]` attributes) the HLSL spelling compiles silently, no warning.

---

## 1. A fragment shader, minimal

The smallest useful shader is a fragment stage that returns a constant colour:

```c
float4 ps() : SV_Target {
	return float4(1, 0, 0, 1);
}
```

A function becomes a shader **entry point** in one of two ways: by a stage attribute
(`[fragment]` here would be explicit), or by *name*. `ps` is the default fragment-stage
name, so this function is a fragment shader with no attribute needed. `vs` and `cs` are the
default vertex and compute names. This name-based convention is how StereoKit's corpus is
written; the [Stages](04-stages-and-interfaces.md) chapter covers both mechanisms.

`SV_Target` is the fragment output. SVSL matches HLSL's system-value semantics, and it
accepts the case variants the wild uses - `SV_Target`, `SV_TARGET`, `SV_Position`,
`SV_POSITION` are all fine.

---

## 2. Adding a vertex stage and passing data between stages

Stages exchange data through a `struct` whose members carry semantics. The vertex stage
returns it; the fragment stage takes it:

```c
struct vsIn {
	float4 pos : SV_Position;
	float2 uv  : TEXCOORD0;
	float4 col : COLOR0;
};
struct psIn {
	float4 pos : SV_Position;   // SV_Position out of VS, FragCoord into PS
	float2 uv  : TEXCOORD0;
	float4 col : COLOR0;
};

psIn vs(vsIn input) {
	psIn o;
	o.pos = input.pos;
	o.uv  = input.uv;
	o.col = input.col;
	return o;
}

float4 ps(psIn input) : SV_Target {
	return input.col;
}
```

The `TEXCOORD0` / `COLOR0` semantics on the interstage members are **legacy HLSL
semantics**. SVSL accepts them, but they do not drive the SPIR-V interface - interstage
members are assigned `location`s by declaration order (skipping system values like
`SV_Position`). The semantics survive into vertex-input reflection so tooling still knows
which attribute is which.

**What SVSL adds:** because ordering is what actually matters between stages, you can drop the
interstage semantics entirely, and reach for `[location(N)]` - the native spelling of
`[[vk::location(N)]]` - only when you need an exact slot:

```c
struct psIn {
	float4 pos : SV_Position;   // still a system value
	float2 uv;                  // location 0 - no semantic needed
	float4 col;                 // location 1
};
```

The vertex *input* struct usually keeps its semantics, since those are what feed vertex-input
reflection. See [Stages and Interfaces](04-stages-and-interfaces.md).

---

## 3. Uniforms and a texture

Real shaders read constants and sample textures. A constant buffer is a block; a texture
and its sampler are two globals that SVSL fuses into a single binding:

```c
cbuffer Params : register(b0) {
	float4x4 mvp;
	float4   tint;
};

Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

psIn vs(vsIn input) {
	psIn o;
	o.pos = mul(mvp, input.pos);   // column-major matrix * vector
	o.uv  = input.uv;
	o.col = input.col;
	return o;
}

float4 ps(psIn input) : SV_Target {
	return diffuse.Sample(diffuse_s, input.uv) * input.col * tint;
}
```

Three things worth noting, each expanded later:

- **Members are used by name.** Inside the shader you write `mvp` and `tint` directly; the
  block name `Params` is a grouping for binding and reflection, not a namespace you index
  through.
- **The texture/sampler pair fuses.** `diffuse` at `t0` and `diffuse_s` at `s0` become one
  combined image-sampler in the SPIR-V, bound at one slot, named after the *texture*. The
  `_s` suffix is convention; pairing is by matching slot number. See
  [Resources](05-resources-and-buffers.md).
- **`register()` is optional.** Runtimes bind by name through reflection, so you can leave the
  slots off and let SVSL assign them deterministically.

**The same shader, in native SVSL:** `cbuffer` becomes `uniform`, `SamplerState` becomes
`Sampler`, and the redundant registers come off:

```c
uniform Params {
	float4x4 mvp;
	float4   tint;
};

Texture2D diffuse;        // auto-assigned t-slot
Sampler   diffuse_s;      // still fuses with diffuse

// vs / ps bodies unchanged
```

The two versions compile to identical SPIR-V. The HLSL spellings (`cbuffer`, `SamplerState`)
are accepted **silently** - no warning - so porting is quiet; the native forms are simply the
recommended way to write new shaders. This "two spellings, one meaning" pattern runs through
the whole language; the full mapping is in [HLSL
Compatibility](09-hlsl-compatibility.md).

---

## 4. Material defaults from bare globals

A bare global with an initializer is not an error and not a private variable - it becomes a
member of an implicit `$Global` uniform buffer, and its initializer becomes a **material
default** baked into the reflection output:

```c
//--name = example/tinted
//--tint:color = 1, 1, 1, 1

float4 tint = {1, 1, 1, 1};   // default value travels into the .sks container
float  glow = 0;
```

The `//--` comments are **material metadata** - a name for the shader, a tag marking `tint`
as a colour for the editor UI, and so on. This is how StereoKit surfaces material
parameters. Full syntax is in [Preprocessor and Metadata](08-preprocessor-and-metadata.md).

---

## 5. Helper functions - including ones that take textures

Factor code into functions freely. Forward references need no prototypes (the parser makes
two passes). The one rule that differs from CPU C: **you may pass textures, samplers, and
images as parameters.**

```c
float4 sample_tinted(Texture2D t, Sampler s, float2 uv, float4 k) {
	return t.Sample(s, uv) * k;
}

float4 ps(psIn input) : SV_Target {
	return sample_tinted(diffuse, diffuse_s, input.uv, tint);
}
```

SPIR-V cannot pass opaque resource types as values, so this looks impossible - but SVSL
**fully inlines every call** during compilation. By the time SPIR-V is emitted, the
parameter has resolved to the concrete global. The same inlining is why **recursion is a
compile error**: there is no call left to recurse. This is the single most important
structural feature of the compiler; [Expressions and
Functions](03-expressions-and-functions.md) covers it.

---

## 6. A compute shader

Compute stages are functions too. The workgroup size is an attribute, and the invocation's
position comes in through system-value semantics on the parameters:

```c
RWStructuredBuffer<float4> output : register(u0);

cbuffer Params : register(b0) {
	float2 resolution;
	float  time;
};

[numthreads(8, 8, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	float2 uv = float2(tid.xy) / resolution;
	output[tid.y * uint(resolution.x) + tid.x] = float4(uv, sin(time), 1);
}
```

**The same compute shader, in native SVSL:** `[numthreads]` becomes `[compute]`, `cbuffer`
becomes `uniform`, and the `RWStructuredBuffer` becomes a `storagebuffer` block. A
`storagebuffer` with no `readonly` is read-write and auto-binds to a `u` slot, so it is the
direct native equivalent:

```c
storagebuffer OutputBuffer {
	float4 output[];             // runtime-sized: length comes from the bound buffer
};

uniform Params {
	float2 resolution;
	float  time;
};

[compute(8, 8, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	float2 uv = float2(tid.xy) / resolution;
	output[tid.y * uint(resolution.x) + tid.x] = float4(uv, sin(time), 1);
}
```

Note the block form indexes the **member** (`output`), not the block (`OutputBuffer`) - the
block name is just the binding group and never appears in the body. That's the one syntactic
difference from `RWStructuredBuffer`, where the buffer object is the thing you index. Both
reflect identically: a read-write buffer at `u0` with a 16-byte element stride. (If you prefer
the object form, `RWStructuredBuffer<float4> output` is a native-accepted resource type too -
the block form is just the more explicit spelling.)

A function named `cs` is a compute entry by convention, so - like `vs`/`ps` - the attribute
only needs to carry the workgroup size, not re-declare the stage.

Compute is where the language's modern half comes alive, and much of it has **no HLSL
spelling at all**: `workgroup` shared memory (`groupshared` is the alias), atomics
(`atomic_add` / `InterlockedAdd`), barriers (`workgroup_barrier()`), subgroup / wave
operations, and `[wave_size(N)]` to pin the wave width. Those live in
[Intrinsics](06-intrinsics.md).

---

## 7. Where to go from here

You've now seen the whole shape of the language. The rest of the guide fills it in:

- The **modern SPIR-V features** - specialization constants, storage images with explicit
  formats, subpass and tile-image reads, the early-Z toolkit, MSAA resolve, multiview - are
  in [Resources](05-resources-and-buffers.md) and [Advanced
  Features](07-advanced-features.md).
- If you are **porting HLSL**, go straight to [HLSL
  Compatibility](09-hlsl-compatibility.md): it lists every legacy form, its native
  replacement, and the handful of places SVSL deliberately diverges from glslang.
