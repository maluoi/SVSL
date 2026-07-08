# Stages and Interfaces

A shader stage is a function. This chapter covers how a function *becomes* an entry point,
how its inputs and outputs map to the pipeline through semantics and locations, and the
interpolation and multiview controls that ride on stage IO.

---

## 1. Entry points

One compiled SPIR-V module is produced **per entry point**. A function is an entry point in
one of two ways:

**By stage attribute** - the native, explicit form:

```c
[vertex]           VSOut  my_vs(VSIn i) { ... }
[fragment]         float4 my_ps(VSOut i) : SV_Target { ... }   // [pixel] is an alias
[compute(8, 8, 1)] void   my_cs(uint3 id : SV_DispatchThreadID) { ... }
```

**By name** - the StereoKit convention. Functions named `vs`, `ps`, and `cs` are treated as
the vertex, fragment, and compute entry points with no attribute needed:

```c
VSOut  vs(VSIn i) { ... }
float4 ps(VSOut i) : SV_Target { ... }
```

The default names are configurable per-compilation (the compiler's `-vs`/`-ps`/`-cs`
options). You can mix the two mechanisms: a function named `cs` still needs `[numthreads]`
or `[compute]` to carry its workgroup size, but not to be *recognized* as compute.

Only **one entry point per stage** is allowed in a module - a second vertex entry is
`duplicate entry point for stage`.

### Compute workgroup size

```c
[numthreads(64, 1, 1)] void cs(...) { ... }   // HLSL spelling
[compute(64, 1, 1)]    void cs(...) { ... }   // native spelling - identical
```

`[numthreads(x,y,z)]` is the HLSL alias for `[compute(x,y,z)]`'s sizes. Sizes must be
positive constants; they may reference [specialization
constants](07-advanced-features.md#specialization-constants) - but note a spec constant in a
workgroup size is **baked to its default** (emitted as a literal `LocalSize`), so, unlike an
array size, it does not stay runtime-specializable.

---

## 2. Stage IO through structs

Vertex inputs, interstage varyings, and fragment outputs are `struct` members (or single
parameters/returns) tagged with **semantics**. The vertex stage's input struct describes the
vertex format; its output struct is the fragment stage's input:

```c
struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos : SV_Position;   // FragCoord when read as a fragment input
	float2 uv  : TEXCOORD0;
	float4 col : COLOR0;
};

psIn   vs(vsIn i)  { ... }
float4 ps(psIn i) : SV_Target { ... }
```

A stage can also take loose parameters instead of a struct - `vs(uint id : SV_VertexID)` or
`ps(psIn i, bool front : SV_IsFrontFace)` - and a fragment stage puts its output semantic on
the return type (`: SV_Target`) or on the members of a returned struct.

---

## 3. Semantics: system values vs. numbered IO

Semantics come in two kinds, and understanding the split removes most confusion.

**System-value semantics** (`SV_*`) map to SPIR-V built-ins - but *only in the stage and
direction where that built-in is valid*. `SV_VertexID` is a built-in only as a vertex
**input**; `SV_DispatchThreadID` only as a compute input; and so on.

**Numbered/legacy semantics** (`NORMAL0`, `TEXCOORD2`, `COLOR1`, …) are *not* built-ins.
They exist to (a) describe the vertex attribute format for reflection and (b) be
human-readable labels. They do **not** drive interstage wiring - that is done by `location`
(§4).

> **Important, and easy to trip on:** a system-value semantic used in the *wrong*
> stage/direction is **not an error**. It silently degrades to a plain numbered varying. For
> example, carrying `SV_InstanceID` from the vertex stage into the fragment stage makes it an
> ordinary interpolated value, not the `InstanceIndex` built-in. This matches how the corpus
> threads IDs between stages, but it means a typo'd or misplaced `SV_` name fails quietly
> rather than loudly. Double-check the table below.

### System-value reference

Recognition is case-insensitive (`SV_Position` and `SV_POSITION` are the same), and a
trailing index is parsed where it applies (`SV_Target2`, `TEXCOORD3`).

| Semantic | SPIR-V built-in | Valid stage / direction |
|----------|-----------------|-------------------------|
| `SV_Position` | `Position` (VS out) / `FragCoord` (FS in) | VS out, FS in |
| `SV_Target`, `SV_Target0`..`N` | fragment output at location *N* | FS out |
| `SV_Depth` | `FragDepth` | FS out |
| `SV_DepthGreaterEqual` | `FragDepth` + `DepthGreater` exec mode | FS out |
| `SV_DepthLessEqual` | `FragDepth` + `DepthLess` exec mode | FS out |
| `SV_VertexID` | `VertexIndex` | VS in |
| `SV_InstanceID` | `InstanceIndex` | VS in |
| `SV_ViewID` | `ViewIndex` (+ `MultiView` capability) | VS in, FS in |
| `SV_IsFrontFace` | `FrontFacing` | FS in |
| `SV_SampleIndex` | `SampleId` | FS in |
| `SV_Coverage` | `SampleMask` | FS in, FS out |
| `SV_PrimitiveID` | `PrimitiveId` | FS in |
| `SV_DispatchThreadID` | `GlobalInvocationId` | CS in |
| `SV_GroupThreadID` | `LocalInvocationId` | CS in |
| `SV_GroupID` | `WorkgroupId` | CS in |
| `SV_GroupIndex` | `LocalInvocationIndex` | CS in |

Anything not in this table (`NORMAL0`, `TEXCOORD*`, `COLOR*`, `TANGENT`, `BINORMAL`, `PSIZE`,
`BLENDWEIGHT`, `BLENDINDICES`, and any `SV_*` used off-stage) is numbered IO.

`SV_Position` is special as a vertex *input*: there it is **not** the `Position` built-in but
a normal numbered vertex attribute (the corpus writes `float4 pos : SV_Position` in the
vertex input struct to mean "attribute 0"). It only becomes the `Position` built-in on the
vertex *output*.

---

## 4. Locations

Interstage members are assigned SPIR-V **`location`s by declaration order**, skipping any
member that resolved to a system-value built-in. You rarely set them by hand - the vertex
output and fragment input agree because they list members in the same order.

```c
struct VSOut {
	float4      pos   : SV_Position;  // built-in - consumes no location
	float2      uv;                   // location 0
	flat uint32 id;                   // location 1
	float3      world;                // location 2
};
```

When you need a specific slot (matching a fixed vertex layout, or a `[[vk::location]]` from
ported code), set it explicitly with `[location(N)]`, the native spelling of
`[[vk::location(N)]]`:

```c
struct VSIn {
	[location(0)] float3 pos;
	[location(3)] float2 uv;
};
```

`[location(N)]` needs a non-negative constant.

---

## 5. Interpolation modifiers

Fragment inputs (equivalently, vertex outputs) can carry an interpolation qualifier:

| Qualifier | Meaning |
|-----------|---------|
| `flat` | no interpolation - the provoking vertex's value (`nointerpolation` is the HLSL alias) |
| `noperspective` | linear screen-space interpolation, no perspective correction |
| `centroid` | sampled at the centroid of the covered area (MSAA) |
| `sample` | per-sample interpolation (MSAA) - invokes the fragment shader per sample |

```c
struct PSIn {
	              float4 pos : SV_Position;
	flat          uint32 matId; // integer varyings must be flat
	noperspective float2 screenUV;
	centroid      float3 world;
};
```

`nointerpolation` compiles the same as `flat`; unlike the `[[vk::*]]` escapes it does **not**
emit a porting hint - legacy interpolation spellings are accepted silently.

---

## 6. `invariant`

The `invariant` qualifier (SPIR-V `Invariant` decoration) forces an output to be computed
bit-identically across entry points that compute it the same way. Put it on `SV_Position`
when several passes (a depth prepass, a shadow pass, the main pass) recompute the same
clip-space position and must agree to the last bit, or they will z-fight:

```c
struct psIn {
	invariant float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};
```

`invariant` is SVSL-dialect - glslang has no HLSL spelling for it. It pairs naturally with
the conservative-depth outputs and `[early_depth_stencil]` covered in [Advanced
Features](07-advanced-features.md#the-early-z-toolkit).

---

## 7. Multiview

Stereo rendering (both eyes in one pass) uses `SV_ViewID`, which maps to the `ViewIndex`
built-in and pulls in the `MultiView` capability automatically. StereoKit threads it in as a
second vertex parameter and indexes its per-view matrix arrays with it:

```c
struct sk_ids_t {
	uint inst : SV_InstanceID;
	uint view : SV_ViewID;
};

psIn vs(vsIn input, sk_ids_t ids) {
	float4 world = mul(float4(input.pos.xyz, 1), sk_inst[ids.inst].world);
	o.pos = mul(world, sk_viewproj[ids.view]);   // ids.view selects the eye
	...
}
```

Using `SV_ViewID` anywhere is all it takes - the `MultiView` capability is inferred; you
never request it. (`MultiView` is core in SPIR-V 1.3, so no extension is needed.)

---

Next: [Resources and Buffers](05-resources-and-buffers.md) - the textures, samplers, images,
and buffer blocks these stages read from.
