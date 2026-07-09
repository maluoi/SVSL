# Resources and Buffers

Everything a shader reads from the outside - textures, samplers, images, and the various
buffer kinds - is declared as a global. This chapter covers each resource type and its
access methods, the three buffer block flavours and their memory layout, and how bindings
are assigned.

---

## 1. Sampled textures and samplers

Sampled textures are read through a companion `Sampler`. The element type is a template
argument and defaults to `float4`:

```c
Texture1D<float4>        tex1D;
Texture2D                diffuse; // <float4> implied
Texture3D<float4>        volume;
TextureCube<float4>      envmap;
Texture2DArray<float4>   layers;
TextureCubeArray<float4> probes;
Texture1DArray<float2>   ramp;

Sampler           diffuse_s; // SamplerState is the HLSL alias
SamplerComparison shadow_s;  // SamplerComparisonState is the HLSL alias
```

`Sampler`/`SamplerComparison` are the native spellings; `SamplerState` /
`SamplerComparisonState` are the HLSL aliases (accepted, no diagnostic).

### Texture / sampler pairing

A texture and a sampler at the **same slot number** fuse into one combined image-sampler in
the SPIR-V - StereoKit always treats them as a unit:

```c
Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);        // pairs with diffuse (both slot 0)
```

The result is a single `OpTypeSampledImage` variable named after the *texture* (`diffuse`).
The `_s` name suffix is convention only - pairing is by matching slot number, not by name.
The sampler object does not appear in the SPIR-V at all.

### Sampling and fetching methods

```c
float4 c = tex.Sample            (smp, uv);
float4 c = tex.SampleLevel       (smp, uv, lod);
float4 c = tex.SampleBias        (smp, uv, bias);
float4 c = tex.SampleGrad        (smp, uv, ddx, ddy);
float  s = tex.SampleCmp         (cmpSmp, uv, ref); // uses SamplerComparison
float  s = tex.SampleCmpLevelZero(cmpSmp, uv, ref);
float4 g = tex.Gather            (smp, uv);         // 4-texel gather
float4 g = tex.GatherRed/Green/Blue/Alpha(smp, uv); // single-channel gather
float4 g = tex.GatherCmp         (cmpSmp, uv, ref);
float4 t = tex.Load              (int3(x, y, mip)); // unfiltered fetch, no sampler
```

### Queries: dimensions and LOD

`GetDimensions` comes in an SVSL **value-returning** form and the HLSL **out-parameter**
form; both are accepted:

```c
uint2 size   = diffuse.GetDimensions(); // native: returns the size
uint3 vsize  = volume .GetDimensions(); // 3D → uint3
uint  len    = tex1D  .GetDimensions(); // 1D → scalar

uint w, h;           diffuse.GetDimensions(w, h);              // HLSL out-param form
uint mw, mh, levels; diffuse.GetDimensions(1, mw, mh, levels); // sizes at mip 1 (+ level count)
uint aw, ah, layers; layers .GetDimensions(aw, ah, layers);    // array → +layer count
```

Out-parameters may be `uint` or `float` (floats convert). LOD queries:

```c
float lod  = tex.CalculateLevelOfDetail         (smp, uv);
float lodu = tex.CalculateLevelOfDetailUnclamped(smp, uv);
```

---

## 2. Storage images

Storage images (`RWTexture*` in HLSL) are read and written directly, without a sampler
(SPIR-V `OpTypeImage`, `Sampled=2`). The native spelling is `Image*<T>`, and the second
template argument is the **native form of `[[vk::image_format]]`** - the image format,
lowercase:

```c
Image2D     <float4>        img;  // format inferred from the element type
Image2D     <float4, rgba8> img8; // explicit format
Image3D     <float4, r32f>  density;
ImageCube   <float4>        cubeStore;
Image2DArray<uint32, r32u>  counters;
```

`Image1D/2D/3D/Cube` plus the `*Array` variants exist. Valid format tokens (from the format
table): `rgba32f rgba16f rg32f rg16f r32f r16f rgba8 rgba8_snorm r11g11b10f rgba16 rgb10a2
rg16 rg8 r16 r8 rgba32i rgba16i rgba8i r32i rgba32u rgba16u rgba8u r32u`. Formats outside
the core set infer `StorageImageExtendedFormats` automatically.

The HLSL spelling is `RWTexture2D<T>` plus a `[[vk::image_format("rgba8")]]` attribute:

```c
[[vk::image_format("rgba8")]]
RWTexture2D<float4> out_tex : register(u3); // == Image2D<float4, rgba8>
```

### Access

Storage images use subscript read/write - the idiomatic form throughout the corpus - or the
explicit `.Load` / `.Store` methods:

```c
float4 v = img[coord];              // load
img[coord] = color;                 // store
float4 w = img.Load(coord);         // explicit load
img.Store(coord, color);            // explicit store
```

Integer-format storage images also support atomics as methods
(`img.InterlockedAdd(coord, v)`, `Min/Max/And/Or/Xor`, with an optional out-parameter for the
prior value). See [Intrinsics → Atomics](06-intrinsics.md#8-atomics).

---

## 3. Multisampled, subpass, and tile-image inputs

These three read attachments rather than sampling textures - the machinery of modern
tile-based and multi-pass rendering.

**Multisampled textures** carry a sample count as the second template argument and are read
per-sample with `Load(coord, sampleIndex)` - they *cannot* be `Sample`d:

```c
Texture2DMS<float4, 4> msaa_color : register(t0);

float4 resolve(int2 p) {
	return (msaa_color.Load(p, 0) + msaa_color.Load(p, 1)
	      + msaa_color.Load(p, 2) + msaa_color.Load(p, 3)) * 0.25;
}
```

Attempting to `Sample` one is an error: `multisampled textures cannot be sampled; use
Load(coord, sample)`.

**Subpass inputs** (input attachments) read the previous render-pass attachment at the
current fragment position - free on a tiler, fragment stage only. The attachment index is the
**second template argument** - the native spelling of `[[vk::input_attachment_index]]`, just
like the format on a storage image. Omit it to auto-assign (defaults to attachment 0). Read
with `.SubpassLoad()` or the interchangeable `.Load()`:

```c
SubpassInput<float4, 0> color; // native: attachment index 0
SubpassInput<float4>    color; // native: auto-assigned index

float4 ps(psIn input) : SV_Target {
	float4 c = color.SubpassLoad(); // reads at this fragment's position
	return float4(c.rgb * 0.75, c.a);
}
```

The HLSL escape spells the index as an attribute instead - it compiles to exactly the same
thing, with a `porting` hint pointing back at the template-argument form:

```c
[[vk::input_attachment_index(0)]] SubpassInput<float4> color;   // == SubpassInput<float4, 0>
```

The multisampled variant `SubpassInputMS<T[, N]>` reads a specific sample:
`color.SubpassLoad(sampleIndex)`.

**Tile images** (`TileImage<T, N>`) read colour attachment *N*, plus the current fragment's
depth and stencil, entirely within tile memory - the tiler never leaves the tile. This is
native SVSL with no HLSL spelling (SPV_EXT_shader_tile_image):

```c
TileImage<float4, 0> tile_color;
TileImage<float4, 1> tile_normal;

float4 ps(psIn input) : SV_Target {
	float4 color  = tile_color.Load();    // no coordinate - reads this fragment's texel
	float4 normal = tile_normal.Load();
	float  depth  = tile_depth();         // free-function intrinsics, not methods
	uint   sten   = tile_stencil();
	...
}
```

`TileImage.Load()` takes no arguments. `tile_depth()` returns `float`, `tile_stencil()`
returns `uint`; both are free-function intrinsics. Tile images are tile memory, not
descriptors - they take **no** `register` or binding.

---

## 4. Buffer blocks

Buffers are declared as **blocks** - a name, an optional `register`, and a member list.
There are three storage flavours. Members are used directly by name inside the shader; the
block name is a grouping for binding and reflection, not a namespace you index through.

### `uniform` - read-only constant buffer

```c
uniform SceneData : register(b0) {
	float4x4 viewProj;
	float3   cameraPos;
	float    time;
};

float4 ps(...) : SV_Target {
	return shade(viewProj, cameraPos, time);   // members referenced directly
}
```

`cbuffer N : register(b#) { }` is the HLSL alias of `uniform` (accepted, no diagnostic).
Uniform blocks are read-only - writing a member is an error - and must use `pack16` layout
(§5).

### `storagebuffer` - read/write, runtime-sized

```c
readonly storagebuffer InputData : register(t0) {
	Vertex vertices[];              // runtime-sized: last member only
};
writeonly storagebuffer OutputData : register(u0) {
	float4 results[16];
};
```

`readonly` / `writeonly` (and `coherent`, `volatile`) are access qualifiers, written
before the storage keyword like every other declaration modifier. A runtime-sized
array (`[]`) is legal only as the **last** member of a `storagebuffer`; its length comes from
the bound buffer. Storage buffers default to **std430** layout (§5).

The HLSL-templated forms `StructuredBuffer<T>` (readonly), `RWStructuredBuffer<T>`, and
`Buffer<T>` are aliases, accessed by subscript:

```c
StructuredBuffer<Particle>   particles    : register(t1);
RWStructuredBuffer<Particle> particles_rw : register(u1);

Particle p = particles[i];
particles_rw[i] = p;
```

Unlike blocks, their elements default to **C layout** — a plain C struct with the same
members matches byte-for-byte (§5). A layout keyword before the declaration overrides:
`pack16 StructuredBuffer<Particle> particles;`.

### `pushconstant` - small, fast, no descriptor

```c
pushconstant Draw {
	float4x4 mvp;
	float4   tint;
	uint     flags;
};
```

The native spelling of `[[vk::push_constant]]`. No `register` (it isn't a descriptor);
members are used by name. Push-constant blocks default to **std430**.

---

## 5. Memory layout: `pack1`, `pack8`, `pack16`, `std430`

Layout controls the byte offset and stride of every member - critical, because the CPU side
must write matching bytes. A layout keyword prefixes the declaration, on blocks and on the
object-form buffers alike, and every mode also answers to its standards name:

```c
pack16 storagebuffer Data  : register(t0) { float4 v[]; };
std140 uniform       Scene : register(b0) { float4x4 vp; float3 eye; float t; };
pack16 RWStructuredBuffer<particle_t> parts : register(u1);
scalar readonly storagebuffer Raw { float data[]; };
```

| Keyword | Alias | Rule | Vulkan |
|---------|-------|------|--------|
| `pack1` | `scalar` | every member aligns to its scalar size; `float3` is 12 bytes; matches a tightly packed C struct | core when nothing straddles a 16-byte boundary, else the `scalarBlockLayout` device feature |
| `pack8` | `relaxed` | vectors align to `min(natural, 8)` | same rule as `pack1` |
| `pack16` | `std140` | `float3`/`float4` align to 16, array elements round up to a 16-byte stride, struct alignment >=16 | core |
| `std430` | - | vectors natural-aligned, no 16-byte array-stride rounding | core |

The aliases are context-sensitive, so `scalar` and `relaxed` remain usable as ordinary
identifiers.

Defaults and constraints:

- `uniform` - **`pack16` only** (also the default). `pack1 uniform Bad { ... }` is an error:
  `uniform buffer 'Bad' must use pack16`. The implicit `$Global` buffer (§6) is always
  `pack16`.
- `storagebuffer` / `pushconstant` blocks - default **`std430`** (the safe core layout that
  validates on every device); any explicit layout keyword is allowed.
- `StructuredBuffer<T>` / `RWStructuredBuffer<T>` / `Buffer<T>` elements - default to
  **C layout where C layout is free**. "C layout" is `pack1`/`scalar`; the default differs
  from writing the keyword only in policy. A valid pack1 layout whose vector straddles a
  16-byte boundary (or whose stride breaks std430 alignment) is only loadable on devices
  with `scalarBlockLayout` - and a bare declaration names no layout, so it must not
  silently acquire a device dependency. The default refuses such structs with a compile
  error; writing `pack1` supplies the consent, emits the same layout, and records the
  requirement in the SKS feature mask (bit 16).

How the same struct lands under each mode:

```c
struct particle_t {
	uint   id;    // C: offset 0
	float2 pos;   // C: offset 4
	float  scale; // C: offset 12, sizeof 16
};
```

| Mode | `id` | `pos` | `scale` | array stride |
|------|------|-------|---------|--------------|
| `pack1` / `scalar` | 0 | 4 | 12 | 16 |
| `pack8` / `relaxed` | 0 | 8 | 16 | 24 |
| `pack16` / `std140` | 0 | 8 | 16 | 32 |
| `std430` | 0 | 8 | 16 | 24 |

Only `pack1` matches the natural C layout byte-for-byte - under the other modes a matching
C struct needs explicit padding members, so reflected `element_size` equals the C `sizeof`
only under `pack1`. `particle_t` never straddles, so as a `StructuredBuffer` element it
needs no keyword and no device feature. A struct like `{ float a; float4 b; }` does
straddle (`b` at offset 4): the default refuses it, and `pack1` opts in. One caveat when
mirroring structs in C: SVSL `bool` is stored as a 32-bit uint - pair it with
`uint32_t`/`int32_t` on the CPU side, never C `bool`.

### Which layout should I use?

Prefer **no keyword at all**: design element structs alignment-neutral, and the C-packed
default, std430, and std140 all agree byte-for-byte - no device requirements, no hidden
padding, and the source stays valid HLSL. The recipes are mechanical: build in
`float4`-sized rows, pair every `float3` with a scalar, keep `float2` members on 8-byte
offsets, and write explicit `_pad` members rather than letting a layout engine insert
invisible ones. When the compiler refuses a struct, reordering or padding it is almost
always the best fix - that's why the error suggests it first.

Reach for a keyword when restructuring isn't an option: `std430` (or `pack16`) when
mirroring a CPU layout you don't control or porting existing code wholesale - the padding
then lives implicitly on the GPU side, and the CPU code must match it. Use `pack1` when
density is the point and no reordering can help: `{ float3 position; float3 velocity; }`
is 24 bytes packed but 32 padded or under std430, a 25% bandwidth cut on a
bandwidth-bound pass - a deliberate trade for knowing your devices support
`scalarBlockLayout`.

### Explicit member offsets

`[offset(N)]` on a member (native spelling of `[[vk::offset]]`) pins its byte offset. It may
only **move an offset forward** and must respect alignment; going backwards or misaligning is
`[offset] on 'x' moves backwards or breaks alignment`.

---

## 6. Globals

Non-block globals - material defaults, constants, and shared memory - cover several distinct
cases:

```c
float4 color = {1, 1, 1, 1};     // bare global with initializer → $Global member + default
static const float PI = 3.14159; // compile-time constant
static float3 accum;             // private per-invocation variable
workgroup float4 tile[8][8];     // workgroup-shared memory (groupshared = alias)
```

- **Bare globals with initializers** are collected into an implicit **`$Global`** uniform
  buffer, and their initializer values become **material defaults** baked into reflection
  (this is how StereoKit surfaces per-material parameters). The initializer must be a
  foldable constant. Extend and tag them with [`//--`
  metadata](08-preprocessor-and-metadata.md).
- **`static const`** is a true compile-time constant - usable in array sizes and constant
  expressions.
- **`static`** (no `const`) is a private per-invocation global - mutable, not shared.
- **`workgroup`** is compute workgroup-shared memory; `groupshared` is the HLSL alias.

---

## 7. Bindings and registers

`register()` places a resource at an explicit descriptor slot, HLSL-style:

```c
Texture2D tex : register(t0);         // prefix letter + slot
Texture2D tex : register(t0, space1); // ... with a descriptor set (space)
Texture2D tex : register(0, 1);       // SVSL direct form: (binding, set), no offsets
```

With the **prefix-letter** form, bindings are offset by register class to avoid
cross-type collisions, matching skshaderc:

| Register class | Descriptor binding |
|----------------|--------------------|
| `b#` - cbuffer / uniform | `# + 0` |
| `t#`, `s#` - texture, sampler, StructuredBuffer | `# + 100` (texture+sampler pairs share) |
| `u#` - RW anything (RWStructuredBuffer, storage image) | `# + 200` |

The **direct `(binding, set)`** form - no prefix letter - is the native spelling of
`[[vk::binding(b, s)]]` and applies **no** offset; the number you write is the descriptor
binding. `space` maps to the descriptor set in both forms.

### `register()` is optional

Because runtimes bind by *name* through reflection, most resources don't care which slot they
land in. Omit `register()` and SVSL auto-assigns:

```c
Texture2D diffuse;                // first free t slot
Sampler   diffuse_s;              // paired with diffuse
uniform Params { float4 color; }; // first free b slot
```

Auto-assignment is **deterministic** (declaration order, lowest free slot in the resource's
class, set 0) and **never collides** with explicit registers - so StereoKit's reserved slots
in `stereokit.hlsli` stay untouched even when you leave your own resources unnumbered. A
texture and its paired sampler are assigned together. Explicit-slot collisions *are* caught:
`register t0 collides with an earlier resource`.

---

Next: [Intrinsic Functions](06-intrinsics.md) - the built-in function library.
