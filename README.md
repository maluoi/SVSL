<p align="center">
	<img src="docs/media/logo.svg" alt="SVSL" width="360">
</p>

# SVSL - SPIR-V Shading Language

SVSL is a shader language that compiles the procedural core of HLSL, and takes it a step further into some of the latest SPIRV features! This project was born to solve 3 problems:
- Compiler binary size (svsl is ~300kb).
- Compiler build time (svsl is ~2s).
- Access to modern SPIRV features.

This allows you to build and ship SVSL inside your codebase without bloating your binaries and spiking your build times! Instead of compiling your shaders in advance by necessity due to a large complex toolchain, you can compile them on-the-fly with a built-in shader compiler!

SVSL produces SPIRV comparable in quality to glslang + SPIRV-Opt. It is still slightly wordier, but after GPU optimizations, produces very comparable ISA (at least on Radeon GPUs!).

This project was created almost entirely with supervised Claude Fable/Opus 4.8. Take that as you will, but this project does have some pretty thorough tests.

Check out the [language docs](docs/guide/), or check below for an overview of the library!

## A taste

SVSL reads like the HLSL you already know! Every construct below has an HLSL spelling that still compiles. Where SVSL gives a modern SPIR-V feature a first-class keyword, the comment names the HLSL form it replaces.

A **fragment shader** - a dissolve/burn effect in relaxed precision:

```c
// Reflection metadata for your project
//--name = demo/dissolve
//--albedo = white
//--edge:range(0, 1) = 0.5

Texture2D albedo;
Sampler   albedo_s; // HLSL: Texture2D + SamplerState (fuse into one binding)
Texture2D noise;
Sampler   noise_s;

pushconstant Burn { float edge; float3 ember; };  // HLSL: [[vk::push_constant]] cbuffer Burn { ... }

[fragment] // Explicitly set the entry point for the fragment shader
half4 ps(float4 pos : SV_Position, float2 uv) : SV_Target { // half4 -> HLSL min16float4 (relaxed precision)
	half4 base = (half4)albedo.Sample(albedo_s, uv);
	float n    = noise.Sample(noise_s, uv).r;

	if (n < edge) demote; // demote, not discard - ddx/ddy stay valid for survivors
	half glow  = (half)saturate(1.0 - (n - edge) * 12.0);
	return half4(lerp(base.rgb, (half3)ember, glow), base.a);  // ember-colored burn edge
}
```

A **compute shader** - aging GPU particles whose state is hand-packed into tight bit fields, something HLSL can't even declare:

```c
//--name = demo/particles

// HLSL: [[vk::constant_id(0)]] const uint
specialization const uint32 MAX_LIFE = 240;

// Bit packed structs! ": width" packs the whole struct into dense bit fields,
// LSB-first HLSL has no struct bit fields - you'd shift & mask by hand
struct Spark {
	uint  cell : 22;
	uint  life : 8;
	bool  hot  : 1;    // cell + life + hot = 31 bits -> one uint32 word
	float fade : un16; // unorm: a [0,1] float stored in 16 bits -> starts a second word
};

storagebuffer Sparks { Spark spark[]; }; // HLSL: RWStructuredBuffer<Spark>
storagebuffer Alive  { uint  count[]; }; // HLSL: RWStructuredBuffer<uint>

[compute(64, 1, 1)] // HLSL: [numthreads(64, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	Spark s = spark[id.x]; // read once - fields unpack into plain uint/float/bool
	if (s.life == 0u) return;

	s.life = s.life - 1u;            // work in normal types; the compiler packs & unpacks for you
	s.hot  = s.life > MAX_LIFE / 2u; // MAX_LIFE/2u folds to OpSpecConstantOp - still specializable
	s.fade = s.fade * 0.94;          // written as a [0,1] float, stored back into 16 unorm bits
	spark[id.x] = s;                 // repacked into the two backing words

	atomic_add(count[0], 1u); // HLSL: InterlockedAdd
}
```

HLSL native spellings (`cbuffer`, `RWTexture2D`, `Wave*`, `Interlocked*`, `[[vk::binding]]`, ...) all compile just fine! You can also ask for optional `porting` hints pointing at the alternative SVSL form. Beyond what's shown: strict `float16`, subgroup/wave ops, multiview, storage images with native format syntax, subpass inputs (`SubpassInput`, `SubpassInputMS`), tile images, `Texture2DMS`, the early-Z toolkit, and `[wave_size(N)]`, and more!

A friendly tour of the language lives in [docs/guide/](docs/guide/)!

## Building

```bash
cmake -B build
cmake --build build

./build/svslc shader.hlsl        # compile a shader → shader.sks
```

To pull SVSL into your own CMake project, use FetchContent! This builds just the static
library (the CLI, tests, and install rules stay off automatically), and linking `svsl::svsl`
brings the `<svsl/svsl.h>` include path with it:

```cmake
include(FetchContent)
FetchContent_Declare(svsl
	GIT_REPOSITORY https://github.com/maluoi/SVSL.git
	GIT_TAG        main)   # pin a tag or commit for reproducible builds
FetchContent_MakeAvailable(svsl)

target_link_libraries(my_app PRIVATE svsl::svsl)
```

## Using svslc

```bash
svslc shader.hlsl                          # → shader.sks
svslc -spv -o out/ shader.hlsl             # → out/shader.vert.spv, .frag.spv, ...
svslc -h  -o shader.h shader.hlsl          # → C header with embedded container
svslc -r shader.hlsl                       # print reflection (bindings, defaults, ...)
svslc --validate shader.hlsl               # compile + spirv-val, no file output
svslc --half=strict16 shader.hlsl          # every `half` becomes a true float16
```

Entry points are found by name (`vs`/`ps`/`cs`, override with `-vs/-ps/-cs`) or by `[vertex]`/`[fragment]`/`[compute(x,y,z)]` attributes.

## Using libsvsl

`#include <svsl/svsl.h>` - the single public header. The whole surface is compile source -> SPIR-V/SKS, and parse SKS back. Every allocation for a compile lives in one arena owned by the result; free it all with one call.

```c
#include <svsl/svsl.h>

svsl_result_t *result = svsl_compile(
	&(svsl_source_t ){ .text      = source, .filename = "shader.hlsl" },
	&(svsl_options_t){ .opt_level = svsl_opt_default });

for (int32_t i = 0; i < r->diagnostic_count; i++)
	report(&r->diagnostics[i]); // severity, source loc, message

if (result->ok) {
	for (int32_t i = 0; i < r->stage_count; i++)
		upload(r->stages[i].stage, r->stages[i].spirv, r->stages[i].spirv_word_count);

	svsl_bytes_t sks = svsl_result_sks(result); // the StereoKit container
	// also: svsl_result_header(result, name), svsl_result_reflection(result), svsl_result_ir(result)
}
svsl_result_free(result); // frees everything above
```

Reading `.sks` containers is self-contained too.

```c
svsl_sks_file_t *file = svsl_sks_parse(bytes, size); // NULL if not a valid SKS
// file->buffers, file->resources, file->vertex_inputs, file->spec_consts, file->stages (SPIR-V)
svsl_sks_free(file);
```

`#include` resolution is a caller callback (`svsl_options_t.include_cb`), so the core never touches the filesystem. `svslc` itself is built entirely on this API.

## Testing

```bash
ctest --test-dir build --output-on-failure   # unit + corpus tests
./build/svslc --validate tests/shaders/ported/*.hlsl
./build/app/svsl_view -test tests/shaders    # batch visual comparison vs skshaderc
./build/app/svsl_view -file <shader.hlsl>    # single-shader detail + SPIR-V diff
```

Correctness is judged on **outputs only**: rendered pixels (average error < 0.01 against skshaderc) and compute buffer contents (bit-exact). A deterministic mutation fuzzer (`svsl_fuzz` target) covers the front end.

## Status

SVSL is an early life compiler, and hasn't been battle hardened yet! Bug reports are very welcome. SVSL is also missing certain newer features that I simply don't have experience or use for yet! Mesh shaders, ray tracing, neural extensions, bindless rendering. Feel free to ask for or contribute these features if desired!
