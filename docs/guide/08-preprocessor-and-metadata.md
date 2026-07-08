# Preprocessor and Metadata

Two text-level facilities run before the language proper: a conventional C preprocessor, and
the `//--` material-metadata annotations that StereoKit reads out of the compiled shader.
Both are handled in a single pass that keeps a **line map**, so every diagnostic points at
the original file and line even after includes and macro expansion.

---

## 1. The preprocessor

SVSL's preprocessor is the C subset shaders actually use.

### Includes

```c
#include "common.hlsli"      // quote form
#include <stereokit.hlsli>   // angle form - resolved identically
```

Both forms resolve the same way: the embedder's include callback (the compiler tries the
requesting file's own directory first, then each `-i` search path). Include guards and
`#pragma once` both work:

```c
#ifndef COMMON_HLSLI
#define COMMON_HLSLI
// ...
#endif
```

```c
#pragma once
```

Any other `#pragma` is accepted and ignored with an `info` diagnostic
(`ignoring '#pragma ...'`), so vendor pragmas in ported code don't break the build.

### Defines

```c
#define SK_MAX_VIEWS 6                 // object-like
#define SQR(x) ((x) * (x))            // function-like
#define LERP(a, b, t) ((a) + ((b) - (a)) * (t))
#undef  SQR
```

Object-like and basic function-like macros are supported. The variadic (`...`) and token
`#`/`##` operators are **not** - they aren't needed by the target corpus. `#define` constants
count as constant expressions, so they can size arrays and feed `#if`.

### Conditionals

```c
#if   SK_MAX_VIEWS > 2
#elif defined(SK_SINGLE_VIEW)
#else
#endif

#ifdef  SK_OPENGL
#ifndef SHADOWS
```

`#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif` with full constant-expression evaluation
and the `defined()` / `defined X` operator. Mismatched directives (`#elif without #if`,
`#elif after #else`) are errors.

### The `include` language directive

There is also a language-level `include "file"` (no `#`) - it resolves through the same
callback and is idempotent (including the same file twice is a no-op). Use `#include` for
HLSL-style headers; `include` is the native form when you want include-once semantics without
a guard.

---

## 2. Material metadata (`//--`)

A comment that begins with `//--` is not a comment to the compiler - it is a **metadata
annotation** attached to the shader and carried into the `.sks` container for the runtime and
editor. Metadata is extracted in the preprocessor pass, *before* comments are stripped, and
the form works inside `/* */` blocks too. Only the top-level source file is scanned - `//--`
annotations inside `#include`d files are ignored.

The shape is:

```
//--<param>[:<tag>] = <value>
```

Real annotations from the corpus:

```c
//--name = sk/unlit               // the shader's unique name (else the filename is used)
//--diffuse = white               // default texture name - interpreted by the consuming engine
//--color:color = 1, 1, 1, 1      // 'color' carries a UI tag and a default value
//--uv_scale: range(0, 2) = 0.5   // a UI hint, stored in the parameter's 'extra' field
//--wave_size = 32                // required subgroup size - alias of [wave_size(32)]
```

What each part does:

- **`name`** sets the shader's identity in the container. Without it, the filename is used.
- **A bare `param = value`** naming a resource (like `diffuse = white`) sets that resource's
  default binding. The value is an opaque string SVSL carries into reflection verbatim - the
  **consuming engine** interprets it (StereoKit, for example, maps names like
  `white`/`black`/`gray`/`normal` to its built-in textures). SVSL validates nothing here, so
  the accepted names are whatever your runtime defines.
- **A `param = value` naming a bare global** provides or overrides that parameter's default.
  Bare-global initializers ([Resources §6](05-resources-and-buffers.md#6-globals)) already
  supply numeric defaults; `//--` values override and extend them.
- **The `:tag`** (e.g. `:color`) marks how the editor should present the parameter.
- **A hint like `range(0, 2)`** is stored in the parameter's `extra` field for UI tooling.
- **`wave_size`** is the metadata alias for the `[wave_size(N)]` entry attribute; same
  power-of-two-in-[4,128] rule, and it lands in reflection.

Between bare-global initializers and `//--` annotations, a material's default parameter block
is fully described in the shader source and round-trips into the container - no external
material definition file is needed.

---

## 3. What the reflection looks like

All of the above - buffers, resources, vertex inputs, entry points, material defaults - is
what `svslc -r` prints and what lands in the `.sks` container. Seeing it makes the
"names over numbers" principle concrete. For this shader:

```c
//--name = example/tinted
//--tint:color = 1, 1, 1, 1

float4 tint = {1, 1, 1, 1};              // bare global → $Global member + default

Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);   // fuses with diffuse

cbuffer Params : register(b0) { float4x4 mvp; };

psIn   vs(vsIn i)  { ... }
float4 ps(psIn i) : SV_Target { ... }
```

`svslc -r` reports:

```
shader example/tinted
entry vertex   vs
entry pixel    ps
buffer Params : b0 space0 - 64 bytes
  mvp              +0         64b  float4x4
buffer $Global : b1 space0 - 16 bytes (has defaults)
  tint             +0         16b  float4 [color] = 1, 1, 1, 1
resource texture    diffuse          : t0 space0
input float4     : SV_Position
input float2     : TEXCOORD0
```

Three things to notice, each an earlier principle made visible:

- **`$Global` auto-assigned to `b1`.** `Params` claimed the explicit `b0`, so the implicit
  bare-globals buffer took the next free slot - auto-assignment never collides with an
  explicit register ([Resources §7](05-resources-and-buffers.md#7-bindings-and-registers)).
- **The sampler is gone.** `diffuse_s` fused into `diffuse`; only the combined texture is a
  resource, and it's named after the texture.
- **The default travelled out.** `tint`'s initializer and its `:color` tag are both in the
  reflection, ready for a material editor - no separate material file.

---

Next: [HLSL Compatibility](09-hlsl-compatibility.md) - the complete porting reference.
