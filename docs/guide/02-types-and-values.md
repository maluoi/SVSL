# Types and Values

SVSL's type system is HLSL's, with the widths made explicit and one genuinely new
distinction: `half` and `float16` are *different types* with different portability
contracts. This chapter covers every type, the literal forms, and how you build and convert
values.

---

## 1. Literals

```c
42            // int32                 42u    42U      // uint32
42L    42l    // int64                 42uL   42lu     // uint64 (u+l in any case/order)
0xFF          // hex   int32           0b1010          // binary int32
0755          // octal int32           0xFFu           // hex uint32
3.14          // float32               3.14f  3.14F    // float32 (explicit)
0.5h   0.5H   // half                  3.14lf 3.14LF   // float64
.5     1.     1e-3   6.022e23          // the usual float spellings
true   false  // bool
```

The suffix picks the type. `f`/`h`/`lf` also promote an integer-looking literal to a float
of that kind - `1f` is a `float32`, `1h` is a `half`, exactly as HLSL allows. Float suffixes
on a hex or binary literal are an error. An integer suffix (`u`, `l`) on a value with a
decimal point is an error.

> The `h` suffix produces a **`half`**, not a `float16`. The two are different types - see
> §3.

---

## 2. Scalar types

| Type | Meaning | Alias |
|------|---------|-------|
| `bool` | boolean (32-bit in memory interfaces) | |
| `int8` `int16` `int32` `int64` | signed integers, exact width | `int` = `int32` |
| `uint8` `uint16` `uint32` `uint64` | unsigned integers, exact width | `uint` = `uint32` |
| `float16` | **exactly** 16-bit float | |
| `float32` `float64` | 32/64-bit floats | `float` = `float32`, `double` = `float64` |
| `half` | **at least** 16-bit float (see §3) | `min16float` = `half` |

The explicit-width names are the SVSL-native spelling; `int`/`uint`/`float`/`double` are the
familiar aliases and compile without complaint. `min16float` is a straight alias of `half`,
accepted silently.

**You never declare SPIR-V capabilities.** Using a non-32-bit width infers the right one
automatically: `int8`/`uint8` → `Int8`, `int16`/`uint16` → `Int16`, `int64` → `Int64`,
`float64` → `Float64`, `float16` → `Float16` - plus the matching 8-/16-bit *storage*
capabilities when such a type appears inside a buffer. See [capability
inference](07-advanced-features.md#capability-inference).

---

## 3. `half` vs `float16` - the one thing that is not HLSL

This is the language's most important type decision, so it gets its own section.

**`half` means "at least 16-bit precision."** It compiles to a `float32` decorated with
`RelaxedPrecision`. Hardware with fast fp16 *may* run it at reduced precision; hardware
without simply runs it as float32. It is always **4 bytes** in a buffer, so your memory
layouts never shift depending on what a device supports. This is the portable default, and
it is exactly what HLSL's `min16float` compiles to under glslang.

**`float16` means "exactly 16 bits, always."** It requires the `Float16` capability (and
16-bit storage capabilities when it lands in a buffer), and it is **2 bytes** in a buffer.
If the target environment cannot express it, compilation fails with a clear diagnostic. Use
it when you control the hardware and want the bandwidth and throughput win.

| | `half` | `float16` |
|---|--------|-----------|
| Precision | ≥ 16 bits (relaxed) | exactly 16 bits |
| SPIR-V | `float32` + `RelaxedPrecision` | true 16-bit float |
| Buffer storage | 4 bytes | 2 bytes |
| Capability | none | `Float16` (+ storage caps) |
| Unsupported target | always works (degrades to fp32) | compile error |
| Use it for | portable "this can be low precision" | deliberate perf/bandwidth on known hardware |

If you want to *measure* the fp16 win, the compiler option `--half=strict16` re-types every
`half` in the program as `float16` in one shot - no source edits. Ship without it.

The `h` literal suffix makes a `half`. In practice, StereoKit shaders are wall-to-wall
`min16float` (→ `half`); treat `half` as your default reduced-precision type and reach for
`float16` only deliberately.

---

## 4. Vectors and matrices

Every scalar type takes a `2`/`3`/`4` vector suffix and an `RxC` matrix suffix:

```c
float2 uv;        half4  color;      int3 coords;    bool4 mask;      uint2 size;
float4x4 viewProj;   float3x3 normalMat;   float3x4 skin;   // <rows> x <cols>
```

**Matrices are column-major.** `mul()` is the matrix/vector multiply (all of HLSL's
overloads); `a * b` on matrices is component-wise, as in HLSL, so use `mul` for linear
algebra.

Access forms:

```c
float4 row = m[0];        // row 0 as a vector - m[i] indexes rows, HLSL-style
float  e   = m._m00;      // element, 0-based: _m<row><col>, _m00 .. _m33
float  e2  = m._11;       // element, 1-based: _<row><col>,   _11 .. _44
float3x3 r = (float3x3)m; // truncation cast (e.g. 4x4 world → 3x3 for normals)
```

Indexing a matrix with `m[i]` selects **row** *i* as a vector, following HLSL - even though
the storage convention is column-major. There are no matrix swizzles - you read one element
at a time. Vectors *do* swizzle (§7).

---

## 5. Arrays

```c
float4   positions[64];
float4x4 bones[MAX_BONES];     // size is any constant expression...
float4x4 slots[TILE];          // ...including a specialization constant
float    data[];               // runtime-sized: last member of a storagebuffer only
```

Array sizes are constant expressions, which includes `#define` constants and
[specialization constants](07-advanced-features.md#specialization-constants). A
**runtime-sized** array (`[]`) is legal only as the final member of a `storagebuffer` block;
its length comes from the bound buffer at runtime.

---

## 6. Structs

```c
struct Vertex {
	float3 position;
	float3 normal;
	float2 uv;
};
```

Structs group data, describe stage interfaces (members carry semantics - see
[Stages](04-stages-and-interfaces.md)), and define the element type of a
`StructuredBuffer<T>`. There is no inheritance and there are no methods; a struct is plain
data. Members are accessed with `.`, and a whole struct can be copied by assignment or
returned from a function.

### Packed bit fields

Compute shaders often hand-pack many small values into a few words to save bandwidth. Give
a member a `: width` suffix and the whole struct becomes **packed** - its members turn into
dense bit fields laid out LSB-first inside a run of backing `uint32` words, just like C bit
fields:

```c
struct Packed {
	uint  a : 4;     // raw unsigned, low 4 bits
	int   b : 6;     // raw signed - sign-extended when you read it
	bool  f : 1;     // a 1-bit flag
	float c : un10;  // unorm: a [0,1] float quantized to 10 bits
	half  d : sn6;   // snorm: a [-1,1] float quantized to 6 bits
	half  h : 16;    // the raw 16-bit half bit pattern
	uint  e;         // no ':' → its natural width (32 bits), in its own word
};
```

The type on the **left** of the colon is what you read and write in code (the *resolve
type*); the token on the **right** is how it's stored. You work in normal types and the
compiler does the shifting and masking:

```c
buf[i].a = 13u;         // masked into 4 bits
buf[i].b = -7;          // stored two's-complement, read back as -7
buf[i].c = 0.75;        // stored as trunc(0.75 * 1023)  - truncates toward zero
bool on  = buf[i].f;    // 1-bit field reads back as a bool
```

The right-hand side is one of:

| right side | stored as | resolve types |
| --- | --- | --- |
| `N` | raw low `N` bits, or a whole 16/32-bit float pattern | int/uint (any size), `bool` (`:1`), `half` (`:16`), `float` (`:32`) |
| `unN` | unorm - `[0,1]` across `N` bits (`N` ≤ 24) | `float`, `half` |
| `snN` | snorm - `[-1,1]` across `N` bits (`N` ≤ 24) | `float`, `half` |

A few rules, all matching C:

- **No field crosses a 32-bit word.** A field that wouldn't fit in the current word starts
  the next one (that's why `h` above lands in word 1). `Packed` ends up 3 words = 12 bytes.
- **A member with no `:` takes its type's natural width** - `uint8` is 8 bits, `half` is
  16, `uint`/`float` are 32 - so packed and whole members mix freely. A type wider than 32
  bits needs an explicit width.
- **Writes truncate.** `un10 = 0.75` stores `round-toward-zero(0.75 * 1023)`; nothing
  clamps an out-of-range value for you. Bit-packing is a speed tool - clamp yourself if you
  need it.

Because the struct's storage is exactly its backing words, it round-trips with a matching
CPU-side layout in a `StructuredBuffer<Packed>`.

For explicit one-off packing without declaring a struct, the
[`bitfield_extract` / `bitfield_insert`](06-intrinsics.md) intrinsics do the same
extract/insert on a bare integer.

### Enums

An `enum` gives names to integer constants - and, when you name the enum, a type alias
for the underlying integer:

```c
enum Mode : uint { ModeA = 1, ModeB = 4, ModeC };  // ModeC counts on: 5
enum { Red, Green, Blue };                          // just three global constants
enum : int16 { One, Two } option = One;             // anonymous, with an inline variable
```

The constant names are **global**, like an HLSL unscoped `enum` - you write `ModeA` or
`Red`, not `Mode::ModeA`. Values start at `0` and increment unless you assign one, after
which counting continues from there.

A named enum is *an alias for its underlying integer type* - there's no separate enum type
and no strict type checking. `Mode m = ModeB;` is really a `uint`, and an enum value
compares, switches, and does arithmetic just like the integer it is:

```c
Mode m = ModeB;
if (m == ModeC) { ... }
switch (m) { case ModeA: ...; case ModeB: ...; }
uint next = m + 1;
```

The underlying type after `:` must be an integer (it defaults to `int`). Enum constants
are compile-time constants, so they work as array sizes, `switch` labels, and - because an
enum *is* an integer - as [packed bit-field](#packed-bit-fields) resolve types:

```c
enum Facing : uint { N, E, S, W };
struct Tile { Facing dir : 2; uint hp : 6; };   // 'dir' is a raw 2-bit field
```

You can also write an `enum { … }` **inline** anywhere a type is expected - a parameter, a
struct member, or a local - when you don't want to name it separately:

```c
uint doThing(enum { Thing1, Thing2 } which) { return (uint)which; }
struct Widget { enum : uint { Hidden, Shown } vis : 1;  uint id : 31; };
```

The constants are still global (you call `doThing(Thing2)` from anywhere), so two inline
enums can't reuse a constant name in the same file.

The trade-off for this simplicity is no nominal safety: nothing stops you assigning an
unrelated integer (or another enum's constant) to a `Mode`. Enums here are a naming and
convenience tool, not a distinct type.

---

## 7. Swizzles

Vector components are named with either the `xyzw` set or the `rgba` set - equivalent, but
you can't mix the two sets inside one swizzle. A swizzle selects 1–4 components in any
order:

```c
float4 c = float4(1, 0.5, 0.2, 1);

float3 rgb  = c.rgb;         float2 rg = c.rg;      float bgr = c.b;  // 1 component → scalar
float3 flip = c.bgr;         float4 dup = c.rrrr;   // repetition ok when reading
c.xw  = float2(0, 0);        // swizzle write (partial assignment)
c.rgb = c.bgr;               // reordered swizzle write
float3 broadcast = c.r.xxx;  // swizzling a scalar broadcasts
```

A swizzle *write* must name each component at most once and can't mix sets:

```c
c.xx = uv;   // ERROR: 'x' written twice
c.xg = uv;   // ERROR: mixed xyzw / rgba sets
```

---

## 8. Constructors and casts

```c
float4 a = float4(v3, 1);        // concatenation: a vec3 plus a scalar
float4 b = float4(1);            // broadcast: (1, 1, 1, 1)
float2 c = float2(x, y);         // component-wise
float4x4 i = float4x4(1);        // identity matrix from a scalar
float3x3 m3 = (float3x3)world;   // truncation cast

int   n = (int)x;                // explicit scalar cast
float f = 3;                     // implicit widening where HLSL allows it
```

Constructors take any mix of scalars and shorter vectors as long as the component counts add
up. Broadcasting a single scalar fills a whole vector or an identity matrix. Casting a larger
matrix to a smaller one truncates (the classic `(float3x3)world` for transforming normals).

Implicit conversions follow HLSL's rules (widening is silent, narrowing needs a cast in the
places HLSL requires one). The exact conversion table is enforced by sema; when a conversion
is illegal you get an error naming both types.

---

Next: [Expressions and Functions](03-expressions-and-functions.md) - what you can *do* with
these values.
