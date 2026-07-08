# Expressions and Functions

The expression grammar is C's, the way HLSL uses it. This chapter covers the operators, the
control-flow statements, and functions - including the two things about functions that
differ from CPU C: **resources can be parameters**, and **every call is inlined**.

---

## 1. Operators

```c
+  -  *  /  %                    // arithmetic (component-wise on vectors)
== != <  <= >  >=                // comparison
&& || !                          // logical
&  |  ^  ~  << >>                // bitwise / shift (integers)
=  += -= *= /= %= &= |= ^= <<= >>=   // assignment
?:                               // ternary
++ --                            // pre/post increment and decrement
```

Vector arithmetic is component-wise, matching HLSL - including `a * b` on two matrices,
which multiplies element by element. For linear-algebra matrix and matrix/vector products
use **`mul()`** (§[Intrinsics](06-intrinsics.md)). Comparisons on vectors produce boolean
vectors; reduce them with `any()` / `all()`.

Shifts and bitwise operators require integer operands. `%` on floats is `fmod`-style.

---

## 2. Control flow

Everything C has, plus the two fragment-kill statements:

```c
if (cond) { ... } else { ... }
for (int i = 0; i < n; i++) { ... }
while (cond) { ... }
do { ... } while (cond);
switch (x) { case 0: ...; break; default: ...; }
break;  continue;  return expr;
discard;   // kill this fragment  (OpKill)
demote;    // demote to a helper invocation  (DemoteToHelperInvocation)
```

`discard` and `demote` both stop the fragment from writing output, but they differ:

- **`discard`** terminates the invocation. Derivatives (`ddx`/`ddy`, texture LOD) computed
  *after* a `discard` in neighbouring lanes are undefined.
- **`demote`** keeps the invocation running as a *helper* - it produces no output, but it
  still participates in derivative computations, so neighbours' `ddx`/`ddy` stay correct.
  Query the state with `is_helper_invocation()`.

```c
[early_depth_stencil]
float4 ps(psIn input) : SV_Target {
	float a = tex.Sample(smp, input.uv).a;
	if (a < 0.5) demote;          // helper from here on...
	float2 d = ddx(input.uv);     // ...so this derivative is still well-defined
	return shade(input, d);
}
```

`demote`, `is_helper_invocation`, and `[early_depth_stencil]` are SVSL-dialect features with
no HLSL/glslang spelling. `discard` and the `clip(x)` intrinsic (kill if any component < 0)
are the HLSL-compatible forms.

### Loop and branch attributes

`[unroll]`, `[loop]`, `[branch]`, `[flatten]` are accepted wherever HLSL accepts them and
are lowered to the matching SPIR-V control mask, so they reach the driver as real intent:

| attribute | on | effect |
|---|---|---|
| `[flatten]` | `if` | `OpSelectionMerge … Flatten` - predicate both sides instead of branching |
| `[branch]`  | `if` | `OpSelectionMerge … DontFlatten` - keep the real branch |
| `[unroll]`  | `for`/`while`/`do` | `OpLoopMerge … Unroll` |
| `[loop]`    | `for`/`while`/`do` | `OpLoopMerge … DontUnroll` |

These carry information the backend cannot infer - whether a branch is cheap enough to
predicate, or a loop's trip count is worth unrolling. They never change correctness, only
code shape, and a strong driver may still override them. `[fastopt]` is accepted but has no
SPIR-V equivalent (advisory only); `[unroll(n)]` is accepted but not yet distinguished from
`[unroll]`.

---

## 3. Functions

Functions look like C. Parameters default to `in`; `out` and `inout` pass results back:

```c
float3 tonemap(float3 hdr) {
	return hdr / (hdr + 1);
}

void minmax(float a, float b, out float lo, out float hi) {
	lo = min(a, b);
	hi = max(a, b);
}

void accumulate(inout float4 sum, float4 sample) {
	sum += sample;
}
```

`out`/`inout` arguments become SSA copies at the call boundary (copy-in/copy-out), so there
is no aliasing surprise. The compiler enforces that an argument passed to an `out` parameter
is **writable** and that its type **matches the parameter exactly** - no implicit conversion
sneaks in on the way back out.

**Forward references need no prototypes.** The parser scans all top-level names before
resolving bodies, so a function may call another declared later in the file. There are no
separate declarations to keep in sync.

### 3.1 Resources as parameters

This is the feature that makes SVSL's helper functions actually reusable: **textures,
samplers, and images may be function parameters.**

```c
float4 sample_fade(Texture2D t, SamplerState s, float2 uv, float fade) {
	return t.Sample(s, uv) * fade;
}

float4 ps(psIn input) : SV_Target {
	float4 a = sample_fade(albedo,   albedo_s,   input.uv, 1.0);
	float4 b = sample_fade(emissive, emissive_s, input.uv, input.glow);
	return a + b;
}
```

SPIR-V has no way to pass an opaque resource as a runtime value, so on the face of it this
can't work. It works because of the next section.

### 3.2 Every call is inlined

**SVSL fully inlines every user-function call during compilation.** There is exactly one
SPIR-V function per entry point; all your helpers are folded into it. This is deliberate and
load-bearing:

- **Resource parameters resolve at inline time.** When `sample_fade` is inlined into `ps`,
  its `Texture2D t` parameter is replaced by the concrete global (`albedo` or `emissive`) at
  that call site. No opaque value is ever passed at runtime.
- **`out`/`inout` become plain copies** once inlined - no pointer indirection reaches the
  SPIR-V.
- **Recursion is a compile error.** With every call inlined, a cycle in the call graph has
  no base case to stop the expansion. You will get:
  `recursion involving 'f' (SPIR-V forbids recursion)`. SPIR-V forbids recursion regardless;
  inlining just makes the rule fall out naturally.

You do not need to think about inlining while writing - write small functions freely, they
cost nothing at runtime. Just know that resource parameters and the no-recursion rule both
follow from it.

---

## 4. A note on where you can and can't declare things

- **Resources are globals only.** You cannot declare a `Texture2D`, `Sampler`, buffer, or
  image as a local variable inside a function: `resources cannot be declared locally`. Pass
  them in as parameters instead (§3.1).
- **Local variables** are ordinary: `float x = ...;`, arrays, structs. `static const` at
  global scope is a compile-time constant; `static` (without `const`) is a private
  per-invocation global. See [Resources and Buffers](05-resources-and-buffers.md#6-globals).

> **Inline SPIR-V.** For an opcode the language has no keyword for, `spirv_asm(T) { … }` is an
> expression that splices raw SPIR-V into the function body and yields a value of type `T`.
> It's the escape hatch to the full instruction set - see [Advanced Features → Inline
> SPIR-V](07-advanced-features.md#inline-spir-v--spirv_asm).

---

Next: [Stages and Interfaces](04-stages-and-interfaces.md) - turning these functions into
shader stages and wiring their inputs and outputs.
