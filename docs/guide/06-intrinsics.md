# Intrinsic Functions

SVSL ships the full HLSL math intrinsic library plus native functions for the modern
concepts HLSL reaches through `Wave*`/`Interlocked*`/barrier intrinsics. This chapter is the
reference. Resource *methods* (`.Sample`, `.Load`, `.GetDimensions`, …) live in
[Resources](05-resources-and-buffers.md); this chapter is the free functions.

Throughout, **`T`** is a scalar or vector generic - an intrinsic written for `T` operates
component-wise on a vector and returns the matching shape.

---

## 1. Math

### Trigonometric and exponential

```
sin  cos  tan  asin  acos  atan  atan2  sinh  cosh  tanh
pow  exp  exp2  log  log2  log10  sqrt  rsqrt  rcp
degrees  radians  ldexp
```

All are float, component-wise. `rsqrt` is the reciprocal square root; `rcp` is the
reciprocal. `atan2(y, x)`, `pow(x, y)`, `ldexp(x, exp)` take two arguments.

### Common math

```
abs  sign  floor  ceil  trunc  round  frac  fmod
min  max  clamp  saturate  lerp  step  smoothstep  fma  mad
```

`abs`/`sign` accept integers or floats; the rest are float. `clamp(x, lo, hi)`,
`lerp(a, b, t)`, `smoothstep(edge0, edge1, x)`, `step(edge, x)`, `fma(a, b, c)` and its alias
`mad`.

### Multiple-result math (out-parameters)

```c
float s, c; sincos(angle, s, c);     // s = sin, c = cos
float ip;   float f = modf (x, ip);  // f = fraction, ip = integer part
float exp;  float m = frexp(x, exp); // m = mantissa,  exp = exponent
```

The out-parameters must be writable variables of the matching type.

---

## 2. Vector and matrix

```
length  distance  normalize  dot  cross  reflect  refract  faceforward   // vector
mul  transpose  determinant  inverse                                     // matrix
```

`length`/`distance`/`dot` return a scalar; `normalize`/`reflect`/`faceforward` return the
input shape; `cross` is `float3`-only; `refract(i, n, eta)` takes a scalar `eta`.

**`mul`** is *the* matrix operation - matrix×vector, vector×matrix, and matrix×matrix, in all
of HLSL's overloads. Remember: `a * b` on matrices is component-wise; use `mul` for real
products. `transpose`, `determinant`, and `inverse` do what they say. (These four are
shape-generic and validate their argument shapes specially.)

---

## 3. Logic and classification

```
any  all  select  isnan  isinf
```

`any(v)` / `all(v)` reduce a boolean vector to a `bool`. `select(cond, a, b)` is a
component-wise ternary. `isnan`/`isinf` return a boolean of the input shape.

## 4. Bit operations

```
countbits  reversebits  firstbithigh  firstbitlow

T    bitfield_extract(T value, uint offset, uint bits)          // read a bit field
T    bitfield_insert (T base, T value, uint offset, uint bits)  // write a bit field
```

Integer, component-wise. `countbits` returns the population count; `firstbithigh`/
`firstbitlow` return bit indices.

`bitfield_extract` returns the `bits`-wide field starting at `offset`, right-justified;
`bitfield_insert` returns `base` with that field replaced by `value`'s low `bits` bits.
`T` is any integer scalar/vector - a **signed** value sign-extends on extract, unsigned
zero-extends. They lower to single hardware instructions (`OpBitFieldSExtract` /
`OpBitFieldUExtract` / `OpBitFieldInsert`), so packing several values into one word is
cheap:

```c
uint p = 0u;
p = bitfield_insert(p, flags, 0u, 4u);    // 4-bit field at bit 0
p = bitfield_insert(p, count, 4u, 12u);   // 12-bit field at bit 4
uint count_back = bitfield_extract(p, 4u, 12u);
```

These are native SVSL (glslang has no HLSL spelling); they are the explicit, one-off form of
the [struct bit-field packing](02-types-and-values.md#packed-bit-fields) - the same
extract/insert on a bare integer instead of a declared packed member.

## 5. Reinterpret and pack

```
asfloat  asuint  asint           // bitwise reinterpret, same width
f16tof32  f32tof16               // half<->float bit conversions
pack_unorm4x8    unpack_unorm4x8 // float4 <-> packed uint32
pack_snorm4x8    unpack_snorm4x8
pack_half2x16    unpack_half2x16 // float2 <-> packed uint32
```

`asfloat`/`asuint`/`asint` reinterpret the bit pattern of a 32-bit value without converting
its numeric value. The `pack_*`/`unpack_*` pairs move between a normalized/half vector and
its packed `uint32` representation.

---

## 6. Fragment-only intrinsics

```
ddx  ddy  ddx_coarse  ddy_coarse  ddx_fine  ddy_fine  fwidth
clip(x)                       // discard the fragment if any component < 0
is_helper_invocation()        // true in a demoted helper lane
tile_depth()  tile_stencil()  // tile-image depth/stencil reads (see ch. 5)
```

Derivatives are valid only in the fragment stage. `clip(x)` is the HLSL discard-on-negative
helper. `is_helper_invocation()` pairs with `demote` (see
[Expressions](03-expressions-and-functions.md#2-control-flow)). `tile_depth`/`tile_stencil`
are tile-image reads (see [Resources](05-resources-and-buffers.md#3-multisampled-subpass-and-tile-image-inputs)).

---

## 7. Subgroup (wave) operations

Subgroup ops coordinate the lanes of one hardware wave. The required
`GroupNonUniform*` capability for each family is inferred from use. These are the **native**
names; the HLSL `Wave*`/`Quad*` intrinsics are accepted aliases (§7.4).

### 7.1 Built-in lane variables

Read-only values (no call syntax):

```c
uint32 subgroup_size;      // lanes in the subgroup
uint32 subgroup_lane_id;   // this lane's index
uint32 subgroup_id;        // this subgroup's index within the workgroup
uint32 num_subgroups;      // subgroups in the workgroup
```

### 7.2 Vote, ballot, broadcast

```c
bool   subgroup_elect();                       // exactly one lane gets true
bool   subgroup_all(bool v);                   // v true in all active lanes?
bool   subgroup_any(bool v);                   // v true in any active lane?
bool   subgroup_all_equal(T v);                // v equal across lanes?
uint4  subgroup_ballot(bool v);                // bitmask of lanes where v is true
uint32 subgroup_ballot_bit_count(bool v);      // popcount of the ballot
uint32 subgroup_ballot_exclusive_bit_count(bool v);   // prefix popcount
T      subgroup_broadcast(T v, uint32 lane);   // value from a specific lane
T      subgroup_broadcast_first(T v);          // value from the lowest active lane
```

### 7.3 Reductions and scans

Each of the seven operators - `add mul min max and or xor` (`and`/`or`/`xor` are
integer-only) - comes in three forms:

```c
T subgroup_add(T v)              // reduction: sum across all active lanes
T subgroup_inclusive_add(T v)    // inclusive prefix scan
T subgroup_exclusive_add(T v)    // exclusive prefix scan
T subgroup_clustered_add(T v, uint32 clusterSize)   // reduction within fixed-size clusters
```

…and likewise `subgroup_mul/min/max/and/or/xor` with the same `inclusive_`, `exclusive_`, and
`clustered_` variants. Shuffles and quad ops:

```c
T subgroup_shuffle     (T v, uint32 lane)    // value from lane
T subgroup_shuffle_xor (T v, uint32 mask)    // value from (lane_id ^ mask)
T subgroup_shuffle_up  (T v, uint32 delta)   // value from (lane_id - delta)
T subgroup_shuffle_down(T v, uint32 delta)   // value from (lane_id + delta)

T quad_broadcast(T v, uint32 quadLane)       // 2x2 quad: value from a quad lane
T quad_swap_horizontal(T v)                  // swap across the quad's X axis
T quad_swap_vertical  (T v)                  // ... Y axis
T quad_swap_diagonal  (T v)                  // ... diagonal
```

Clustered, shuffle, and quad ops each infer their own `GroupNonUniform*` capability
(`Clustered`, `Shuffle`, `ShuffleRelative`, `Quad`) from use - you never request them.

### 7.4 HLSL `Wave*` / `Quad*` aliases

Accepted, normalized to the native names (no diagnostic):

| HLSL | Native |
|------|--------|
| `WaveGetLaneCount` / `WaveGetLaneIndex` | `subgroup_size` / `subgroup_lane_id` |
| `WaveIsFirstLane` | `subgroup_elect` |
| `WaveActiveAnyTrue` / `WaveActiveAllTrue` / `WaveActiveAllEqual` | `subgroup_any` / `subgroup_all` / `subgroup_all_equal` |
| `WaveActiveBallot` | `subgroup_ballot` |
| `WaveActiveSum` / `Product` / `Min` / `Max` | `subgroup_add` / `mul` / `min` / `max` |
| `WaveActiveBitAnd` / `BitOr` / `BitXor` | `subgroup_and` / `or` / `xor` |
| `WavePrefixSum` / `WavePrefixProduct` | `subgroup_exclusive_add` / `exclusive_mul` |
| `WaveActiveCountBits` / `WavePrefixCountBits` | `subgroup_ballot_bit_count` / `..._exclusive_bit_count` |
| `WaveReadLaneAt` / `WaveReadLaneFirst` | `subgroup_broadcast` / `subgroup_broadcast_first` |
| `QuadReadAcrossX` / `Y` / `Diagonal` | `quad_swap_horizontal` / `vertical` / `diagonal` |
| `QuadReadLaneAt` | `quad_broadcast` |

> **One deliberate divergence:** HLSL and DXC define `WavePrefixSum` as an *exclusive* scan,
> but glslang emits an *inclusive* one. SVSL follows the HLSL semantics (exclusive) - so its
> output for `WavePrefixSum` is correct where glslang's is wrong. See [HLSL
> Compatibility](09-hlsl-compatibility.md#4-deliberate-divergences-from-glslang).

---

## 8. Atomics

Atomic read-modify-write on a `storagebuffer` member, a `workgroup`/`groupshared` variable,
or an integer storage image. The **native** functions take the destination by reference and
**return the prior value**:

```c
T atomic_add(ref T dest, T v)   atomic_sub   atomic_min   atomic_max
  atomic_and   atomic_or   atomic_xor   atomic_exchange
T atomic_compare_exchange(ref T dest, T compare, T v)
```

```c
RWStructuredBuffer<float> accum;
groupshared float gs_peak;

atomic_add(accum[slot], v);
atomic_max(gs_peak, v);
float prior = atomic_exchange(accum[63], v);
```

Integer atomics are the portable set; `atomic_add`/`min`/`max`/`exchange` on **float** are
supported too (SPV_EXT_shader_atomic_float, native SVSL - glslang has no HLSL spelling).

### Scope and memory order

Two things the atomic figures out or lets you say - both places SVSL goes past glslang:

- **Scope is inferred from where the destination lives.** An atomic on a `groupshared`
  variable synchronizes only within the workgroup, so it emits **Workgroup** scope; a
  buffer or image atomic is device-visible and emits **Device** scope. glslang hardcodes
  Device for everything - correct, but over-synchronizing on shared memory (which matters
  most on tiled/mobile GPUs). There's no syntax for it; it just does the tighter thing.

- **Memory order** is an optional trailing argument on the **native** forms -
  `relaxed` (default), `acquire`, `release`, `acq_rel`:

  ```c
  uint taken = atomic_exchange(lock, 1u, acquire);   // acquire a lock
  atomic_exchange(lock, 0u, release);                // release it
  atomic_add(counter, 1u);                           // relaxed - the default
  ```

  Relaxed emits no memory-semantics bits, so it stays **bit-identical to glslang**. A stated
  order emits its ordering bits combined with the destination's storage-class memory bit
  (Workgroup / Uniform / Image). `seq_cst` is rejected with a diagnostic - the Vulkan memory
  model has no sequential consistency, so `acq_rel` is the strongest you can ask for. Order
  is native-only; the `Interlocked*` aliases keep their HLSL out-parameter form, so nothing
  about existing HLSL code changes.

The HLSL `Interlocked*` forms are accepted aliases. They use an **out-parameter** for the
prior value instead of a return:

```c
InterlockedAdd(result[0], value);            // discard prior
InterlockedAdd(result[6], 1u, orig);         // orig receives the prior value
InterlockedExchange(result[11], 4242u, prev);
InterlockedCompareExchange(result[13], 42u, 777u, prevHit);
```

`InterlockedCompareStore` is also supported (it maps to compare-exchange) - glslang silently
compiles it to nothing, another place SVSL is more correct.

---

## 9. Barriers

```c
workgroup_barrier();            // execution + workgroup-memory sync
workgroup_memory_barrier();     // workgroup memory only, no execution sync
device_memory_barrier();        // device memory only
all_memory_barrier();           // all memory, no execution sync
device_memory_barrier_sync();   // execution + device memory
all_memory_barrier_sync();      // execution + all memory
```

HLSL aliases (accepted, normalized): `GroupMemoryBarrierWithGroupSync` → `workgroup_barrier`,
`GroupMemoryBarrier` → `workgroup_memory_barrier`, `DeviceMemoryBarrier` →
`device_memory_barrier`, `AllMemoryBarrier` → `all_memory_barrier`,
`DeviceMemoryBarrierWithGroupSync` → `device_memory_barrier_sync`,
`AllMemoryBarrierWithGroupSync` → `all_memory_barrier_sync`.

---

## 10. Rejected: legacy D3D9 intrinsics

`lit`, `dst`, and `msad4` are recognized *only to reject them clearly* - they have no SPIR-V
equivalent:

```
'lit' is a legacy D3D intrinsic with no SPIR-V equivalent; compute the terms directly
```

Compute the terms by hand instead. This is a hard error, not a porting hint.

---

Next: [Advanced Features](07-advanced-features.md) - specialization constants, the early-Z
toolkit, and capability inference.
