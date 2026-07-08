# SVSL IR Optimization Plan

Status: **design draft**. Extends the Optimization decision in docs/DECISIONS.md (was: fixed
inline/chains/fold/dce only). Goal: a small, data-oriented optimizer that shrinks emitted
SPIR-V without linking SPIRV-Tools and without breaking the bit-exact correctness oracle.

---

## 1. The one constraint that shapes everything

The correctness bar (docs/DECISIONS.md) is **output equality with skshaderc**: pixels < 0.01 avg
error, compute buffers *bit-identical*. So an "optimization" is only legal if it produces a
program whose runtime results match the reference. This splits candidate passes into two
classes:

- **Value-preserving passes** — they reuse an already-computed value or delete
  computation that provably has no observers. The numeric result of the program is
  *unchanged bit-for-bit*. Safe unconditionally. (store-to-load forwarding, copy prop,
  CSE, dead-store elim, extract-of-construct, redundant-load elim, better DCE.)
  **These are the whole high-value core of the plan.**

- **Value-changing algebraic passes** — `x*1.0 → x`, `x+0.0 → x`, `a*b+c → fma`,
  reassociation. Under strict IEEE these change bit patterns (`-0.0`, NaN, Inf, rounding),
  so they are only legal where they match what glslang+spirv-opt already did to the
  reference, or behind an explicit opt-in `--ffast-math` that the oracle is *not* run
  against. Integer algebraic identities (`x*1`, `x+0`, `x&x`) are exact and always safe.

Design rule: **default optimization level ships only value-preserving passes + integer
identities.** Everything float-algebraic is opt-in and excluded from the pixel/bit oracle.
This keeps the "match the reference" promise intact while still cutting the local-variable
traffic that full inlining generates.

---

## 2. Why there's low-hanging fruit

Mandatory full inlining (`src/ir/ir_build.c`) + `out/inout` SSA copies + variables-stay-as-
var/load/store (no phi construction) means the IR after lowering is thick with:

- `store %v, %x` immediately followed by `%y = load %v`, no intervening store — every
  inlined call boundary and every local assignment makes these.
- repeated identical access chains (`sk_inst[ids.inst]` used for `.world` and `.color`),
  repeated loads of the same global, repeated pure arithmetic.
- `extract (construct a,b,c,d), 2` — `float4(pos.xyz,1).xyz` style, from constructor +
  swizzle chains.

glslang's output is compact precisely because spirv-opt runs local-SSA, DCE, and copy-prop
over exactly this shape. We can capture most of that win with a handful of local passes,
because our forward-only-reference invariant makes them nearly trivial.

---

## 3. Architecture

Keep the "fixed pipeline, no pass manager" philosophy. Three small additions:

### 3.1 Passes return `bool changed`

Change the signatures so a driver can iterate to fixpoint:

```c
bool svsl_ir_fold      (svsl_ir_func_t *fn, const svsl_types_t *types);
bool svsl_ir_dce       (svsl_arena_t *arena, svsl_ir_func_t *fn);
bool svsl_ir_forward   (svsl_ir_func_t *fn, const svsl_types_t *types); // store→load, load-CSE, copy-prop
bool svsl_ir_cse       (svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_types_t *types);
bool svsl_ir_peephole  (svsl_ir_func_t *fn, const svsl_types_t *types); // structural identities
```

### 3.2 A tiny fixed driver (not a pass manager)

```c
// src/ir/passes/optimize.c
void svsl_ir_optimize(svsl_arena_t *arena, svsl_ir_func_t *fn,
                      const svsl_types_t *types, svsl_opt_level_ level) {
	if (level == svsl_opt_none) return;
	for (int32_t iter = 0; iter < SVSL_OPT_MAX_ITERS; iter++) {   // bounded, e.g. 8
		bool changed = false;
		changed |= svsl_ir_fold    (fn, types);
		changed |= svsl_ir_peephole(fn, types);
		changed |= svsl_ir_cse     (arena, fn, types); // before forward — see §5b ordering
		changed |= svsl_ir_forward (fn, types);
		changed |= svsl_ir_dse     (fn);
		if (!changed) break;
	}
	svsl_ir_dce(arena, fn);   // single final sweep
}
```

`svsl_ir_build` calls `svsl_ir_optimize` where it currently calls fold+dce. The order is a
fixed list of function calls — the same design as today, just longer and iterated. No
registry, no metadata, no dynamic scheduling.

### 3.3 Optional compaction pass (dump-only quality-of-life)

Nops are free at emit time (skipped), so compaction is not required for correctness. But a
`svsl_ir_compact` that renumbers to drop nops makes `--dump-ir` readable and golden tests
stable. Run it *only* for the dump path, never before emit (emit relies on stable indices).
Defer unless the IR dumps get noisy.

### 3.4 Invariants every new pass must uphold

1. **Forward-only references stay forward-only.** Never make an instruction reference a
   higher index. CSE/copy-prop redirect users to an *earlier* definition — always legal.
2. **Rewrite in place or nop out.** Never shift indices (except the dedicated compaction
   pass). Replacing an op means overwriting its fields; killing it means `op = nop`.
3. **Respect structured control flow.** `forward`/CSE must not move a value across a
   block boundary, a store, a barrier, an atomic, or an image op unless proven safe. The
   simplest safe scope is *within a straight-line run between CF markers* — start there.
4. **Opaque values never appear** (textures/samplers) — already guaranteed, but CSE must
   skip resource ops' operand identity carefully (`svsl_ir_tex` args mix res indices and
   value ids).

### 3.5 Options plumbing (honesty rule)

Add `svsl_opt_level_` to `svsl_options_t`/`svsl_sema_options_t` and a CLI flag. Honor it or
reject it — never silently ignore (CLAUDE.md ground rules).

```
-O0            no optimization (lowering + final DCE only — DCE is needed for valid CF)
-O1  (default) all value-preserving passes + integer identities   ← oracle runs here
-O2            adds float-algebraic identities that match glslang's known output
--ffast-math   aggressive float algebra; NOT covered by the pixel/bit oracle
```

Record the level in the IR dump header so golden tests pin it.

---

## 4. The optimization menu

Ordered by (value ÷ cost). ✅ = value-preserving (safe, oracle-covered). ⚠️ = value-changing
(opt-in, excluded from oracle).

### Tier 1 — do these first (value-preserving, high yield)

| # | Pass | What it does | Why it pays |
|---|---|---|---|
| 1 | ✅ **Store-to-load forwarding** | Within a straight-line run, `store %p,%x` then `load %p` with no intervening store/barrier/aliasing op → replace the load's users with `%x`. | The dominant win. Every inlined call and `o.field = …; … = o.field` collapses. Feeds DCE to delete the variable entirely. |
| 2 | ✅ **Redundant-load elimination** | Two `load %p` with no store between → second reuses the first. | Repeated reads of the same global/cbuffer member. |
| 3 | ✅ **Dead-store elimination** | `store %p,%x` with no later load of `%p` (or fully overwritten first) → nop. Requires DCE to stop pinning *all* stores as roots; pin only stores to output/global/aliasable memory. | Kills the write half once forwarding removed the readers. |
| 4 | ✅ **Local CSE (value numbering)** | Hash pure ops by (op, type, operands); a later identical op reuses the earlier result. Forward-only refs make a single forward pass exact within a block. | Repeated chains (`sk_inst[i]`), repeated arithmetic, duplicated swizzles. |
| 5 | ✅ **extract/insert-of-construct** | `extract (construct c0..cn), k → ck`; `shuffle` of a `construct` → pick components directly. | `float4(pos.xyz,1).xyz`, matrix-cast + component access — pervasive in the corpus. |
| 6 | ✅ **Better intrinsic DCE** | Use the existing `svsl_intr_tag_` table to mark pure intrinsics (trig, min/max, dot…) DCE-able; keep only barriers/atomics/derivative-with-effects pinned. | Today *all* intrinsic results are pinned live — dead pure-math intrinsics survive. |

### Tier 2 — solid follow-ups

| # | Pass | What it does | Class |
|---|---|---|---|
| 7 | ✅ **Vector/matrix constant folding** | Extend fold to `construct` of all-constant components → one constant composite; `extract`/`shuffle` of a constant composite → constant. Emit as `OpConstantComposite`. | value-preserving |
| 8 | ✅ **Constant intrinsic folding** | Fold `min/max/abs/clamp/saturate/floor/ceil/…` on constant args, matching IEEE exactly (same care fold.c already takes: skip float16/float64/frem). | value-preserving |
| 9 | ✅ **Swizzle/shuffle composition** | Collapse chained `shuffle`/`extract` into one; identity swizzle (`v.xyzw` on a 4-vec) → `v`. | value-preserving |
| 10 | ✅ **Access-chain CSE** | Fold identical chains to one pointer (special-case of #4 for `svsl_ir_chain`). | value-preserving |
| 11 | ✅ **Boolean/compare folding** | Fold `select(const_cond,a,b)`, `and/or/not` of constants, comparisons of constants. | value-preserving |
| 12 | ⚠️ **Integer algebraic identities** | `x+0, x-0, x*1, x*0→0, x&x, x|x, x^0, x<<0`. Exact for integers → actually safe, promote to Tier 1 once #1–6 land. | safe (int) |

### Tier 3 — value-changing, opt-in only (`-O2` / `--ffast-math`)

| # | Pass | Note |
|---|---|---|
| 13 | ⚠️ **Float algebraic identities** (`x*1.0`, `x+0.0`, `x*0.0`, `x/1.0`) | Changes `-0.0`/NaN/Inf behavior. Only under `-O2` if it matches glslang's emitted form; else `--ffast-math`. |
| 14 | ⚠️ **FMA contraction** (`a*b+c → OpExtInst Fma`) | Different rounding than separate mul+add. Opt-in. |
| 15 | ⚠️ **Reassociation / distribution** | Not IEEE-associative. Opt-in, low priority. |

### Tier 4 — emit-level peepholes (backlog §10 already lists 16–17)

| # | Peephole | Note |
|---|---|---|
| 16 | ✅ **DONE** `splat + OpFMul` → **OpVectorTimesScalar** | Pure encoding change, bit-exact. −2008 words. |
| 17 | ✅ **DONE** dynamic vector index → **OpVectorExtractDynamic** | `extract_dynamic` IR op; correct + tested, but ~0 corpus impact (real dynamic indexing is into arrays). InsertDynamic not implemented (no reader/writer pattern seen). |
| 18 | ✅ **DONE** `matrix * scalar` → **OpMatrixTimesScalar** | Already emitted from the direct `mul(mat, scalar)` form; no splat exists to strip. |
| 19 | ✅ **Selection/loop control masks** | Statement hints (`[unroll]`/`[branch]`/`[flatten]`) → SPIR-V loop/selection control operands (currently advisory-only). Only if a case shows it matters. |

### Explicitly deferred (leave to external spirv-opt)

Loop-invariant code motion, loop unrolling, full mem2reg with phi construction, general
CFG simplification / branch elimination across merges, function-level inlining heuristics
(we always fully inline), register allocation. These need a real CFG + phi and buy little
for shader-sized code — the decision-log stance holds: pipe through `spirv-opt` if someone
wants them. Note this in the plan, don't build it.

---

## 5. Testing & verification (non-negotiable)

Every pass lands with:

1. **IR golden test** (`test_ir.c`, `--dump-ir`) — a focused before/after shader showing
   the transform, pinned at its `-O` level.
2. **Corpus + spirv-val** — full `tests/shaders/` corpus still compiles and validates
   (Vulkan 1.1 env) at every `-O` level.
3. **The oracle** — `svsl_view -test tests/shaders` must stay pixel-match, and compute
   shaders stay *bit-identical*, at `-O0` **and** `-O1`. A pass that diverges the oracle at
   `-O1` is by definition a bug (it changed a value). `-O2`/`--ffast-math` are exempt and
   run under a separate, looser threshold if run at all.
4. **Metric** — op-count stats already flow into `.sks` (`total`, `tex_read`,
   `dynamic_flow` per stage, in the `.sks` reflection). Diff these before/after per pass to quantify
   the win; add a `--dump-ir`-based instruction count to the test summary.
5. **Float-safety review** — any fold path must match the care already in `fold.c` (no
   float16/float64 folding, no frem, clamp/NaN handling on float→int). New value-preserving
   passes need none of this (they don't compute), which is exactly why they're the core.

---

## 5b. Performance log

Tracked by `tests/svsl_bench` (best-of-N over the 131-shader corpus; Debug lib
build on a busy laptop, so timings are **rough ballparks for relative
cost/benefit**, not absolute throughput — the instruction/word counts, being
deterministic, are exact). Optimizer cost is isolated as `ir(O1) − ir(O0)` since
lowering is identical at both levels. Payoff is live IR instructions and emitted
SPIR-V words at -O1 (both exact).

| Landed | Optimizer cost / corpus (approx) | live insts (O0→O1) | SPIR-V words (O0→O1) |
|---|---|---|---|
| baseline (fold + dce) | ~0.7 ms | 37201 → 36323 (−2.4%) | 298017 → 295360 (−0.9%) |
| **#1/#2/#3 forward + dead-store** | ~3 ms | 37201 → 28883 (−22.4%) | 298017 → 259719 (−12.8%) |
| **#4/#10 local CSE** | ~5 ms | 37201 → 24689 (−33.6%) | 298017 → 255193 (−14.4%) |
| **#5/#9/#12 peephole** | ~6 ms | 37201 → 24649 (−33.7%) | 298017 → 254952 (−14.5%) |
| **#6 intrinsic-purity DCE + CSE** | ~6 ms | 37201 → 24549 (−34.0%) | 298017 → 254440 (−14.6%) |
| **per-stage dead-global elim** (emit) | free (emit ~faster) | 24549 (unchanged) | 254440 → 222903 (−25.2% vs raw) |
| **member-forward SROA** (construct→store→member-load) | negligible | 24549 → 24271 | 222903 → 221552 |
| **entry-input SROA** (emit) | free | 24271 (unchanged) | 221552 → 207683 |
| **object-aware forward** (redundant param/uniform loads + same-ptr store→load) | ~+1 ms IR | 24271 → 23160 | 207683 → 203546 |
| **output-struct SROA** (emit) | free | 23160 (unchanged) | 203546 → 197332 |
| **dominance forward** (durable entries cross CF markers) | ~flat (one pre-scan) | 23160 → 22471 | 197332 → 194035 |
| **vector×scalar** (#16, strip splat → OpVectorTimesScalar) | negligible | 22471 → 22140 | 194035 → 192027 |

The dead-global row is an **emit-level** fix, not an optimizer pass, so it helps
every `-O` level (O0 SPIR-V words 298017 → 266518 too) and doesn't move
`live_insts` (dead globals are declarations, not IR instructions). Found by
diffing disassembly (below): a fragment shader was declaring the entire system
cbuffer / instance buffer / `$Global` even when it only returned its color input.
`create_globals` now emits only the buffers/resources the stage's IR references
(`svsl_ir_func_globals` in `usage.c`).

### vs skshaderc (glslang + full spirv-opt)

Comparing SPIR-V sizes on the 37 corpus shaders both compilers accept
(`scratchpad/cmp_size.py` — compiles each with both, sums SPIR-V from the `.sks`):

| metric | svslc -O1 | skshaderc | ratio | (before this turn) |
|---|---|---|---|---|
| total words (setup + body) | 73266 | 70964 | **1.03×** | was 1.12× |
| body instructions (from first `OpFunction`) | 6643 | 5216 | **1.27×** | was 1.50× |

(This 37-shader set is mostly straight-line, so cross-CF forwarding barely moves it;
its bigger effect is on the branchy compute/fragment shaders that dominate the full
corpus — see the −309 total loads below.)

**We are now at or below skshaderc on 25 of 35 corpus shaders** (build items 13–15 below
took this from 18; the snapshot table above predates them). Originally 18 (total words, 0.93–0.98):
`builtin_default`, the whole example lighting set (`54_diffuse`…`59_pulse`),
`68_matcap`, `53_vertexcolor`, and more. Not from a better *body* — ours is still
1.27× — but from a leaner *setup*: dead-global elimination emits only referenced
buffers/resources, and input/output SROA declares only the interface members used
(never a whole `psIn`/`vsIn` type + decorations). glslang always declares the full
cbuffer blocks and interface structs, so on shaders where the preamble dominates the
total, our setup savings outweigh the body overhead. On body-heavy shaders it flips
(`pbr` 1.07, `granite` 1.48) — instruction-level body work is where the gap remains.

Diffing `53_vertexcolor` disassembly (the worst offender at 2.1×) showed the
delta was **two separate things**, and the bigger one was *not* in the optimizer:

1. **Dead globals (setup)** — the fragment shader (`return input.color`) declared
   the whole system cbuffer, instance buffer and `$Global` with all their types and
   decorations. Fixed by per-stage dead-global elim: PS dropped 541 → 182 words,
   corpus total 1.37× → **1.21×**. Setup is now close to skshaderc.
2. **Body (SROA)** — *fixed.* The fragment used to load *all three* stage inputs
   (`pos`/FragCoord, `color`, `layer`), `OpCompositeConstruct` the whole `psIn`
   into a Function var, then `OpAccessChain`+`OpLoad` member 1. `analyze_param_sroa`
   now detects that a struct stage parameter is only read via constant member
   chains and skips the shadow var + construct: each member chain resolves
   straight to that member's input variable, and only referenced members are
   declared (location counter still advances for skipped members so the used ones
   keep the locations the other stage expects). The `vertexcolor` PS body is now
   `load color; store output; return` — identical to skshaderc, 95 words.

Three more wins landed and closed most of the remaining gap:

3. **Redundant loads (object-aware forward)** — the old forward pass tracked only
   whole-var pointers. It now tracks loads/stores by exact pointer for every
   *forwardable root* (local vars, params, and read-only globals: uniform/push-
   constant buffers, const globals, read-only structured buffers), with a per-root
   generation bumped on stores so only same-object entries invalidate. Repeated
   `load param` (the instance-id), duplicate `ptr`+`load` of a cbuffer member, and
   `store %chain; load %chain` all collapse. Writable globals are left untouched
   (sound without pointer canonicalization). PBR VS loads 149→44, ptrs 23→10.
4. **Output-struct SROA (emit)** — the mirror of entry-input SROA. A return-struct
   local var that is only written through constant member chains and loaded whole
   into the return is elided: each member store writes straight to its stage-output
   variable, so the whole `psIn` is never `OpCompositeConstruct`ed into a Function
   var and torn apart again. `analyze_output_sroa` + the chain/return hooks in
   `emit_spirv.c`. The default VS now declares no `_ptr_Function_psIn` at all.
5. **Dominance forwarding** — forwarding used to flush the whole table at every CF
   marker, so a value set before a branch and read inside was never threaded. It
   now marks entries *durable* (surviving markers) when their root is never stored
   in a conditional region AND the entry was established at depth 0 — the two
   conditions that make the cached value provably dominate every later use with no
   phi. Single-assignment locals and read-only globals now forward into branch and
   loop bodies. A pre-scan flags conditionally-stored roots; everything else flushes
   as before. Corpus loads 3386 → 3077 (145 of them across CF boundaries); words
   197332 → 194035. Sound direction verified by `test_ir_cross_cf_forward` (a
   branch-reassigned local is *not* forwarded past the merge) plus the pixel oracle
   on every branchy renderable shader and `spirv-val` on all 104 at -O0/-O1/-O2.

**Local member promotion + shuffle composition + multi-index input SROA (build items
13–15)** then cut the body gap from 1.27× to **1.16×** and pushed **25 of 35** shaders
to or below skshaderc's size (was 18). The signature they attack — a local vector/struct
written once and read component-wise via `store %v; OpAccessChain %v[k]; OpLoad`, plus the
`.rgb`/swizzle shuffles glslang extracts straight through — was the dominant residual in
every near-parity shader. Item 13 turns each such member load into one `OpCompositeExtract`
off the value (no spill); item 14 folds `extract(shuffle)`/`shuffle(shuffle)` onto the
source; item 15 lets `input.member.comp` reads keep their struct param scalar-replaced.
`47_stereo_display`'s PS body is now byte-identical to skshaderc, and `70_granite`'s body
overhead fell from +380 to +56 instructions.

**Overwriting-store DSE + a dead-declaration gate (build items 16–17)** then cleaned up the
last emit-level residue: a store shadowed by a later store to the same pointer (the
`o.color = a; o.color.rgb *= b` double-write), and constants/pointers left orphaned when SROA
resolves a member chain to an interface variable or a chain re-inlines a buffer-member pointer
(**118 dead scalar constants → 0** corpus-wide). With these, `sk/default_shader` is solved —
**both stages' bodies are byte-identical to skshaderc** and smaller in total.

Remaining body headroom (now **1.15×**), rough impact order: small constant-trip loop
unrolling (glslang unrolls PCF/sample loops — `46_shadow_receiver`, `49_text`, `70_granite`
still carry `OpLoopMerge`/`OpPhi` where skshaderc has none) and block/branch merging.
Dominance forwarding is deliberately conservative — it gives up on conditionally-stored
locals (would need phi/merge tracking); that tail is smaller and much riskier than what
landed.

Notes on cost/benefit: forwarding+DSE and CSE carry the win (−33% live IR, −14%
SPIR-V). Peephole adds little on *this* corpus — it's float-math heavy, so the
integer identities and extract-of-construct rarely fire — but it is cheap and
folds constant-indexed vectors (`float4(1,2,3,4)[2]` → `3`) that integer/struct-
heavy shaders hit more. `-O2` adds float `x*1.0`/`x/1.0` (24637 insts / 254895
words); it barely moves here and is deliberately excluded from the oracle.

**Oracle:** at every stage above, `svsl_view -test tests/shaders` reports all 91
renderable shaders pixel-identical to skshaderc (avg error 0.00000) at -O1, and
the full corpus still passes `spirv-val`. The value-preserving passes are
confirmed bit-exact against the reference compiler.

### Profiling / algorithmic health

`perf record` on `svsl_bench -O1` (cpu-clock sampling): the whole optimizer is
~14% of compile time (front-end dominates); no pass is pathological — CSE is the
heaviest (~4.5%, mostly `remap_operands` + hashing), the rest ≤2.5% each. Every
pass is a linear O(n) sweep; there is no O(n²) in shader size (generation-stamped
tables flush in O(1); chain roots are pre-flattened). The `SVSL_OPT_MAX_ITERS=8`
cap is never approached, so no pass oscillates. The cost pattern to keep watching
as passes are added: each pass adds one O(n) sweep per fixpoint iteration, so cost
grows linearly in pass-count — fine today, worth revisiting only if the list gets long.

**Pass ordering (measured).** The passes mutually feed each other, so the *order
within an iteration* changes how fast the fixpoint is reached — not the output,
which is identical at any order that converges (verified: every ordering below
produced byte-identical SPIR-V, 197332 words). Instrumenting the driver to count
iterations-to-fixpoint across the 221-function corpus:

| order (within iteration) | total iters | forward fires | note |
|---|---|---|---|
| fold, peephole, forward, dse, **cse** (naive) | 572 | 324 | cse merges ptrs *after* forwarding → forward waits a whole iteration to use them |
| forward, dse, fold, peephole, cse | 569 | 324 | forward-first doesn't help; cse still trails |
| **cse**, fold, peephole, forward, dse | 541 | 218 | cse-first, but constants from fold are deduped a cycle late |
| fold, peephole, **cse**, forward, dse ✅ | **518** | **218** | **adopted** — fold/peephole feed cse, cse feeds forward |

The lever is **cse before forward**: once CSE collapses duplicate address
computation (`ptr`/`chain`) to one id, forwarding dedups those loads in the same
sweep instead of the next iteration — cutting total iterations ~9% and forwarding
sweeps by a third. `fold`→`peephole`→`cse` keeps the simplify-then-dedup order so
constants and canonical forms are deduped the cycle they appear. `dse` stays last
(it needs forwarding to have removed the readers first). Reproduce with the
temporary `-stats` path in `bench_main.c` (add a per-pass fire counter to
`optimize.c`); the ordering rationale is documented in that file's header.

Method to reproduce: `cmake --build build --target svsl_bench && ./build/tests/svsl_bench -n 150`.
The front-end (pp/lex/parse/sema, ~36 ms) dominates total compile time; the
whole optimizer is a few ms on top, so cost is judged against emit+IR, not the
front-end.

## 6. Suggested build order

1. ✅ **Done.** Signature change to `bool changed` + `svsl_ir_optimize` driver + `-O` flag.
   Behavior-neutral wrapper over the old fold+dce. (`passes/optimize.c`, `ir_operands.c`.)
2. ✅ **Done.** #1 store-to-load forwarding + #2 redundant-load + #3 dead-store.
   (`passes/mem.c`: `svsl_ir_forward`, `svsl_ir_dse`.) The biggest win.
3. ✅ **Done.** #6 intrinsic-purity DCE + CSE. No table column needed: a barrier is the
   only side-effecting intrinsic and it is the only one returning **void**, so
   `svsl_ir_intrinsic_is_pure` = "result is non-void". Dead pure intrinsics now DCE; duplicate
   ones CSE (barriers guarded out). (`ir_operands.c`, `passes/cse.c`, `passes/dce.c`.)
4. ✅ **Done.** #4 local CSE + #10 access-chain CSE (chains are in the pure set).
   (`passes/cse.c`.) #2 redundant-load landed with forwarding.
5. ✅ **Done.** #5 extract-of-construct + #9 shuffle-identity + #12 integer identities
   (in `-O1`) + a first `-O2` float slice (#13 `x*1.0`/`x/1.0`). (`passes/peephole.c`.)
6. ⬜ **#7 vector const fold** + **#8 constant intrinsic fold** + **#11 boolean fold** —
   extend `fold.c` to composites and (with the purity column) pure intrinsics.
7. ✅ **Done (emit-level).** Per-stage dead-global elimination + member-forward SROA in the
   forwarding pass. Dead-global cut corpus total words 1.37× → 1.21× vs skshaderc.
8. ✅ **Done (emit-level).** Entry-input SROA: a struct stage parameter read only via
   constant member chains skips the shadow-var/construct entirely; each `chain(param,[k])`
   resolves to member `k`'s input variable and only referenced members are declared
   (`analyze_param_sroa` + the prologue/chain hooks in `emit_spirv.c`). Cut corpus total
   1.21× → **1.12×** vs skshaderc; `vertexcolor` PS body now byte-for-byte skshaderc's.
9. ✅ **Done.** Object-aware forwarding: track loads/stores by exact pointer for every
   forwardable root (local vars, params, read-only globals), per-root generation
   invalidation. Collapses redundant param/uniform loads and same-pointer chain
   store→load. (`passes/mem.c`: `svsl_ir_forward` now takes `prog`.) Cut O1 words
   207683 → 203546.
10. ✅ **Done (emit-level).** Output-struct SROA: a return-struct local var written only
   through constant member chains and returned whole is elided — members store straight
   to the stage-output variables (`analyze_output_sroa` + chain/return hooks in
   `emit_spirv.c`). Cut O1 words 203546 → **197332**; corpus total 1.12× → **1.05×**,
   body 1.50× → **1.31×** vs skshaderc.
11. ✅ **Done.** Dominance forwarding: entries whose root is never conditionally stored and
   which were established at depth 0 are *durable* and survive CF markers, so single-
   assignment locals and read-only globals forward into branch/loop bodies. A pre-scan
   flags conditional roots; no phi/merge machinery. (`passes/mem.c`: `depth_delta`, the
   `cond`/`e_dur` tables; `test_ir_cross_cf_forward`.) Cut O1 words 197332 → **194035**,
   corpus loads 3386 → 3077. Body 1.31× → **1.29×** (bigger on branchy shaders).
12. ✅ **#16/#17/#18 done.** Emit-level opcode selection for the multiply family:
   - **#16 `OpVectorTimesScalar`** — a peephole strips the `mul(vec, splat)` splat to a
     scalar operand (`strip_vector_splat` in `passes/peephole.c`; DCE reclaims the dead
     construct), emit selects `OpVectorTimesScalar` (`emit_intrinsics.inc`). Bit-exact. Cut
     O1 words 194035 → **192027**, body 1.29× → **1.27×**; `test_ir_vector_times_scalar`.
   - **#18 `OpMatrixTimesScalar`** — already handled: the front-end emits `mul(mat, scalar)`
     directly (no splat), and emit lowered it to `OpMatrixTimesScalar` from the start. No
     splat form exists to strip, so nothing to do.
   - **#17 `OpVectorExtractDynamic`** — a dynamic index into a *non-addressable vector value*
     used to spill the value to a Function variable + access chain; the forward pass now
     rewrites `load(chain(var,[dyn]))` on a live vector to a new `extract_dynamic` IR op, so
     the spill dies (DCE) and emit uses `OpVectorExtractDynamic`. (`ir.h`, `passes/mem.c`,
     `passes/cse.c`, `ir_operands.c`, `ir_dump.c`, `emit_intrinsics.inc`;
     `test_ir_rvalue_index` + renderable `check_dynamic_index`.) **Correct and bit-exact but
     ~0 corpus impact**: real shaders dynamically index *arrays* (which correctly still use
     access chains — `OpVectorExtractDynamic` is vector-only), not vector rvalues, so the
     pattern doesn't occur in the current corpus. Kept because it's complete, tested, and
     matches glslang for the pattern when it appears.
13. ✅ **Done.** Local member promotion: the forward pass now rewrites
   `load(chain(var,[const_k]))` on any single-assignment local whose current value is a
   live composite (vector/struct/matrix/array) — not just an `OpConstruct` — into a static
   `svsl_ir_extract value, k`. The spill var/store/access-chain/load collapse to one
   `OpCompositeExtract`; DSE+DCE reclaim the rest, matching skshaderc. (`passes/mem.c`, the
   general twin of #17's dynamic path.)
14. ✅ **Done.** Shuffle composition (peephole): `extract(shuffle v, k)` folds to
   `extract(v, lane[k])` and `shuffle(shuffle v, …)` to a single shuffle over the source; the
   inner shuffle then dies (DCE). (`passes/peephole.c`: `compose_shuffle`.) Removes the
   redundant `.rgb`/swizzle chains glslang extracts through directly.
15. ✅ **Done (emit-level).** Multi-index input SROA: a struct stage parameter read via
   `input.member.comp` (a 2+-index member chain) no longer disqualifies the whole param —
   `sroa_member_chain` accepts trailing indices and emit resolves the chain to an
   `OpAccessChain` into the member's stage-input variable. (`emit_spirv.c`,
   `emit_intrinsics.inc`.) Un-SROA'd `psIn`/`vsIn` Function vars dropped from 8 files to 5.

   Combined effect (13–15): O1 words 192394 → **182698** (−5.0%), live insts 22174 → 20478
   (−7.6%); body **1.27× → 1.16×** vs skshaderc; shaders at/below parity **18 → 25 of 35**.
   `stereo_display` PS body is now byte-identical to skshaderc; `70_granite` body +380 → +56
   insts. 92/92 pixel oracle bit-exact; `spirv-val` clean at -O0/-O1/-O2; the swizzle-store
   golden `test_ir_swizzle_stores` updated (the `.bgr` read now folds into the inserts).
16. ✅ **Done.** Overwriting-store DSE: a store to an exact local pointer whose value a later
   store to the *same* pointer shadows — with no load of that pointer's root and no CF
   boundary between — is dead. The never-read pass can't see it (the root is read whole by
   `return o`, so it keeps every member store), but `o.color = a; o.color.rgb *= b;` writes
   `o.color` twice with the first unobserved. Exact-pointer-safe because CSE has already
   merged duplicate addresses; any intervening load of the root invalidates. (`passes/mem.c`:
   `dse_overwriting`.) The default VS now stores `o.color` once, matching skshaderc.
17. ✅ **Done (emit-level).** Dead-declaration gate: `analyze_emit_liveness` marks which IR
   values an *emitted* operand actually reads, mirroring the chain path — a buffer/resource
   `ptr` folded into its chain, and an SROA'd member-index constant, are referenced in the IR
   but orphaned in the output. The body loop skips emitting such orphaned `const`/`ptr`/`undef`
   leaves. (`emit_spirv.c`.) Kills the member-index constants left by input/output SROA
   (**118 dead scalar constants → 0** corpus-wide) and the whole-array access chains left when
   a chain re-inlines a buffer-member pointer.

   Combined effect (16–17): O1 words 182698 → **179826**, live insts 20478 → 20449; body
   1.16× → **1.15×** vs skshaderc; **26 of 35** shaders at/below parity. The `sk/default_shader`
   is now essentially solved — **both stages' bodies are byte-identical to skshaderc** (VS 91,
   PS 9 insts) and smaller in total (VS 250 vs 264, PS 44 vs 50). 92/92 pixel oracle bit-exact;
   `spirv-val` clean at -O0/-O1/-O2.
18. ⬜ Broaden `-O2` float algebra (#14–15) behind the oracle exemption, only if wanted.
19. ⬜ (Maybe) forward conditionally-stored locals with a proper merge (phi-like) — the
   riskier tail dominance forwarding deliberately skips; only if a shader shows it matters.

Bench harness: `tests/bench_main.c` (`svsl_bench` target). Each landed step kept the
corpus green + oracle bit-exact; none introduced a pass framework — just more entries in
one fixed, iterated call list (`passes/optimize.c`).
