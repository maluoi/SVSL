# SVSL Code Review

Maturity/polish review of the whole codebase (blue-sky — nothing published, structural
changes welcome). Six subsystems were read in full; every 🔴 finding below was verified
against the source (many reproduced), not taken on faith from a scan.

**Severity:** 🔴 correctness/bug · 🟠 maintainability/design · 🟡 comment or doc drift · 🔵 nit.
**Verified:** ✓ = reproduced or confirmed by reading the exact path · ~ = reported from the
quoted code, not independently reproduced (flagged for a confirming test).

**Build health (baseline):** clean under `-Wall -Wextra -Wshadow`; all 10 ctest suites pass;
corpus compiles and `spirv-val`s. None of the findings below surface as compiler warnings —
they need targeted tests or adversarial input.

---

## Top priority — correctness

These change emitted results or crash the compiler. Ordered by impact.

### 🔴✓ 1. Early `return` nested in a loop/switch inside an inlined helper is silently dropped
`src/ir/ir_build.c:1500` + `src/back/emit_intrinsics.inc:609`

A non-tail inlined `return` lowers to `svsl_ir_break`; `break` resolves to
`cf_innermost_breakable` — the innermost **loop or switch**. When the return sits inside a
`for`/`while`/`do`/`switch` in the callee, the break exits only that inner construct and
execution falls through to the rest of the callee, whose tail return overwrites `result_var`.

Reproduced (`--dump-ir`): `float helper(float x){ for(i…) if(x>i) return 100; return -1; }`
called as `helper(2.5)` yields **-1**, not 100 — the dump shows `store %3 100` then, past
`end_loop`, `store %3 -1`. Returns nested only in `if/else` work (if-frames aren't breakable).
The corpus has no helper with an early return inside a loop, so the visual oracle never hit it.

**Fix:** distinguish a return-break from a loop/switch-break — e.g. a dedicated
return-marker op that branches to the wrapper merge past any intervening frames, or a
`returned` guard var tested after each inner construct. Add a corpus/compute check for it.

☑ **Fixed.** `ir_build.c` now allocates a `returned` bool flag for functions with a
return inside a loop/switch; the return sets it and does a normal (single-level) break,
and each such loop/switch emits `if (returned) break;` after its body to cascade the
exit outward one valid structured level at a time (glslang's scheme; a direct multi-level
break to the wrapper is invalid SPIR-V). Only functions that actually have the pattern
grow the flag — the corpus IR is unchanged. Regression test: `test_ir_return_in_loop`.

### 🔴✓ 2. Constant folding hardcodes 32-bit signed arithmetic — wrong for unsigned div/rem and all 64-bit
`src/ir/passes/fold.c:70`

The `!is_float` branch runs for every non-float scalar but narrows operands to `int32` and
does signed `/` `%`. Reproduced: `0xFFFFFFFFu / 2u` folds to **0** (`const float 0` in the
dump) instead of `2147483647`. Any 64-bit integer fold also truncates to the low 32 bits.
`add`/`sub`/`mul` survive by two's-complement luck; div/rem and 64-bit widths do not.

**Fix:** branch on the scalar's signedness and width — `uint64_t` math for unsigned, full
64-bit width, store all 64 result bits in `replace_with_const`.

☑ **Fixed.** `fold.c` now folds integer arithmetic (and `neg`) at the scalar's real width
(8/16/32/64) and signedness: unsigned ops use `uint64_t`, signed ops sign-extend and compute
in `uint64_t` (no overflow UB), `INT_MIN/-1` is guarded, and the result is masked to width.
Regression assertion added to `test_ir_passes` (`0xFFFFFFFFu / 2u` → `2147483647`).

### 🔴✓ 3. Signed-shift and `INT64_MIN / -1` UB in the sema constant evaluator
`src/sema/sema.c:232`

```c
case svsl_tok_shl: *out = a << (b & 63); return true;   // a is int64_t
case svsl_tok_slash: if (b == 0) return false; *out = a / b; return true;
```
`a << n` with negative/overflowing `a` is C11 UB (`(-1) << 2`); `INT64_MIN / -1` and
`% -1` overflow. Same "fold UB" class the fuzzer targeted, but these paths stayed signed.
**Fix:** shift on `uint64_t` (`(int64_t)((uint64_t)a << (b&63))`); guard `a==INT64_MIN && b==-1`.

☑ **Fixed.** `const_eval_int` now shifts through `uint64_t` and rejects `INT64_MIN/-1`
div/rem as non-constant. sema.c compiles UBSAN-clean.

### 🔴✓ 4. Image atomics miscompile every op except `Add`
`src/back/emit_intrinsics.inc:1432` + `src/tables/intrinsics.c:268`

Image atomics stamp `channel` (the method-table aux code: `Min=1, Max=2, And=3, Or=4,
Xor=5, Exchange=6`) straight into `signed_ops[]`/`unsigned_ops[]`, which reserve index 1 for
`ISub` (the method table has no `sub` slot). Result: `RWTexture.InterlockedMin`→`ISub`,
`Max`→`SMin`, `And`→`SMax`, … Only `Add` (index 0) is correct. Buffer atomics are fine —
they route through the enum-ordered `SVSL_INTR_ATOMIC_OP` which *does* include `sub`. **No
test exercises RWTexture atomics** (`check_atomics.hlsl` uses a buffer), so it's untested.
Float image min/max at `:1440` inherits the same off-by-one (`atomic_op==2||3` assumes buffer order).

**Fix:** renumber the method-table atomic aux codes to match `signed_ops` (Min=2…Exchange=7),
or remap at emit. Add an RWTexture-atomic compute check.

☑ **Fixed.** Method-table aux codes renumbered to the `svsl_intr_atomic_` order (Add=0,
Min=2, Max=3, And=4, Or=5, Xor=6, Exchange=7 — index 1/sub skipped), so image atomics index
`signed_ops[]`/`unsigned_ops[]` identically to buffer atomics. Verified with `spirv-dis`:
every op now emits its correct `OpAtomic*` (`SMin`/`UMin`/`SMax`/`UMax`/`And`/`Or`/`Xor`/
`Exchange`, no stray `ISub`). Regression shader `tests/shaders/checks/check_image_atomics.hlsl`
(corpus-compiled + spirv-val'd).

### 🔴✓ 5. Unbounded recursion → stack overflow on adversarial input (two sites)
`src/front/parser.c:531` and `src/front/pp.c:373`

The parser's depth guard is released before `parse_binary` recurses on the RHS of
right-associative operators, so `a=a=…=1` and `1?1:1?1:…` overflow the stack.
`PARSE_MAX_CHAIN` only bounds the left-associative iteration. The preprocessor `#if`
evaluator (`expr_primary`/`expr_binary`/`expr_ternary`) has no depth guard at all.
Reproduced: all three SIGSEGV (`a=`×80k, ternary×80k, `(`×200k). The fuzzer missed these
long-chain shapes. **Fix:** hold the depth counter across the `parse_binary` RHS recursion;
add a depth counter + diagnostic to the pp expression evaluator.

☑ **Fixed.** Parser: a `parse_binary_rhs` wrapper holds `p->depth` across every binary/ternary
RHS descent (`PARSE_MAX_DEPTH` now trips on right-associative chains). Preprocessor: added a
`depth` field to `pp_expr_t` and guarded `expr_primary` and `expr_ternary` (the two recursion
entry points), capped at `PP_EXPR_MAX_DEPTH`. All four repros (`a=…`, `1?…:`, `(…)`, `!…`) now
report `expression nesting too deep` and exit cleanly (1), no SIGSEGV.

### 🔴✓ 6. Flag-bit collision: `precise` and `legacy_spelling` both use `1 << 9`
`src/front/ast.h:181`

```c
svsl_var_flag_precise         = 1 << 9,
svsl_var_flag_legacy_spelling = 1 << 9,
```
`groupshared` and legacy `uniform`/`cbuffer` globals set `legacy_spelling`; the `precise`
reader (`ir_build.c:1399`) sees the same bit. **Latent, not currently firing:** that reader
is gated behind `if (var->init)` on the *local* decl path, and the colliding writers are
globals without initializers — so no live miscompile today. But it corrupts the moment
`legacy_spelling` gets a reader (see doc-drift #14) or a precise value flows through a shared
path. **Fix:** move `legacy_spelling` to `1 << 10`. (Related: the flag is currently dead —
see #14.)

☑ **Fixed.** `legacy_spelling` moved to `1 << 10`; `precise` keeps `1 << 9` to itself. The
flag is still unread (the keyword-form porting hints that would consume it — `cbuffer`,
`groupshared` — remain backlog, see #14), but the collision that mis-flagged those globals
as `precise` is gone.

### 🔴✓ 7. `svsl_result_header` over-reads a stack buffer on long shader names
`src/out/header_write.c:25`

`snprintf(line[128], …)` returns the *would-have-been* length; the prefix (~33 chars) plus a
95-char sanitized ident exceeds 128, so the following `for (i<n)` reads `line[128..n-1]` past
the buffer into the header. `reflect.c:22` clamps for exactly this reason; this site doesn't.
**Fix:** `if (n > (int32_t)sizeof(line)-1) n = sizeof(line)-1;`

☑ **Fixed.** `line[]` grown to 192 (fits the max 95-char ident) and `snprintf`'s return
clamped to the buffer size before the copy loop.

### 🔴✓ 8. `svslc -r` reads `kinds[]` out of bounds for tile-image resources
`src/out/reflect.c:87`

`kinds[]` has 6 entries but `svsl_res_tileimage == 6`; `kinds[res->kind]` indexes past the
array → garbage pointer to `%s`. **Fix:** add `"tileimage"` (7 entries).

☑ **Fixed.** `kinds[]` now has 7 entries (`"tileimage"` at index 6).

### 🔴✓ 9. `svsl_layout_size` silently truncates structs with >256 members
`src/sema/layout.c:106`

`uint32_t offsets[256]` with `count < 256 ? count : 256` → an oversized nested struct gets an
**undersized** size, so the following member overlaps it (silent miscompile). Only the nested
/ `svsl_layout_size` path is affected (top-level buffers size their array correctly).
**Fix:** size the scratch from the member count (arena), drop the magic `256`.

☑ **Fixed.** `svsl_layout_members`' `out_offsets` is now nullable; `svsl_layout_size` passes
`NULL` with the full member count (it only needs the total). No scratch array, no `256` cap.

### 🔴✓ 10. `//--` default overrides assume 32-bit components
`src/sema/sema.c:686`

`parse_value_list` writes every component as 4 bytes at `component*4` and picks int-vs-float
by a 32-bit-only test. A `//--flags = 5` on a `uint16`/`int64`/`float16` member writes the
wrong width and encoding and can overrun into the next member. The initializer path
(`fold_scalar`) handles widths; only the `//--` override path is wrong. **Fix:** route
through `fold_scalar` / member stride. *(Reported from the code; not reproduced.)*

☑ **Fixed.** New `write_meta_scalar` writes each token at the member's real width/encoding
(mirrors `fold_scalar`: f16 rounds, int64/f64 write 8 bytes, `half` stays 4); `parse_value_list`
strides by `svsl_scalar_size` instead of a hard-coded 4.

### 🔴✓ 11. Fixed-size composite/interface arrays truncate silently (a recurring pattern)
`ir_build.c:110` (`parts[64]`), `emit_spirv.c:1393` (OpEntryPoint interface capped at 64),
`:1310` (`output_vars[24]`), `:1282` (`parts[32]`), `sks_write.c:307,323`
(`buffer_indices[64]`, `resource_indices[128]`), `api.c:23` (`SVSL_API_MAX_STAGES 8`).

Each silently drops elements past the cap — an array/struct ≥64 elements emits an invalid
`OpConstantComposite`; a >64-variable entry point emits SPIR-V that fails `spirv-val`; extra
buffers/resources/stages vanish from the container. None hit by the corpus, but all violate
"honored or rejected, never silently ignored." **Fix:** arena-size from the real count, or
emit a diagnostic at the cap. Worth a single sweep — this is one habit, many sites.

☑ **Fixed (swept).** Every listed site is now arena-sized from the real count, no fixed cap:
`emit_zero` / array + struct `lower_init` (`ir_build.c`), the three SPIR-V struct-type emitters
and the const-composite path, `output_vars` (sized to the return type), the input-struct
`parts` (which was a genuine OOB read — the loop clamped at 32 but `CompositeConstruct` used the
full member count), the OpEntryPoint interface, `buffer_indices`/`resource_indices`
(`sks_write.c`), and `blobs` (`api.c`, `SVSL_API_MAX_STAGES` deleted). The sweep also uncovered
two more of the same habit: the switch `case_labels[64]` + OpSwitch `words[128]` (which the IR
switch's own un-capped `lits` now feed), and the `cf[32]` / if-else `stack[64]` control-flow
stacks (unbounded push → stack overflow past 32-deep nesting) — both now arena-sized. Left as a
deliberate *guarded* threshold: `SVSL_EMIT_MAX_MEMBERS` gates struct-param SROA (>32-member
structs fall back to the shadow-var path, no truncation). Corpus regressions
`check_wide_composites.hlsl` (80 members / 80 cases) and `check_deep_nesting.hlsl` (40 nested
ifs) validate under `spirv-val`.

### 🔴✓ 12. A few narrower sema gaps
- `Texture2DMSArray` loses its multisampled flag — matched by the generic `Texture*` branch
  after the exact `Texture2DMS` compare fails; interns as a plain 2D array texture
  (`sema.c:144`). **Fix:** detect the `MS` infix generically, or reject the name.
- Local `TileImage` declarations slip past the resource-in-function guard — it open-codes the
  resource-kind set and omits `svsl_type_tileimage` that `is_resource_kind` includes
  (`check.c:1226`). **Fix:** call `is_resource_kind` so the two lists can't drift.
- CLI `-i`/`-D` past their caps (16/32) leave the next argv to be parsed as an input file —
  the `++i` lives inside the cap-guarded assignment (`cli/main.c:289`). **Fix:** advance and
  diagnose regardless of the cap.

☑ **Fixed (all three).** `texdim_from_name` now strips an `MS` infix generically and reports it,
so `Texture2DMSArray` keeps its multisampled flag and the special-cased `Texture2DMS` block is
gone. The resource-kind predicate moved to `svsl_type_is_resource` (`types.c`) and both the
sema global-resource path and the check.c local-decl guard call it — the two lists can no longer
drift, and local `TileImage` is now rejected. The CLI consumes the `-i`/`-D` operand before the
cap check and prints an "ignoring …" note past the cap, so an over-cap flag no longer eats the
next argv as an input file.

---

## Design & maintainability

### 🟠✓ 13. `emit_intrinsic` still dispatches by re-parsing the intrinsic name
`src/back/emit_intrinsics.inc:211`

Barriers, subgroup ops, `saturate`, `clip`, `tile_depth`, etc. are selected by `strcmp`/
`strncmp` and character-index probes (`name[5]=='d'`, `sub[7]=='e'`). This is exactly the
name-re-parsing the post-M9 "intrinsic identity is a `svsl_intr_tag_`" hardening claimed to
remove — and the char-index tricks misroute silently if an intrinsic is renamed. **Fix:**
key the dispatch on the tag/table row, matching the string once.

☑ **Fixed (table-driven).** The intrinsic row now carries an `svsl_emit_` category + the
`op[3]` opcode variants (float/signed/unsigned), and `emit_intrinsic` `switch`es on
`intr->emit`. The IR already stores the table index, so the name is matched exactly once (at
find time) and never re-parsed to choose a handler — a rename can't misroute. The redundant
name-keyed `intr_map` table is deleted; its opcodes moved onto the rows (same pattern as
`formats.c`/`semantics.c`, which already store `SpvImageFormat`/`SpvBuiltIn` on rows). HLSL
aliases inherit their native's category; `subgroup_*`/`quad_*` still parse their structured
suffix inside the one `svsl_emit_subgroup` arm (that sub-namespace *is* a grammar, not a
re-identification). Verified with `spirv-dis` across the signed/unsigned/float variants, matrix
ops, subgroup scans, Wave aliases, the builtin-var load, and barriers; regression corpus
`check_intrinsic_dispatch.hlsl`.

The refactor surfaced a **latent gap: `select()` had no emit mapping** — it was in the table
and type-checked, but `lower_intrinsic` never lowered the `svsl_intr_select` tag, so any use
errored with "intrinsic has no SPIR-V mapping yet". Fixed by lowering it to `svsl_ir_select`
(the same `OpSelect` the `?:` ternary produces; emit already splats a scalar condition to the
operand shape and decomposes matrices). Covered for scalar/vector conditions in the same check.

### 🟠 14. The "legacy forms emit porting hints" feature is spec'd but unbuilt — dead flag + stale docs
`src/sema/sema.c:63` is the **only** `svsl_severity_porting` emission site, and it fires only
for `[[vk::*]]`. No hint is produced for `cbuffer`, `min16float`, `groupshared`,
`StructuredBuffer`, `SamplerState`, `Wave*`, `Interlocked*`, or legacy semantics — yet
LANGUAGE_SPEC §1.2/§11/§12/§13/§14, PLAN §3/§4.1/§4.2, and the README all promise they do.
The user guide (`09-hlsl-compatibility.md`) already documents the *correct* behavior. The
`legacy_spelling` flag (written twice, read zero times — dead scaffolding, and the cause of
collision #6) is the stub for this unbuilt feature.

**Decision needed (✎):** either (a) build the legacy porting hints (consume the flag, fire
the severity) so spec/PLAN/README become true, or (b) cut the promise from spec/PLAN/README
and delete the dead flag. Everything else here assumes nothing; this one is a product call.

☑ **Resolved (a), opt-in.** Added a `porting_hints` option (`-Wporting` / API, **off by
default**) that emits `porting` hints on `[[vk::*]]` escapes (now gated behind the flag too),
legacy resource/scalar type spellings (`min16float`, `SamplerState`, `SamplerComparisonState`,
`StructuredBuffer`, `RWStructuredBuffer`, `RWTexture*`), and HLSL intrinsic aliases (`Wave*`,
`Interlocked*`, `GroupMemoryBarrier*` — table-driven via `opt_native`). Hints dedupe per
source spelling and never fire without a native alternative. Spec, guide, and DECISIONS
updated to describe the opt-in behavior; regression test `test_sema_porting_hints`. **Still
backlog:** the declaration-keyword forms (`cbuffer`/`groupshared`/`nointerpolation`) and
legacy semantics (see `docs/BACKLOG.md`).

### 🟠✓ 15. Duplicated logic clusters (single-source-of-truth)
- **`usage.c`**: `mark_res_used` is a near-verbatim copy of `mark_resource`, and the two
  instruction-walk loops (`func_globals` vs `analyze_usage`) re-implement the same
  ptr/tex/image resource walk — including the fused-sampler pairing, which must stay in sync
  (`usage.c:5,44,113`). Collapse to one visitor.
- **`func_info` linear re-scan inside per-item loops**: `emit_spirv.c:1253` and `usage.c:75`
  both re-find the entry's `svsl_func_info_t` on every parameter/member — O(insts×params×funcs).
  Hoist once per entry.
- **Attribute parsing**: `[[vk::binding]]` parse-or-error is copied at `sema.c:737` and `:1137`;
  `[offset]`/`[vk::offset]` at `:503` and `:768`; the `arg_count==1 && const_eval_int` idiom
  recurs ~8×. Extract `attr_int` / `attr_binding` helpers.
- **pp comma-splitting** (`pp.c:199` vs `:524`) and the **parser declarator emit loop**
  (`parser.c:1113` vs `:1192`) are each written twice.

☑ **Fixed (all four).** `usage.c` now has a single `mark_func_resources` visitor (with shared
`mark_resource`/`mark_buffer` taking stage-mask arrays); the per-function bool pass just passes
a "used" bit, so the fused-sampler pairing lives in exactly one place. The `func_info` scan is
now `svsl_program_func_info` (sema.c), called once per entry at all four sites (usage.c +
emit_spirv ×3) instead of per-parameter. Attribute parsing gained `attr_int1` (the
`arg_count==1 && const_eval_int` idiom, 4 sites) and `attr_binding` (the two `[[vk::binding]]`
blocks). The `#define` parameter loop now calls the existing `pp_parse_args`, and both
declarator-emit loops share `push_var_list_decls`.

### 🟠✓ 16. Smaller design smells
- `svsl_sks_write` has a `(void)`-cast `ref_diags` param and can only `return true`, yet
  callers branch on it — dead scaffolding (`sks_write.c:297`). Make it `void`.
- `svsl_result_header` calls `svsl_result_sks`, re-running usage analysis + serialization each
  call; `-sks -h` serializes the container three times (`api.c:127`). Memoize the blob on
  `impl_t` (module-private, so it respects single-source).

☑ **Fixed (both).** `svsl_sks_write` returns `void` and dropped the unused `ref_diags` param;
the caller no longer branches on a result that was always `true`. The container is memoized on
`impl_t.sks` and serialized once — `-sks`, `-h`, and reflection now share the one blob.

---

## Comments & documentation drift

### 🟡 17. PLAN.md is significantly stale — needs a reconciliation pass

☑ **Resolved by retirement.** PLAN.md (the from-empty-repo implementation plan) and the
v9 proposal were removed now that the project is built; the durable content moved to
`docs/DECISIONS.md` (decision log, prototype lessons, SKS v9 rationale) and
`docs/BACKLOG.md` (future work, with the shipped API item dropped). The stale directory
diagram and "API not built" claims went with it. All in-repo references were repointed.

- **§6.4 / §10:** say the public C API is "not built yet … version-defines stub" and list it
  as backlog. It's fully shipped (`svsl.h` 264 lines, `api.c`, passing `test_api`) — and with
  a *different, better* shape than the sketch (transparent `svsl_result_t` with `.ok`/
  `.diagnostics`/`.stages` + product accessors + an SKS **reader** `svsl_sks_parse`, vs the
  sketched opaque handle with `svsl_result_ok()`/`_diag()` accessors and `warn_mask`). Rewrite
  §6.4 to the real surface; remove the §10 item.
- **§6.5:** the directory diagram lists files that don't exist (`sema/entry.c`, `meta.c`,
  `ir/ir.c`, `passes/inline.c`, `chains.c`, `back/caps.c`, `tables/methods.c` etc.,
  `src/compile.c`, `app/src/main.cpp`). Regenerate from the real tree.
- **§6.6:** omits the `-O0/1/2` flag the CLI actually has.
- **§3/§4:** the legacy-porting-warn claims (see #14).

### 🟡✓ 18. LANGUAGE_SPEC.md drift (the M9 "honesty pass" missed these)
- Porting-hint claims across §1.2/§11–§14 (see #14) — the spec dropped its Draft banner but
  this whole area is unverified against the code.
- Says `$Globals`; the code emits `$Global` (singular, matching glslang/skshaderc —
  `sema.c:1179`). Spec text is the stale one; fix the spec.

☑ **Fixed.** Verified the porting area against the built feature: §1.2, §6 (`[[vk::*]]` table),
§11, and §13 were accurate; the two overclaims were fixed — the §14 alias-summary header said
*every* row emits a porting hint, when only the type spellings, intrinsic aliases, and
`[[vk::*]]` escapes do (keyword forms / attribute aliases / legacy semantics are accepted
silently; hints are backlog), and the §6 escape-table header now notes the `-Wporting` gating.
`$Globals` → `$Global` (singular) in §5.

### 🟡✓ 19. `float16` rounding comment overstates the guarantee
`src/sema/types.c:16` — comment says "round-to-nearest-even" but both the normal and
subnormal paths round half **away from zero** (no sticky-bit/LSB tie test). Either implement
RNE or change the comment to "round half up" (can diverge from glslang by 1 ULP on exact ties).

☑ **Fixed (implemented RNE).** Made the comment true rather than weaker: both paths now do a
proper round-to-nearest-even (tie → keep the even LSB) using a round-bit/sticky test, matching
glslang. Regression `test_f16` in `test_util.c` checks the two tie directions (even LSB stays,
odd LSB rounds up) plus basic/overflow encodings.

### 🟡✓ 20. Stale IR operand doc
`src/ir/ir.h:56` — `svsl_ir_atomic` comment says the cmpxchg value is "in aux"; it's in
`args[2]` (`value_arg_mask` returns `0x7`). Fix to
`args[0]=pointer, args[1]=value, args[2]=compare (cmpxchg), args[3]=op|(order<<8)`.

☑ **Fixed.** Comment corrected to the actual operand layout.

### 🟡✓ 21. Development-process narration in comments (against the "talk about the code" rule)
- `spirv_builder.c:61` — six lines of "simple approach … is overkill … but be safe anyway"
  plus an unused `marker` var, ahead of a plain whole-stream dedup. Replace with one line
  describing the actual behavior; delete `marker`.
- `emit_spirv.c:779` — "Assigning only one per member (**the old behaviour**) collided…".
- `optimize.c:2,16` — "the two-call pipeline SVSL **always had**", "**now able** to use the
  merged pointers." Trim to present-tense rationale.
- `types_builtin.h:14` — "only … types that take `<T>`" is false (samplers take none;
  `Texture2D` used bare). `types_builtin.c:5` — "longest-match-first" describes an ordering
  dependency the loop doesn't actually rely on.

The project-wide dev-process grep was otherwise clean — ~40 hits were legitimate present-tense
descriptions. Only the above narrate history.

☑ **Fixed (all).** `spirv_builder.c` narration collapsed to one line, `marker` deleted;
`emit_spirv.c` "old behaviour" → present-tense; `optimize.c` "always had"/"now able" trimmed to
present rationale; `types_builtin.h` `<T>` claim replaced with the actual resource-name list, and
`types_builtin.c`'s comment now says the suffix validation makes row order irrelevant (rather
than implying a load-bearing ordering).

---

## Nits

- ☑ 🔵 `emit_spirv.c:153` — dead ternary, both arms are `elem->scalar`. Reduce to one arg.
- ☑ 🔵 SPIR-V binary version word `0x00010300` (`spirv_builder.c:138`) and SKS version `9`
  (`sks_write.c:346`) are bare literals; `svsl.h` already defines `SVSL_SKS_VERSION`. Name both.
  Added `SVSL_SPIRV_VERSION` too.
- ☑ 🔵 Type-cache key copies 7 operands but the emitter writes up to 8 (`spirv_builder.c:88`) —
  latent aliasing if an 8-operand type ever appears. Asserted `count<=7` (wide structs are
  emitted uncached, so nothing reaches this path with >7 operands).
- ☑ 🔵 Octal `08`/`09` silently parses as 0 (`strtoull(...,8)`) while the comment claims C/glslang
  parity (`lexer.c:171`). Now diagnoses `8`/`9` digits in an octal literal.
- ☑ 🔵 `dump_var` omits `writeonly/coherent/volatile/precise/interp` flags — invisible to AST
  golden snapshots (`ast.c:216`). Added the memory-qualifier flags (interp were already dumped).
- ☑ 🔵 Spec-const collision diagnostic prints the name where it says "id" (`sema.c:869`). Now
  reports the numeric id and names the constant already holding it.
- ☑ 🔵 Incremental skip-if-newer ignores `#include` mtimes (`cli/main.c:237`) — matches
  skshaderc; noted the gap in a comment.
- ☑ 🔵 Arena's "16-byte aligned" promise (`arena.h:2`) relies on `malloc` alignment (fine on the
  target; noted the assumption). `svsl_array_grow_` memcpy's an unchecked possibly-NULL OOM
  result (`array.c:9`) — noted as consistent with the codebase's fatal-OOM stance.

---

## Repo hygiene

- **`docs/SKS_V9_PROPOSAL.md`** is a real design doc that PLAN §5.1 links by path, but it's
  **untracked** — commit it.
- **`docs/IDEAS.md`** is untracked raw brainstorming that contradicts shipped design (proposes
  dot-scoped `Opt.One` enum access; the spec/impl use bare unscoped enum constants). Move it
  out of `docs/` or clearly mark it non-normative scratch so it isn't mistaken for intent.

---

## Themes & suggested order

Four recurring root causes account for most of the correctness list:

1. **Constant-folding assumes narrow signed ints** (#2, #3) — audit both fold sites for
   signedness/width together.
2. **Unbounded recursion / fixed-size stack arrays with silent truncation** (#5, #9, #11) —
   one habit across parser, pp, layout, emit, sks. A uniform "grow-or-diagnose" sweep closes
   the class.
3. **The inliner's control-flow lowering** (#1) — the single scariest bug; needs a real
   return mechanism, not a break.
4. **The unbuilt "legacy porting hints" feature** (#6, #14, and much of the doc drift) — one
   product decision (build it or cut it) resolves a bug, a dead flag, and most of the spec drift.

Recommended attack order: **#1 → #2/#3 → #4 → #5 → the #14 decision** (which unblocks #6 and
the doc-drift cluster), then the #11 truncation sweep, then maintainability (#13, #15), then
comments/nits. The doc-drift pass (#17–#20) is cheap and high-value — PLAN and the spec are the
project's memory, and they're currently lying about the API status, the directory layout, and
porting behavior.
