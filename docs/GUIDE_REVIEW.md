# docs/guide/ Review — Findings & Fix Checklist

> **STATUS (applied):** All findings resolved. H1–H4, M1–M4, L1–L4 fixed in `docs/guide/`;
> M1/M2/M3/M4 mirrored into `LANGUAGE_SPEC.md`. Improvements F1–F7 done, including a new
> `10-cheatsheet.md` appendix and a byte-accurate `svslc -r` reflection example (chapter 8).
> L5 intentionally left as-is (see below). The default-texture value list (L2) was reworded
> as consumer-interpreted metadata per author input, not pinned to a fixed list.


Source-verified review of `docs/guide/` (all 9 chapters + README) against the
implementation (the compiler is the source of truth per CLAUDE.md). Every claim below was
cross-checked against `src/`. The majority of the guide verified TRUE with exact
diagnostic-string matches; this file tracks only the errors and the improvements to make.

Status legend: `[ ]` todo · `[x]` done · `[-]` skipped (with reason)

---

## 🔴 High severity — factually wrong, would break user code

- [x] **H1. `$Globals` → `$Global` (singular).** Emitted/reflected name is `$Global`
  (`src/sema/sema.c:1222`), matching glslang. Fix all occurrences:
  - `01-tour.md:155`
  - `05-resources-and-buffers.md:282, 302, 308`
  - `09-hlsl-compatibility.md:130`
  - README prose (mental-model paragraph) — check for `$Globals`
  - LANGUAGE_SPEC §4.3 already correct (`$Global`).

- [x] **H2. storagebuffer/pushconstant default is `std430`, not `pack1`.**
  `src/sema/sema.c:797` (`default: layout = is_uniform ? pack16 : svsl_layout_std430`).
  Four layout kinds exist: `pack1/pack8/pack16/std430` (`src/sema/layout.h:11-17`).
  - `05-resources-and-buffers.md:235` — "Storage buffers default to `pack1` (tight)" → std430
  - `05-resources-and-buffers.md:259` — "Push-constant blocks default to `pack1`" → std430
  - `05-resources-and-buffers.md:284` — "storagebuffer / pushconstant — default `pack1`" → std430
  - `05-resources-and-buffers.md:273-277` layout table **omits std430** — add a 4th row
    (model on LANGUAGE_SPEC §4.2 `(none) → std430`). std430 = vectors natural-aligned, no
    16-byte array-stride rounding; validates everywhere with no device feature.

- [x] **H3. `m[i]` returns a ROW, not a column.** `src/sema/check.c:1026`
  (`svsl_type_vector_id(types, t->scalar, t->cols); // HLSL m[i] = row i`).
  - `02-types-and-values.md:105` — "column 0 as a vector (index a matrix → column)" is
    backwards → row. Also add a clarifying sentence reconciling "column-major convention"
    with "HLSL-style `m[i]` indexes rows."

- [x] **H4. Chapter 6 contradicts Chapter 2 on packed bit fields.**
  `06-intrinsics.md:102` calls struct bit-field packing "planned for a future release." It
  is fully implemented (parser → `sema.c:329-404` → `ir_build.c:355-448`) and documented in
  `02-types-and-values.md §6` + spec §3.4.1. Reword ch6 §4 to reference the implemented
  feature: "the explicit form of [struct bit-field packing](02-types-and-values.md#packed-bit-fields)."

## 🟠 Medium severity — describes mechanisms that don't exist

- [x] **M1. No SPIR-V version targeting; no version-mismatch error.** Version hardcoded
  `src/back/spirv_builder.c:8` (`0x00010300u`); no target option, no "requires higher
  version" diagnostic anywhere. Spec §15 correctly says "fixed." Fix:
  - `06-intrinsics.md:194-195` — remove the "depends on SPIR-V version being targeted …
    error naming both" claim.
  - `07-advanced-features.md:252-256` — remove the "feature that needs a higher SPIR-V
    version … compile error that names both" claim; reframe as "target is fixed at 1.3."

- [x] **M2. `[wave_size]` does NOT emit `RequiredSubgroupSize`/`SPV_EXT_subgroup_size_control`
  into SPIR-V.** It rides in `.sks` reflection, applied at pipeline creation
  (`sks_write.c`); never referenced in `emit_spirv.c`. Feature works, mechanism differs.
  - `07-advanced-features.md:208` — reword ("recorded in `.sks` reflection and applied as
    the required subgroup size at pipeline creation").
  - `07-advanced-features.md:246` capability-table row — same fix (not a SPIR-V cap/ext).

- [x] **M3. Capability-inference table (ch7) rows wrong.**
  - `07:238` `SV_ViewID → MultiView + SPV_KHR_multiview`: extension NOT emitted (grep: none
    in `src/back/`); MultiView is core in 1.3. → `MultiView` only.
  - `04-stages-and-interfaces.md:228` — "SPV_KHR_multiview extension are inferred" → drop
    the extension, keep the MultiView capability.
  - `07:243` `Texture2DMS / SubpassInputMS → multisampled image capabilities`: no MS cap is
    emitted → delete this row.
  - `07:237` `pack1 → SPV_EXT_scalar_block_layout`: it's a Vulkan device feature
    (`VK_EXT_scalar_block_layout`), nothing in SPIR-V. Ch5 (`05:275,286`) is right; make ch7
    consistent (VK_EXT, device feature).
  - Minor: "float atomics → `SPV_EXT_shader_atomic_float`" → split names `_add` / `_min_max`.

- [x] **M4. Spec constant in compute workgroup size is baked to its default, not
  specializable.** Backend substitutes compile-time default, emits literal `LocalSize` (not
  `LocalSizeId`), `src/back/emit_spirv.c:~1416`. Array sizes DO stay specializable.
  - `04-stages-and-interfaces.md:45` — add caveat that workgroup-size spec-const refs bake
    to the default (not runtime-specializable).
  - `07-advanced-features.md` spec-const section — clarify same. (Also overstated in spec §6.)

## 🟡 Low severity — wording / consistency

- [x] **L1.** `08-preprocessor-and-metadata.md:71` — language-level `include` is an
  idempotent textual paste in the preprocessor (`pp.c:792-804`), not "processed as a
  declaration." Keep include-once, drop "as a declaration rather than a textual paste."
- [x] **L2.** Default-texture value list disagrees: `08:94` = `white/black/gray/normal/...`;
  README = `white/black/gray/rough/flat/normal`. Compiler validates none (`sema.c:1355`).
  Pick ONE canonical list, use in both. **[NEEDS USER INPUT — which list]**
- [x] **L3.** `02-types-and-values.md:173` says `round(0.75 * 1023)` but `:192` says
  round-toward-zero; code truncates. Make :173 consistent (truncate / round-toward-zero).
- [x] **L4.** `08` — note metadata `//--` is only scanned in the main file; `//--` inside
  `#include`d files is ignored (`pp.c:99`). Add a one-line limitation note.
- [-] **L5.** SKIPPED (deliberate): a few quoted diagnostics are prefixes of the real strings
  (which append `('<name>')`) — "resources cannot be declared locally", "duplicate entry point
  for stage", wave_size & spec-const-id messages. Left as-is: they read clearly as
  representative message text, and pinning exact `printf`-format strings in prose would be
  brittle. Revisit if you want the guide to quote diagnostics verbatim.

## Improvements — flow / clarity / completeness

- [x] **F1. Chapter numbering.** README TOC labels Tour "—" then numbers Types "2"; body
  text treats Tour as ch1. Make consistent. **[NEEDS USER INPUT — number Tour as Ch1 or
  keep unnumbered intro]**
- [x] **F2. README TOC undersells ch7** — add `spirv_asm` and `precise` to its blurb
  (README:69); add a ch3 pointer to `spirv_asm` (it's an expression).
- [x] **F3. Consolidated porting cheat-sheet** — the HLSL↔native mapping is scattered
  (ch9 §2/§3 + inline in chs 2/5/6). Add a single reference card/appendix. **[SCOPE — ask]**
- [x] **F4. Full four-way layout table** in ch5 §5 (pack1/pack8/pack16/std430 + "default
  where" column). Folds into H2.
- [x] **F5.** Add `m[i]`=row to ch9 §5 gotchas list (after H3).
- [x] **F6. Reflection-output example** — "names over numbers" (principle #5) has no
  concrete reflected-output example. May belong in tooling docs. **[SCOPE — ask]**
- [x] **F7. Front-loaded "surprises for HLSL/DXC users"** short list in README/Tour
  (half≠float16, `m[i]`=row, matrix `*` component-wise, silent SV-degradation, resources
  can't be local). **[SCOPE — ask]**

## Note
M1, M2, M3(SV_ViewID), M4 also exist in `LANGUAGE_SPEC.md` (guide grew from spec). Decide
whether to mirror fixes there. **[NEEDS USER INPUT]**

## Verified TRUE (no action) — for the record
System-value table (all 16 rows), every intrinsic list + Wave*/Quad* alias mappings, atomics
(scope/order/seq_cst rejection), barriers, binding offsets (b+0/t,s+100/u+200), packing
alignment rules, enums, swizzles, literals + suffix error cases, `half`/`float16` split,
`$Global` pack16, porting-hint split (vk::/types/intrinsics warn; cbuffer/groupshared/
nointerpolation silent), diagnostics severities, preprocessor conditionals/defines/pragma,
metadata semantics, storage-image format token list (23 tokens, exact). All exact.
</content>
</invoke>
