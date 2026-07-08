# Adding Test Shaders Robustly

How to get a new shader from idea to *actually verified*. The trap this guide
exists to prevent: a shader that compiles, validates, and shows up as `ok` or
`skip` in the test output while verifying nothing. Follow the checklist at the
end before declaring a new test done.

## 1. Where the shader goes

| Directory    | Contents | Hand-edit? |
|--------------|----------|------------|
| `tests/shaders/checks/`    | hand-written runtime feature checks (`check_*.hlsl`) | **yes — new feature tests go here** |
| `tests/shaders/ported/`    | numbered feature corpus, maintained in-repo | yes (keep the `NN_name` prefix) |
| `tests/shaders/builtin/`   | pinned StereoKit builtins   | no — overwritten by `refresh.sh` |
| `tests/shaders/examples/`  | pinned StereoKit examples   | no — overwritten by `refresh.sh` |
| `tests/shaders/morrowind/` | pinned SKMorrowind corpus   | no — overwritten by `refresh.sh` |
| `tests/shaders/include/`   | headers only, never compiled directly | — |

Never add a hand-written test to a pinned directory — the next `refresh.sh`
deletes it. Give check shaders a `//--name = check/<feature>` first-line tag.

## 2. What runs automatically (and what does NOT)

Two harnesses pick shaders up:

1. **Corpus compile test** (`ctest --test-dir build`, `test_corpus.c`): every
   `.hlsl`/`.svsl` in all five directories runs the full pipeline and, when
   `spirv-val` is on PATH, validates. This is free — the file existing is enough.
   It proves *compiles + valid SPIR-V*, nothing about behavior.
2. **svsl_view** (`./build/app/svsl_view -test tests/shaders`): compiles with
   both svslc and skshaderc, runs both on the GPU, compares outputs. Note the
   no-argument default corpus **excludes `ported/`** — always pass
   `-test tests/shaders` for full coverage.

svsl_view classifies each shader:

- **vs + ps pair** → pixel diff (256², avg error < 0.01), plus a stereo
  multiview pass (skipped if the shader writes `SV_RenderTargetArrayIndex` —
  Vulkan forbids that in multiview). `SubpassInput` / `SubpassInputMS` shaders
  run as postfx / manual-resolve passes instead.
- **compute-only** → looked up in `compute_cfgs[]` (app/src/compare.c) by
  **exact filename stem**. No entry → `skip: compute-only (no test config)`.
  **A new compute shader is not tested until you add its table entry.**
- **skshaderc can't compile it** (SVSL-native syntax) → render shaders skip;
  compute shaders still run every non-reference tier from their config.

## 3. Pick the verification tier

Work down this list; stop at the first that applies. Stronger is better.

1. **Reference bitwise** (default, strongest): plain HLSL that skshaderc
   compiles, with deterministic outputs → buffer words must match **bitwise**,
   texture bytes within 0.01 avg. Nothing to do beyond a config entry — but
   first check the divergence list below.
2. **CPU-derived goldens** (`expect[]`): for SVSL-native shaders or divergent
   intrinsics. Bitwise for integer outputs, epsilon for float. Derive values
   from first principles (fill formulas in §5) — that makes the test a real
   correctness check, not a change detector.
3. **CPU algorithm replay**: when the output has an algorithm-level property a
   ~20-line CPU function can reproduce exactly (see `radix_golden` — the
   gpu_sort chain vs a CPU stable radix pass). Wave-size independent checks of
   this kind are the strongest option for wave-heavy code.
4. **Observed pins** (epsilon `expect[]` from a dump): only when the exact
   value is legitimately driver-dependent (linear filtering, unorm rounding).
   Say so in a comment. This is change detection, not correctness.

Every tier additionally requires the run to have **changed some output** —
untouched outputs fail (`FAILED: outputs untouched`), because two dispatches
that silently no-op "match" perfectly.

### Known glslang divergences — never bitwise-compare through these

The reference compiler is *wrong* here (verified; see check shaders' comments):

- `WavePrefixSum` → glslang emits InclusiveScan; HLSL/DXC/SVSL say exclusive.
- `InterlockedCompareStore` → glslang compiles it to **nothing**.
- `ldexp` with float exponent → glslang emits invalid SPIR-V.

If a shader's output depends on one of these, set `.no_reference = true` in its
config (skshaderc may *compile* it fine — the output is still wrong) and use a
golden tier instead.

## 4. Write the shader deterministic

The comparison only works if identical inputs give identical bits on one GPU:

- **No scheduling-dependent outputs.** Atomics: only write order-independent
  quantities — final values of commutative reductions (add/min/max/and/or/xor),
  or the *total* of atomic-returned originals (invariant even though each
  original isn't). Last-writer-wins slots, CAS winners, and float atomic-add
  results (rounding is order-dependent) are racy: either don't write them or
  leave them out of `expect[]` with a comment (see `78_spec_atomics`,
  `check_atomic_order`).
- **Wave ops compared per-lane bitwise** need `//--wave_size = 32` (first
  lines) so both pipelines get the same subgroup size (`check_wave`). Outputs
  that are wave-size *invariant* (histograms, prefix-sum totals, sort results)
  don't need it.
- **Initialize what you read.** `groupshared` starts undefined; buffer contents
  come from the harness fills (§5) — design math so fill garbage can't divide
  by zero or index out of bounds (fills are kept off zero for cbuffers, but
  storage-buffer word 0 IS 0.0).
- Every buffer slot your shader writes should be either checked by a tier or
  deliberately excluded with a comment.

## 5. Harness input model (what your shader will see)

The harness creates and fills everything deterministically, identically for
both compilers:

| Input | Fill |
|-------|------|
| storage buffer word `i`, `fill_hash` (default) | float bits of `((i * 2654435761u) & 0xFFFF) / 65535.0f` — word 0 = `0.0f` |
| storage buffer, `fill_zero` | all zeros |
| storage buffer, `fill_ramp` | word `i` = `i` as uint (use for payload/stability checks) |
| storage texture byte `i` | `(uint8)((i * 2654435761u) >> 16)` |
| sampled textures | scene assets picked by reflected dimensionality: 2D→checker, cube→cubemap, array→shadow array, 3D→gradient volume |
| named cbuffer word `i` (anything not `$Global`) | `0.25f + 0.75f * hash(i)` — word 0 of a b0 cbuffer is **exactly 0.25** |
| `$Global` loose uniforms | shader defaults, overridden by cfg `params[]` via `skr_compute_set_param` |

Key asymmetry: **only `$Global` (loose uniforms) is settable via `params[]`**
— sk_renderer auto-manages just that buffer for compute. Members of a *named*
`cbuffer` cannot be set from the config; they get the 0.25..1.0 fill. If your
test needs a specific parameter value, declare it as a loose uniform.

## 6. Authoring a `compute_cfgs[]` entry

All in `app/src/compare.c`. `.file` must equal the exact filename stem.

```c
{ .file    = "check_myfeature",
  .passes  = { { .dispatch = { 1, 1, 1 } } },   // groups, not threads
  .buffers = { { "result", 16, fill_zero } },   // count = ELEMENTS, not words
  .params  = { { "knob", sksc_shader_var_uint, .value = { 4 } } }, // $Global only
  .expect  = { { 0, 0x7E0 },                              // bitwise word
               { 3, .value = 25.47f, .eps = 1e-4f },      // float word ± eps
               { 2, .value = 209, .eps = 3, .tex = true } } }, // texture byte
```

Workflow:

1. Write the shader; `cmake --build build`; `ctest --test-dir build` (compile +
   validate must pass first).
2. Add a minimal entry: dispatch + every reflected storage buffer (the run
   FAILS if a reflected buffer has no entry — that guard keeps configs
   complete) + `$Global` params if any.
3. Author goldens against the authoring dump:
   `./build/app/svsl_view -file tests/shaders/checks/check_myfeature.hlsl`
   (without `-no-diff`) — when there's no reference it prints the first 24
   output words (hex + float) and 16 texture bytes, and per-golden mismatch
   lines once `expect[]` entries exist.
4. Prefer deriving values by hand/python from §5's fill formulas; document the
   derivation in a comment (`// sum 0..63`, `// fill(s) + s + 24`).
5. Indexing: `expect[].index` counts the **concatenated words of all config
   buffers in table order** (a buffer contributes `count * element_size / 4`
   words; element size comes from reflection — check `svslc -r`). `.tex`
   indices are bytes into the concatenated texture readback.
   Edge case: a bitwise expectation of word 0 == 0 can't be expressed
   (`{0, 0}` reads as the array terminator) — use the epsilon form.
6. Epsilon guidance: `1e-4` for float math (covers FMA contraction), `1e-3`
   for float atomic sums, `±1` for exact unorm bytes, `±3..4` for filtered /
   rounded texture bytes.
7. Rerun `-file`, then the full `-test tests/shaders`, then `ctest`.

### Multi-pass chains

`passes[]` (up to 4) dispatches sibling shaders in order on shared buffers and
textures, fully fence-synced; `.file = NULL` means the shader under test, and
per-pass `params` override the shared ones (`38_gpu_sort_init` runs itself
twice with different `e_initPass`). Chain the *complete* algorithm — the
gpu_sort chain silently produced colliding garbage until the histogram→prefix
conversion pass (init `e_initPass=2`) was included. If an intermediate stage
looks wrong, temporarily read buffers back between passes in `run_compute`.

## 7. Reading results / failure modes

| Symptom | Meaning |
|---------|---------|
| `skip compute-only (no test config)` | your shader is NOT tested — write the config |
| `FAILED: outputs untouched` | the dispatch didn't run or wrote nothing: look for `[skr:crit] ... missing binding` (buffer name mismatch? parameter in a named cbuffer? — §5) |
| `no reference (skshaderc can't compile)` on a *render* shader | expected for SVSL-native syntax; only compile+validate covers it |
| golden mismatch | rerun with `-file` for per-entry expected/got lines |
| bit-exact fails after a toolchain bump | check whether skshaderc's optimizer changed, and the divergence list in §3, before suspecting svslc |
| `(smoke only)` in an ok line | the entry has no goldens — add some |

## 8. Done checklist

- [ ] Shader is in a non-pinned directory with a `//--name` tag.
- [ ] `ctest --test-dir build` passes (compile + spirv-val).
- [ ] `./build/app/svsl_view -test tests/shaders` shows the shader as `ok`
      with a real tier in its message (`bit-exact`, `golden words`,
      `radix golden`) — not `skip`, not `(smoke only)`.
- [ ] Racy / driver-dependent output slots are excluded **with a comment**.
- [ ] Golden values have a derivation comment (or are marked as observed pins).
- [ ] Run `-test` twice — output identical (determinism).
