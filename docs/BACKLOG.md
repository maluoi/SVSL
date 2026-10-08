# SVSL Backlog

Discussed and worth doing, but no code yet. Nothing here is promised by the spec (which
documents only what exists). Rationale for shipped choices is in `docs/DECISIONS.md`; open
bugs and cleanups from the code review are tracked in `docs/CODE_REVIEW.md`.

## Runtime / sk_renderer side

- **Act on the v9 feature mask**: `skr_shader_create` should check `meta.features` against
  device support and fail early with a clear message; today the mask is written and loaded
  but not consulted.
- **Use per-stage `wave_size`** at pipeline creation (subgroup-size-control `pNext`).
- **sksc reflection should fill the v9 shape/format bytes** — skshaderc currently writes 0
  (unreported); SPIRV-Reflect has the data.
- **TileImage runtime test** — blocked on `VK_EXT_shader_tile_image` in a local driver (RADV
  lacks it) and an skr pass model for it.
- **Advance the sk_renderer pin past v11.** SKS v11 (QCOM reflection — see
  docs/DECISIONS.md) is implemented in both working trees; until sk_renderer tags a
  release containing it, svsl_view's FetchContent pin (`app/CMakeLists.txt` GIT_TAG)
  stays behind and needs the `-DFETCHCONTENT_SOURCE_DIR_SK_RENDERER` scratch-build
  override, and the sibling `~/SK/sk_renderer/build` is configured with
  `FETCHCONTENT_SOURCE_DIR_SVSL=~/SK/SVSL` so its embedded libsvsl tracks this
  working tree (clear the cache var to return to the pin).
- **The sk_renderer `-t sw` removal is ahead of its libsvsl pin.** `sksc_svsl.cpp` now
  rejects `-t sw`, but that build's `FETCHCONTENT_SOURCE_DIR_SVSL` is cleared, so its
  embedded libsvsl is still the pinned pre-flip revision — which would happily accept
  `-t sw` and, more to the point, still infers a texel format for undeclared storage
  images. Nothing in either build depends on `-t sw` any more (the `SKR_USE_WEBGPU` arm
  is `-t w`), so this is only a hand-run-CLI difference until the pin moves. Verify the
  storage-image format record against the emitted WGSL after the bump — that agreement is
  what `test_wgsl_storage_format_record` pins on this side.
- **Split SKS feature bit 13 (formatless) into read and write.** Now that an undeclared
  storage image format means `Unknown` (docs/DECISIONS.md), nearly every compute shader
  sets bit 13, and the bit is joint — so a write-only shader (the mipgen case) makes the
  runtime demand `shaderStorageImageReadWithoutFormat` too, the half with weaker old-mobile
  coverage. SVSL already tracks the two capabilities separately; the blocker is
  sk_renderer, where `sksc_file.h` has one `sksc_feature_bit_formatless` and the runtime
  one joint `has_storage_without_format` flag (`vk/_sk_renderer.h`). Splitting needs both
  repos to move together: a new bit (24 is free), a second runtime flag, and a pin bump.
- **Apply v11 at runtime in skr**: read `meta.tile_apron` into
  `VkRenderPassTileShadingCreateInfoQCOM` when creating a tile-shading render pass
  (needs the skr tile pass model), and create shape-bit-6 samplers with
  `VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM`. The descriptor-type mapping for the
  four new register values is already in `skr_shader.c`.
- **QCOM runtime verification on Adreno** — the extensions have no desktop implementation;
  compile + spirv-val + emitted-word unit tests (`tests/test_qcom.c`) + v11 container
  decode are the current bar. A device pass needs the runtime work above.

## Library surface

- **Extend porting hints to the remaining legacy forms.** `-Wporting` (opt-in, off by
  default) now hints on `[[vk::*]]` escapes, legacy resource/scalar type spellings, and HLSL
  intrinsic aliases. Still silent even under the flag: the declaration keywords `cbuffer` →
  `uniform`, `groupshared` → `workgroup`, `nointerpolation` → `flat`, and legacy semantics.
  Wiring those needs the parser to record which spelling was used (the `legacy_spelling` var
  flag is in place for the keyword cases).

- **Never bitwise-compare alignment-sensitive structured-buffer layouts against
  skshaderc.** glslang's HLSL mode emits a hybrid element layout that matches nobody —
  DX-packed member offsets with a std430-rounded array stride (its own code carries a
  TODO admitting the inconsistency; see docs/DECISIONS.md "Structured-buffer element
  layout"). SVSL's object-form elements are C-packed by default with keyword overrides
  (spec §4.2), so alignment-sensitive cases need golden-value tiers
  (`check_pack_layout`); alignment-neutral structs still compare bitwise.

## Testing

- **Ours-only compute checks** for features glslang can't compile (float atomics, spec-const
  expressions at runtime, image atomics, pack1/pack8 layouts) — run SVSL output alone and
  assert buffer contents against precomputed values.
- **`Texture2DMS` runtime pixel test** beyond the MSAA shader-resolve path.
- **Replace the harness's name-based `tex_kind_t` guessing** with the v9 shape bytes it now
  has available.
- **RWTexture image-atomic compute check** — the image-atomic op table had no coverage (a
  miscompile hid there; see `docs/CODE_REVIEW.md` #4).

## Feature menu (mobile/VR-leaning, rough priority order)

- **Fragment shading rate** (`SPV_KHR_fragment_shading_rate`: `SV_ShadingRate` in/out) and
  **fragment density map** (`SPV_EXT_fragment_density_map`) — the foveated-rendering pair.
- **Barycentrics** (`SPV_KHR_fragment_shader_barycentric`: `SV_Barycentrics`,
  `GetAttributeAtVertex`).
- **`EvaluateAttribute*` / `InterpolateAt*`** (`InterpolationFunction` capability).
- **Optimizer: compile cost.** The corpus IR phase is ~20 ms against 15 ms before
  docs/PLAN_optimizer_llvm.md (emit got 1.6 ms cheaper; total compile +5%). No hotspot is
  left: the edit commit, CSE, combine, DSE and cfg are 2–5% of a corpus compile each. The
  structural fix would be pass-level dirty tracking (skip constructs unchanged since a
  pass last saw them). Unrolled encoders take 2–3x longer to compile (ASTC 6x6 19 → 34 ms).
- **Optimizer: unroll tuning on other GPUs.** The unroll policy (≥ 32-element arrays,
  all-or-nothing per array, 32k-instruction nests, a register budget of 64 scalars inside
  a rolled loop and 256 straight-line) was measured on Adreno 740 with sk_texenc and a
  synthetic encoder sweep (docs/PLAN_optimizer_llvm.md Phase 3). Mali and desktop drivers
  may want other budgets; the synthetic generator is the tool for checking.

## Documentation

- Document the `svsl_ir_tex` operand encoding (the IR dump is the only current reference).
- The half→fp16 dual-variant container idea (`docs/DECISIONS.md`, half vs float16) — revisit
  when sk_renderer wants it.
