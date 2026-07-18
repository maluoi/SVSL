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
- **Codegen niceties**: `OpVectorTimesScalar` instead of splat+multiply, dynamic vector
  indexing via `OpVectorExtractDynamic`, statement hints → SPIR-V loop/selection control masks
  if a case ever shows they matter.

## Documentation

- Document the `svsl_ir_tex` operand encoding (the IR dump is the only current reference).
- The half→fp16 dual-variant container idea (`docs/DECISIONS.md`, half vs float16) — revisit
  when sk_renderer wants it.
