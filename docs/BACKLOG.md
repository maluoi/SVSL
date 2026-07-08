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
- **QCOM tile shading** (`VK_QCOM_tile_shading`): interesting for Quest-class hardware; needs
  an skr rendering model before language syntax is worth designing.

## Library surface

- **Extend porting hints to the remaining legacy forms.** `-Wporting` (opt-in, off by
  default) now hints on `[[vk::*]]` escapes, legacy resource/scalar type spellings, and HLSL
  intrinsic aliases. Still silent even under the flag: the declaration keywords `cbuffer` →
  `uniform`, `groupshared` → `workgroup`, `nointerpolation` → `flat`, and legacy semantics.
  Wiring those needs the parser to record which spelling was used (the `legacy_spelling` var
  flag is in place for the keyword cases).

- **Object-form `StructuredBuffer<T>` is hardcoded std430 and diverges from both glslang
  and C.** For `struct { uint id; float2 pos; float scale; }`: plain C / D3D packs
  0/4/12 stride 16; SVSL std430 packs 0/8/16 stride 24; glslang emits a hybrid that
  matches nobody — DX offsets (0/4/12) but a std430-rounded stride (24), so C arrays
  break from element 1 even with skshaderc. Alignment-neutral element structs (all of
  StereoKit's builtins) are unaffected; `check_bool_store` deliberately orders members
  that way. Block-form buffers already solve this via `pack1` (byte-identical to C,
  verified; see spec §4.2). Options for the object form: accept a layout keyword there
  (`StructuredBuffer<T>` + pack1 spelling TBD), or flip its default to DX packing with
  scalarBlockLayout inferred only when a member actually straddles. Reflection
  element_size follows whatever is chosen. Don't bitwise-compare alignment-sensitive
  cases against skshaderc — the glslang hybrid can't be matched meaningfully.

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
