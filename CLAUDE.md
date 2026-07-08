# SVSL — SPIR-V Shading Language

SVSL is a shader language and compiler that compiles an HLSL-compatible language directly to
SPIR-V, with first-class syntax for modern SPIR-V features that HLSL obscures. It is a clean
rewrite of the concept prototyped in `~/projects/spirv_sl` (do not reuse its code directly;
its lessons are captured in docs/DECISIONS.md).

The project produces:
- **libsvsl** — a zero-dependency C11 library (lexer → parser → sema → IR → SPIR-V).
- **svslc** — a CLI compiler that outputs StereoKit `.sks` shader files, raw `.spv`, or C headers.
- **svsl_view** — a dev-only visual test app (Vulkan/ImGui) for pixel-diffing SVSL output
  against the reference compiler `skshaderc`.

Design rationale lives in **docs/DECISIONS.md**, the language surface in
**docs/LANGUAGE_SPEC.md**, and future work in **docs/BACKLOG.md** — read the relevant one
before making architectural decisions. The primary consumer is StereoKit: SVSL must compile
StereoKit's builtin shader corpus and emit its SKSHADER container format.

## Building

```bash
cmake -B build
cmake --build build
# cmake is preferred; the build system uses ninja
```

Targets: `svsl` (static lib), `svslc` (CLI), `svsl_tests` (unit/corpus tests), `svsl_view`
(visual test app; only built when `SVSL_BUILD_VIEW=ON`, pulls sk_app/sk_renderer/imgui via
FetchContent).

## Running Tests

```bash
ctest --test-dir build --output-on-failure   # all unit + corpus tests
./build/tests/svsl_tests                     # direct run
./build/app/svsl_view -test tests/shaders    # batch visual comparison vs skshaderc
./build/app/svsl_view -file <shader.hlsl>    # single-shader detail + SPIR-V diff
```

Corpus compile tests run every shader in `tests/shaders/` and (when `spirv-val` is on PATH)
validate the output. Visual tests render with both skshaderc and svslc and compare pixels;
match threshold is average error < 0.01. Compute shaders compare buffer/texture outputs
instead, driven by the `compute_cfgs[]` table in `app/src/compare.c`.

**Adding a test shader? Read `docs/dev/adding-test-shaders.md` first** — a new compute
shader is silently skipped until it gets a config entry, and the guide covers corpus
placement, verification tiers, and golden-value derivation. Dev guides for agents live
in `docs/dev/`.

## Dependency Policy

**The core library and CLI link nothing but libc.** SPIR-V opcodes/enums come from a vendored
single-file `vendor/spirv.h` (from SPIRV-Headers) — never hand-write SPIR-V constants.
Validation shells out to `spirv-val` if present; it is never linked. Only `svsl_view` may use
FetchContent dependencies (sk_app, sk_renderer, imgui), and it is off by default.

## Code Style

Modern C11. Tabs for indentation, never spaces. Opening brace at the end of the line.

- Use `stdint.h` types: `int32_t` not `int`, `uint8_t` not `unsigned char`.
- Use `//` comments, not `/* */`.
- Prefer simplicity, low indentation, and early-outs.
- Use designated initializers for structs, inline at call sites when possible:
  `compile(&(svsl_options_t){ .spirv_version = SVSL_SPIRV_1_3 });`
- **Functional style**: functions should be pure where possible, taking specific arguments
  rather than whole contexts. Exception: public functions of singleton-like modules.
- **Data-oriented design**: prefer flat arrays and indices over pointer graphs; prefer
  data tables over branching logic (intrinsics, keywords, semantics, and type conversions
  are table entries, not code).
- **Single source of truth**: caching a value in multiple places is a source of bugs. Derive,
  don't duplicate. If a cache is genuinely needed, it must be private to one module.
- Parameter prefixes: `opt_` = optional (nullable), `out_` = written to, `ref_` = read and
  written.
- **Code in columns**: align related declarations so they form visual columns:
  ```c
  int32_t     value1   = 10;
  svsl_vec3_t position = {0, 1, 2};
  ```
- Memory: all per-compilation allocations come from one arena owned by the result object.
  The caller frees everything with a single `svsl_result_free()`. No mixed ownership.
- Naming: `svsl_` prefix on all public symbols, `snake_case`, typedefs end in `_t`,
  enums end in `_` with values as `svsl_stage_vertex`-style lowercase.

## Ground Rules Learned From the Prototype

- The spec (`docs/LANGUAGE_SPEC.md`) documents only committed v1 features — deferred
  features live in docs/DECISIONS.md and docs/BACKLOG.md, never in the spec. Each change
  verifies the touched spec sections against the implementation and fixes mismatches.
- No dead scaffolding: don't write functions "for later". Add them when a caller exists.
- Every diagnostic has a source location that survives the preprocessor (line mapping).
- Options passed to the API must be honored or rejected — never silently ignored.
- When behavior is ambiguous, match `skshaderc` (glslang) output — it is the reference the
  visual tests compare against.
- No JSON, anywhere — input, output, or config. Machine-readable data is the C API, the
  binary container, or plain text tables.

## Key External References

- StereoKit shader corpus (must compile): `StereoKit/StereoKitC/shaders_builtin/`, `StereoKit/tools/include/stereokit.hlsli`, `StereoKit/Examples/Assets/Shaders/` in the StereoKit repo (https://github.com/StereoKit/StereoKit)
- SKS container format (authoritative): `sksc_file.h` and `sksc.cpp` in the sk_renderer repo (https://github.com/StereoKit/sk_renderer)
- Reference compiler: `skshaderc` built from `sk_renderer` (binary at `sk_renderer/build/skshaderc/skshaderc`)
- There's a good chance you may find active checkouts of these projects in sibling directories next to this one.