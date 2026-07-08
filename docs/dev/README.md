# Dev Guides

Task-oriented guides for working **on** the compiler and its test infrastructure,
written for AI coding agents: decision trees, exact commands, and verification
steps rather than narrative. (The user-facing language guide lives in
`docs/guide/`; design rationale in `docs/DECISIONS.md`.)

- [Adding test shaders](adding-test-shaders.md) — get a new shader from idea to
  *verified* with no silent coverage gaps: corpus placement, verification tiers,
  authoring `compute_cfgs[]` entries, deriving golden values, failure modes.
