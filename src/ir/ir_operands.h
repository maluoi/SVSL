// Per-op operand layout: which args[]/aux entries of an instruction are value
// references (vs. literals like an extract index or a resource id). Single
// source of truth shared by every IR pass that walks the dataflow graph —
// DCE, forwarding, dead-store, CSE. Encodes the same op semantics as ir.h.

#pragma once

#include "ir.h"

// Bitmask over args[0..3]: bit a set → args[a] is a value/pointer reference.
uint32_t svsl_ir_value_arg_mask(const svsl_ir_inst_t *inst);

// True when the instruction's aux pool holds value references (chain indices,
// construct components, intrinsic/tex operands) rather than literals.
bool svsl_ir_aux_holds_values(const svsl_ir_inst_t *inst);

// True for ops whose execution is observable regardless of whether their
// result is used (stores to observable memory, control flow, barriers, …).
// Intrinsics are impure only when they return void — the barriers; every
// value-producing intrinsic (math, derivatives, subgroup, pack) is a pure
// function of its operands and may be eliminated when its result is dead.
bool svsl_ir_has_side_effects(const svsl_ir_inst_t *inst, const svsl_types_t *types);

// True when this intrinsic instruction is a pure value (safe to DCE/CSE): it
// produces a non-void result. Barriers (void) are the only impure intrinsics.
bool svsl_ir_intrinsic_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types);

// True for control-flow ops that end a straight-line run. Local passes
// (forwarding, CSE) reset their per-region state here so a reused value always
// dominates its use.
bool svsl_ir_ends_run(svsl_ir_op_ op);

// Redirect every value/pointer operand of instruction `idx` through `remap`
// (following remap chains to their fixed point). `remap[i] == i` means unchanged;
// `count` is the instruction count (operands referencing >= count, e.g. literals
// mislabeled, are left alone). Shared by the forwarding and CSE apply steps.
void svsl_ir_remap_operands(svsl_ir_func_t *fn, uint32_t idx,
                            const uint32_t *remap, uint32_t count);
