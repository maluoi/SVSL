// Semantic name ↔ SPIR-V builtin mapping (case-insensitive, per stage/direction).

#pragma once

#include "../util/str.h"

#include <stdbool.h>
#include <stdint.h>

typedef enum svsl_sem_io_ {
	svsl_sem_vs_in = 0,
	svsl_sem_vs_out,
	svsl_sem_ps_in,
	svsl_sem_ps_out,
	svsl_sem_cs_in,
} svsl_sem_io_;

typedef struct svsl_semantic_info_t {
	bool     is_builtin;  // false: regular location-numbered IO
	uint32_t builtin;     // SpvBuiltIn value
	int32_t  target_index;// SV_TargetN output location (when not builtin)
	uint8_t  depth_mode;  // FragDepth only: 0 = replacing, 1 = SV_DepthGreaterEqual,
	                      // 2 = SV_DepthLessEqual (conservative depth execution modes)
} svsl_semantic_info_t;

// Classifies `semantic` for a given stage/direction. Returns false for empty or
// unknown semantics (treated as plain location-numbered IO by the caller).
bool svsl_semantic_lookup(svsl_str_t semantic, svsl_sem_io_ io, svsl_semantic_info_t *out_info);
