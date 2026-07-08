// Storage image format name → SPIR-V ImageFormat mapping. Shared so sema can
// validate format names at the declaration and the backend can emit the value.

#pragma once

#include "../util/str.h"

#include <stdbool.h>
#include <stdint.h>

// Looks up an image format name ("rgba8", "r32f", ...). Returns false for
// unknown names; out_spv receives the SpvImageFormat value on success.
bool svsl_image_format_find(svsl_str_t name, uint32_t *out_spv);

// SpvImageFormat for a storage image type: the declared name when present
// (sema already validated it), else inferred from the texel type like glslang
// — float4 → Rgba32f, int2 → Rg32i (32-bit formats; count 3 rounds up to 4).
// Inference keeps images off the optional StorageImage*WithoutFormat capabilities.
// Shared by SPIR-V emission and the SKS v9 resource records.
struct svsl_types_t;
struct svsl_type_t;
uint32_t svsl_image_format_for(const struct svsl_types_t *types, const struct svsl_type_t *t);

// True when the SpvImageFormat value needs SpvCapabilityStorageImageExtendedFormats
// (everything outside SPIR-V's base storage format set, e.g. R8, Rg32f).
bool svsl_image_format_extended(uint32_t spv_format);
