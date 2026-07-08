// Builtin type-name recognition. The parser needs "is this identifier a type?"
// to disambiguate declarations, casts, and constructors. Sema adds full type info.

#pragma once

#include "../util/str.h"

#include <stdbool.h>

// True for scalar/vector/matrix names (float, half3, min16float4, float4x4, ...),
// resource types (Texture2D, StructuredBuffer, Image2D, Sampler, ...), and void.
bool svsl_type_name_is_builtin(svsl_str_t name);

// True for the built-in resource object type names (textures, images, buffers,
// samplers, subpass inputs, tile images).
bool svsl_type_name_is_resource(svsl_str_t name);
