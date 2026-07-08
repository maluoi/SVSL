// SPIR-V opcode mnemonic → SpvOp value, for `spirv_asm` inline-assembly blocks.
// The rest of the compiler never needs opcode-by-name lookup — only the inline
// assembler does, so the table lives apart from the opcode *emission* helpers.

#pragma once

#include "../util/str.h"

#include <stdint.h>

// Returns the SpvOp for a mnemonic like "OpFAdd", or -1 if unknown. Only the
// canonical (unsuffixed) spelling of each opcode is recognised; KHR/EXT/etc.
// promoted aliases resolve through their core name.
int32_t svsl_spirv_opcode_lookup(svsl_str_t name);
