// C header output: the SKS container as an embeddable byte array (skshaderc -h).

#pragma once

#include "sks_write.h"
#include "../util/arena.h"
#include "../util/str.h"

// Renders "const unsigned char sks_<name>[] = {...};" text, arena-owned.
const char *svsl_header_write(svsl_arena_t *arena, svsl_str_t name, svsl_sks_blob_t blob);
