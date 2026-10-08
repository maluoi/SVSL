// Character classes for the preprocessor and lexer: one table lookup per
// character on their hottest loops (identifier and number scans).

#pragma once

#include <stdbool.h>
#include <stdint.h>

enum { svsl_char_ident_start = 1, svsl_char_digit = 2, svsl_char_hex = 4 };

#define SVSL_CHAR_CLASS(c) \
	((((c) >= 'a' && (c) <= 'z') || ((c) >= 'A' && (c) <= 'Z') || (c) == '_' ? svsl_char_ident_start : 0) | \
	 ((c) >= '0' && (c) <= '9' ? svsl_char_digit | svsl_char_hex : 0) | \
	 (((c) >= 'a' && (c) <= 'f') || ((c) >= 'A' && (c) <= 'F') ? svsl_char_hex : 0))
#define SVSL_CHAR_ROW4(b)  SVSL_CHAR_CLASS(b), SVSL_CHAR_CLASS((b) + 1), SVSL_CHAR_CLASS((b) + 2), SVSL_CHAR_CLASS((b) + 3)
#define SVSL_CHAR_ROW16(b) SVSL_CHAR_ROW4(b), SVSL_CHAR_ROW4((b) + 4), SVSL_CHAR_ROW4((b) + 8), SVSL_CHAR_ROW4((b) + 12)
#define SVSL_CHAR_ROW64(b) SVSL_CHAR_ROW16(b), SVSL_CHAR_ROW16((b) + 16), SVSL_CHAR_ROW16((b) + 32), SVSL_CHAR_ROW16((b) + 48)

static const uint8_t svsl_char_class_[256] = { SVSL_CHAR_ROW64(0), SVSL_CHAR_ROW64(64) }; // bytes >= 128: 0

static inline bool svsl_is_ident_start(char c) { return svsl_char_class_[(uint8_t)c] & svsl_char_ident_start; }
static inline bool svsl_is_ident_char (char c) { return svsl_char_class_[(uint8_t)c] & (svsl_char_ident_start | svsl_char_digit); }
static inline bool svsl_is_digit      (char c) { return svsl_char_class_[(uint8_t)c] & svsl_char_digit; }
static inline bool svsl_is_hex_digit  (char c) { return svsl_char_class_[(uint8_t)c] & svsl_char_hex; }
