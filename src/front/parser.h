// Parser: token array → AST. Two passes — pass 1 scans top-level declarations for
// struct names (forward references work without prototypes), pass 2 builds the AST
// with Pratt expression parsing. Recovers after errors and keeps reporting.

#pragma once

#include "ast.h"
#include "lexer.h"
#include "../diag.h"
#include "../util/arena.h"

// Never returns NULL; on errors the AST holds everything parsed successfully.
svsl_ast_t *svsl_parse(svsl_arena_t *arena, const svsl_token_list_t *tokens,
                       svsl_diag_list_t *ref_diags);
