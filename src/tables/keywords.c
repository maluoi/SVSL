#include "keywords.h"

typedef struct keyword_row_t {
	const char *name;
	svsl_kw_    kw;
} keyword_row_t;

static const keyword_row_t keyword_table[] = {
	{ "if",              svsl_kw_if              },
	{ "else",            svsl_kw_else            },
	{ "for",             svsl_kw_for             },
	{ "while",           svsl_kw_while           },
	{ "do",              svsl_kw_do              },
	{ "switch",          svsl_kw_switch          },
	{ "case",            svsl_kw_case            },
	{ "default",         svsl_kw_default         },
	{ "break",           svsl_kw_break           },
	{ "continue",        svsl_kw_continue        },
	{ "return",          svsl_kw_return          },
	{ "discard",         svsl_kw_discard         },
	{ "demote",          svsl_kw_demote          },
	{ "struct",          svsl_kw_struct          },
	{ "enum",            svsl_kw_enum            },
	{ "cbuffer",         svsl_kw_cbuffer         },
	{ "uniform",         svsl_kw_uniform         },
	{ "storagebuffer",   svsl_kw_storagebuffer   },
	{ "pushconstant",    svsl_kw_pushconstant    },
	{ "workgroup",       svsl_kw_workgroup       },
	{ "groupshared",     svsl_kw_groupshared     },
	{ "static",          svsl_kw_static          },
	{ "const",           svsl_kw_const           },
	{ "specialization",  svsl_kw_specialization  },
	{ "register",        svsl_kw_register        },
	{ "include",         svsl_kw_include         },
	{ "in",              svsl_kw_in              },
	{ "out",             svsl_kw_out             },
	{ "inout",           svsl_kw_inout           },
	{ "readonly",        svsl_kw_readonly        },
	{ "writeonly",       svsl_kw_writeonly       },
	{ "coherent",        svsl_kw_coherent        },
	{ "volatile",        svsl_kw_volatile        },
	{ "flat",            svsl_kw_flat            },
	{ "noperspective",   svsl_kw_noperspective   },
	{ "centroid",        svsl_kw_centroid        },
	{ "sample",          svsl_kw_sample          },
	{ "nointerpolation", svsl_kw_nointerpolation },
	{ "invariant",       svsl_kw_invariant       },
	{ "precise",         svsl_kw_precise         },
	{ "pack1",           svsl_kw_pack1           },
	{ "pack8",           svsl_kw_pack8           },
	{ "pack16",          svsl_kw_pack16          },
	{ "spirv",           svsl_kw_spirv           },
	{ "spirv_asm",       svsl_kw_spirv           },
	{ "true",            svsl_kw_true            },
	{ "false",           svsl_kw_false           },
};

svsl_kw_ svsl_keyword_lookup(svsl_str_t ident) {
	// runs per identifier token in the lexer: reject rows on the first character
	char c0 = ident.len > 0 ? ident.ptr[0] : '\0';
	for (int32_t i = 0; i < (int32_t)(sizeof(keyword_table) / sizeof(keyword_table[0])); i++)
		if (keyword_table[i].name[0] == c0 && svsl_str_eq_cstr(ident, keyword_table[i].name))
			return keyword_table[i].kw;
	return svsl_kw_none;
}
