// Compiler attribute shims. Everything here compiles away to nothing on
// compilers that don't support the attribute, so it never changes semantics.

#pragma once

// printf-style format checking on our own variadic wrappers (diagnostics, the
// text emitters). GCC/Clang verify the arguments against the format string;
// MSVC has no equivalent, so it expands to nothing there.
#if defined(__GNUC__) || defined(__clang__)
	#define SVSL_PRINTF(fmt_index, args_index) __attribute__((format(printf, fmt_index, args_index)))
#else
	#define SVSL_PRINTF(fmt_index, args_index)
#endif
