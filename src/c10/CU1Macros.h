#pragma once

#if defined(__GNUC__)
#define C10_CU1_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_CU1_EXPORT
#endif // defined(__GNUC__)

#define C10_CU1_IMPORT C10_CU1_EXPORT

// This one is being used by libc10_cu1.so
#ifdef C10_CU1_BUILD_MAIN_LIB
#define C10_CU1_API C10_CU1_EXPORT
#else
#define C10_CU1_API C10_CU1_IMPORT
#endif
