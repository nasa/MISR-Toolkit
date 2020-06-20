#ifndef ERRORMACROS_H
#define ERRORMACROS_H

#include <stdio.h>

#define HDFEOS_ERROR_CHECK(msg) \
  if (hdfeos_status_code == FAIL) { \
     fprintf(stderr, "Error: %s at line %d\n", msg, __LINE__); \
     exit(1); \
  }

#define MEM_ERROR_CHECK(msg) \
  if (mem_status_code == NULL) { \
     fprintf(stderr, "Error: %s at line %d\n", msg, __LINE__); \
     exit(1); \
  }

#define MTK_ERROR(msg) \
  { \
     fprintf(stderr, "Error: %s at line %d\n", msg, __LINE__); \
     exit(1); \
  }

#ifdef MISRWARN
  #define WRN_LOG_JUMP(msg) \
   { \
      fprintf(stderr,"Warning: %s in %s <Line: %d>\n", \
	msg, FUNC_NAMEm, __LINE__); \
      goto ERROR_HANDLE; \
   }
#else
  #define WRN_LOG_JUMP(msg) goto ERROR_HANDLE;
#endif

#endif /* ERRORMACROS_H */
