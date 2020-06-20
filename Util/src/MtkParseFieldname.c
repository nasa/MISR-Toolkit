/*===========================================================================
=                                                                           =
=                            MtkParseFieldname                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

/* For strcasestr prototype in string.h on Linux64 */
#define _GNU_SOURCE

#include "MisrUtil.h"
#include "MisrError.h"
#ifdef __linux__
#include <sys/types.h>
#endif
#ifndef REGEXP_WORKAROUND
#include <regex.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <limits.h>

/** \brief Parses extra dimensions from fieldnames
 *
 *  \return Base fieldname, number of extra dimensions and extra dimension list
 */

MTKt_status MtkParseFieldname( const char *fieldname,   /**< Field name */
			       char **basefieldname,    /**< Base field name */
			       int *ndim,		/**< Number dimensions */
			       int **dimlist            /**< Dimension list */
) {
  MTKt_status status_code;      /* Return status code for error macros */
#ifdef REGEXP_WORKAROUND
  char *endptr = NULL;		/* End pointer */
#else
  int reg_status;		/* Regular expresssion parser status */
  regex_t preg;			/* Regular expression structure */
  regmatch_t pm;		/* Matched structure */
#endif
  int i = 0;			/* Dim index */
  char c[] = { "[" };		/* Character set */
  size_t l;			/* Length */
  char buf[MAXSTR];		/* Working buffer */
  char *bufptr = NULL;		/* Pointer to working buffer */
  char *baseptr = NULL;	        /* Pointer to basefieldname to return */
  int *dimptr = NULL;	        /* Pointer to dimlist to return */
  char *ptr;			/* String pointer */

  if (fieldname == NULL || basefieldname == NULL ||
      ndim == NULL || dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* --------------------------------------------------- */
  /* Allocate basefieldname and dimlist output buffers   */
  /* --------------------------------------------------- */

  baseptr = (char *)calloc(MAXSTR, sizeof(char));
  if (baseptr == NULL) MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  dimptr = (int *)calloc(MAXDIMS, sizeof(int));
  if (dimptr == NULL) MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* --------------------------------------------------- */
  /* Do a regular expression search for extra dimensions */
  /* --------------------------------------------------- */

  i = 0;
  strcpy(buf,fieldname);
  bufptr = buf;

#ifdef REGEXP_WORKAROUND
  bufptr = strchr(bufptr, '[');
  while (bufptr != NULL && i < MAXDIMS) {
    dimptr[i++] = (int)strtol(bufptr + 1, &endptr, 0);
    if (bufptr + 1 == endptr || *endptr != ']') {
      i=0;
      break;
    }
    bufptr = strchr(endptr, '[');
  }
#else
  reg_status = regcomp(&preg, "\\[ *[0-9][0-9]* *\\]", REG_EXTENDED);
  if (reg_status != 0) MTK_ERR_CODE_JUMP(MTK_FAILURE);
  reg_status = regexec(&preg, bufptr, 1, &pm, 0);
  while (reg_status == 0 && i < MAXDIMS) {
    dimptr[i++] = (int)strtol(bufptr + pm.rm_so + 1, NULL, 0);
    bufptr += pm.rm_eo;
    reg_status = regexec(&preg, bufptr, 1, &pm, REG_NOTBOL);
  }
  regfree(&preg);
#endif

  *ndim = i;
  *dimlist = dimptr;

  /* --------------------------------------------------- */
  /* Eliminate extra dimensions to isolate basefieldname */
  /* --------------------------------------------------- */

  l = strcspn(fieldname, c);
  strncpy(baseptr, fieldname, l);
  baseptr[l] = '\0';

  /* --------------------------------------------------- */
  /* Reduce virtual/extended fieldname to it's root name */
  /* --------------------------------------------------- */

  if (strcasestr(baseptr, "raw ") != NULL)
    memmove(baseptr, &baseptr[4], strlen(baseptr)-4+1);

  else if (strcasestr(baseptr, "flag ") != NULL)
    memmove(baseptr, &baseptr[5], strlen(baseptr)-5+1);

  else if ((ptr = strcasestr(baseptr, " Radiance")) != NULL)
    strcpy(ptr, " Radiance/RDQI");

  else if ((ptr = strcasestr(baseptr, " RDQI")) != NULL)
    strcpy(ptr, " Radiance/RDQI");

  else if ((ptr = strcasestr(baseptr, " DN")) != NULL)
    strcpy(ptr, " Radiance/RDQI");

  else if ((ptr = strcasestr(baseptr, " Equivalent Reflectance")) != NULL)
    strcpy(ptr, " Radiance/RDQI");

  else if ((ptr = strcasestr(baseptr, " Brf")) != NULL)
    strcpy(ptr, " Radiance/RDQI");

  *basefieldname = baseptr;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (baseptr != NULL) free(baseptr);
  if (dimptr != NULL) free(dimptr);
  return status_code;
}
