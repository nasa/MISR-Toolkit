/*===========================================================================
=                                                                           =
=                             MtkOrbitToPath                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int path;

  if (argc != 2 || atoi(argv[1]) < 4)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkOrbitToPath(atoi(argv[1]),&path);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkOrbitToPath Failed.\n");
  }

  printf("Path: %d\n",path);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "\nUsage: %s <orbit>\n\n",argv[0]);
  fprintf(stderr, "  Orbit >= 4\n");
  fprintf(stderr, "\nExample: MtkOrbitToPath 12115\n");
  return status_code;
}

