/*===========================================================================
=                                                                           =
=                            MtkTimeToOrbitPath                             =
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
  int orbit;

  if (argc != 2)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkTimeToOrbitPath(argv[1],&orbit,&path);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkTimeToOrbitPath Failed.\n");
  }

  printf("Orbit %d  Path: %d\n",orbit,path);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "\nUsage: %s <time>\n\n",argv[0]);
  fprintf(stderr, "Where: <time>=YYYY-MM-DDThh:mm:ssZ\n");
  fprintf(stderr, "  Time must be on or after 2000-02-24 00:00:00 UTC\n");
  fprintf(stderr, "\nExample: MtkTimeToOrbitPath 2002-02-02T02:00:00Z\n");
  return status_code;
}

