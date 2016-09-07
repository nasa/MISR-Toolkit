/*===========================================================================
=                                                                           =
=                               MtkDdToRad                                  =
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
  double rad;

  if (argc != 2)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkDdToRad(atof(argv[1]),&rad);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkDdToRad Failed.\n");
  }

  printf("Radians: %f\n",rad);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "\nUsage: %s <Decimal Degrees>\n\n",argv[0]);
  fprintf(stderr, "Example: MtkDdToRad 130.08284167\n");
  return status_code;
}
