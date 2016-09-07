/*===========================================================================
=                                                                           =
=                             MtkFileToOrbit                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */
#include <stdlib.h>

int main(int argc, char *argv[])
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int orbit;                    /* Orbit Number */

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <MISR Product File>\n",argv[0]);
    exit(1);
  }

  status = MtkFileToOrbit(argv[1], &orbit);
  MTK_ERR_COND_JUMP(status);

  printf("%d\n",orbit);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Error opening file: %s\n", argv[1]);
  return status_code;
}
