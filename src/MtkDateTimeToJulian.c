/*===========================================================================
=                                                                           =
=                           MtkDateTimeToJulian                             =
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

int main(int argc, char *argv[])
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  double jd;                    /* Julian Date */

  if (argc != 2)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkDateTimeToJulian(argv[1],&jd);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkDateTimeToJulian Failed.\n");
  }

  printf("%f\n",jd);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Usage: %s <Date and Time>\n",argv[0]);
  fprintf(stderr, "  Date and Time in ISO 8601 format.\n");
  fprintf(stderr, "Example: MtkDateTimeToJulian 2002-05-02T02:00:00Z\n");
  return status_code;
}

