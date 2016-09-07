/*===========================================================================
=                                                                           =
=                           MtkJulianToDateTime                             =
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
  char datetime[MTKd_DATETIME_LEN];

  if (argc != 2 || atof(argv[1]) < 1721119.5)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkJulianToDateTime(atof(argv[1]),datetime);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkJulianToDateTime Failed.\n");
  }

  printf("%s\n",datetime);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Usage: %s <Julian Date>\n",argv[0]);
  fprintf(stderr, "  Julian Date >= 1721119.5\n");
  fprintf(stderr, "Example: MtkJulianToDateTime 2453728.27313\n");
  return status_code;
}

