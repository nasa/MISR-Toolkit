/*===========================================================================
=                                                                           =
=                           MtkOrbitToTimeRange                             =
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
  char starttime[MTKd_DATETIME_LEN];
  char endtime[MTKd_DATETIME_LEN];

  if (argc != 2 || atoi(argv[1]) < 995)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkOrbitToTimeRange(atoi(argv[1]),starttime,endtime);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkOrbitToTimeRange Failed.\n");
  }

  printf("%s\n",starttime);
  printf("%s\n",endtime);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Usage: %s <Orbit Number>\n",argv[0]);
  fprintf(stderr, "  Orbit Number >= 995\n");
  fprintf(stderr, "Example: MtkOrbitToTimeRange 24372\n");
  return status_code;
}
