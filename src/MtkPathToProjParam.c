/*===========================================================================
=                                                                           =
=                            MtkPathToProjParam                             =
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

#define PRINT_INT(I) printf(#I " = %d\n",I)
#define PRINT_LONG(I) printf(#I " = %ld\n",I)
#define PRINT_LLONG(I) printf(#I " = %lld\n",I)
#define PRINT_FLOAT(F) printf(#F " = %f\n",F)

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_MisrProjParam pp;
  int i;

  if (argc != 3)
  {
    fprintf(stderr,"Usage: %s <path> <resolution meters>\n",argv[0]);
    exit(1);
  }

  status = MtkPathToProjParam(atoi(argv[1]), atoi(argv[2]), &pp);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkPathToProjParam failed!\n");
  }

  PRINT_INT(pp.path);
  PRINT_LLONG(pp.projcode);
  PRINT_LLONG(pp.zonecode);
  PRINT_LLONG(pp.spherecode);

  for (i = 0; i < 15; ++i)
    printf("pp.projparam[%d] = %.10f\n",i,pp.projparam[i]);

  PRINT_FLOAT(pp.ulc[0]);
  PRINT_FLOAT(pp.ulc[1]);
  PRINT_FLOAT(pp.lrc[0]);
  PRINT_FLOAT(pp.lrc[1]);

  PRINT_INT(pp.nblock);
  PRINT_INT(pp.nline);
  PRINT_INT(pp.nsample);

  for (i = 0; i < 179; ++i)
    printf("pp.reloffset[%d] = %f\n",i,pp.reloffset[i]);

  PRINT_INT(pp.resolution);

  return 0;

ERROR_HANDLE:
  return status_code;
}
