/*===========================================================================
=                                                                           =
=                             MtkFileLGID_test                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char *lgid;

  MTK_PRINT_STATUS(cn,"Testing MtkFileLGID");

  /* Normal test call */
  status = MtkFileLGID("../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf",&lgid);
  if (status == MTK_SUCCESS && strcmp(lgid,"MISR_AM1_AGP_P039_F01_24.hdf") == 0)
  {
    free(lgid);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */

  status = MtkFileLGID(NULL,&lgid);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileLGID("../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf",NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  status = MtkFileLGID("../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc",&lgid);
  if (status == MTK_SUCCESS && strcmp(lgid,"MISR_AM1_AS_LAND_P039_O002467_F08_23.nc") == 0)
  {
    free(lgid);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileLGID("../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc",NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
