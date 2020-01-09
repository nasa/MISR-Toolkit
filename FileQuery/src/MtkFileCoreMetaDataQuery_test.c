/*===========================================================================
=                                                                           =
=                      MtkFileCoreMetaDataQuery_test                        =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int nparam;
  char **paramlist;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileCoreMetaDataQuery");
 
  /* Normal test call */
  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf", &nparam, &paramlist);

  if (status == MTK_SUCCESS && nparam == 29 &&
      strcmp("LOCALGRANULEID",paramlist[0]) == 0)
  {
    MtkStringListFree(nparam,&paramlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  status = MtkFileCoreMetaDataQuery("", &nparam, &paramlist);
  if (status == MTK_HDF_SDSTART_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileCoreMetaDataQuery(NULL, &nparam, &paramlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf", NULL, &paramlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf", &nparam, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  
  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc", &nparam, &paramlist);

  if (status == MTK_SUCCESS && nparam == 38 &&
      strcmp("LOCALGRANULEID",paramlist[0]) == 0)
  {
    MtkStringListFree(nparam,&paramlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc", NULL, &paramlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataQuery("../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc", &nparam, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
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
