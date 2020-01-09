/*===========================================================================
=                                                                           =
=                        MtkFileCoreMetaDataGet_test                        =
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
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MtkCoreMetaData metadata = MTK_CORE_METADATA_INIT;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileCoreMetaDataGet");
 
  /* Normal test call */
  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", "LOCALGRANULEID", &metadata);
  if (status == MTK_SUCCESS && metadata.datatype == MTKMETA_CHAR &&
      metadata.num_values == 1 &&
      strcmp(metadata.data.s[0],"MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf") == 0)
  {
    MtkCoreMetaDataFree(&metadata);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", "GRINGPOINTSEQUENCENO", &metadata);
  if (status == MTK_SUCCESS && metadata.datatype == MTKMETA_INT &&
      metadata.num_values == 560 && metadata.data.i[0] == 560)
  {
    MtkCoreMetaDataFree(&metadata);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", "GRINGPOINTLONGITUDE", &metadata);
  if (status == MTK_SUCCESS && metadata.datatype == MTKMETA_DOUBLE &&
      metadata.num_values == 560 && (metadata.data.d[0] - 54.8446223067902) < 0.0000001)
  {
    MtkCoreMetaDataFree(&metadata);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  status = MtkFileCoreMetaDataGet("", "LOCALGRANULEID", &metadata);
  if (status == MTK_HDF_SDSTART_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", "abcd", &metadata);
  if (status == MTK_FAILURE) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileCoreMetaDataGet(NULL, "LOCALGRANULEID", &metadata);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", NULL, &metadata);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", "LOCALGRANULEID", NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  status = MtkFileCoreMetaDataGet("../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc", "LOCALGRANULEID", &metadata);
  if (status == MTK_SUCCESS && metadata.datatype == MTKMETA_CHAR &&
      metadata.num_values == 1 &&
      strcmp(metadata.data.s[0],"MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.nc") == 0)
  {
    MtkCoreMetaDataFree(&metadata);
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
