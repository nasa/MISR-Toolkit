/*===========================================================================
=                                                                           =
=                             MtkFileType_test                              =
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
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  char filename[200];		/* HDF filename */
  MTKt_FileType filetype;	/* File type */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileType");

  /* Normal test call */
  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");

  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_AGP) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GP_GMP) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_RCCM_GM_P037_O012049_AN_F01_0011.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GRP_RCCM) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GRP_ELLIPSOID_GM) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GRP_TERRAIN_GM) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GRP_ELLIPSOID_LM) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_LM_P161_O025396_DF_SITE_ARRUWAYS_F03_0022.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_GRP_TERRAIN_LM) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_AS_AEROSOL) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_AS_LAND) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_TC_ALBEDO_P037_O029058_F04_0007.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_TC_ALBEDO) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_TC_CLASSIFIERS_P037_O029058_F04_0006.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_TC_CLASSIFIERS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_TC_STEREO_P037_O029058_F07_0013.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_TC_STEREO) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_PP_P037_AN_22.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_PP) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_TC_CLOUD) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020_conv.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_CONVENTIONAL) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_AS_LAND) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_AS_AEROSOL) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_CMV_T20200609165530_P025_O108918_F01_0001.hdf");
  status = MtkFileType(filename, &filetype);
  if (status == MTK_SUCCESS && filetype == MTK_CMV_NRT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileType(NULL, &filetype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileType(filename, NULL);
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
