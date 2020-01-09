/*===========================================================================
=                                                                           =
=                        MtkFileGridFieldCheck_test                         =
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
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridFieldCheck");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "");
  strcpy(gridname, "");
  strcpy(fieldname, "");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "");
  strcpy(fieldname, "");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_INVALID_GRID) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_INVALID_FIELD) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Radiance/RDQI");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Equivalent Reflectance");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red BRF");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Radiance/RDQI[2]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_STEREO_P037_O029058_F07_0013.hdf");
  strcpy(gridname, "SubregParams");
  strcpy(fieldname, "StereoHeight_BestWinds");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_STEREO_P037_O029058_F07_0013.hdf");
  strcpy(gridname, "SubregParams");
  strcpy(fieldname, "StereoHeight_BestWinds[0]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth[2]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth[0][3]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth[6]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_INVALID_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth[a]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");
  strcpy(gridname, "RegParamsAer");
  strcpy(fieldname, "RegBestEstimateSpectralOptDepth[]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandBRF[2][5]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandBRF[2][10]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_INVALID_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandBRF[2]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandBRF[2][3][2]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "Raw LandBRF[2][3]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "Makefile");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "Raw LandBRF[2][3]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileGridFieldCheck(NULL, gridname, fieldname);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  status = MtkFileGridFieldCheck(filename, NULL, fieldname);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldCheck(filename, gridname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Bidirectional_Reflectance_Factor[2][3]");

  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldCheck(filename, NULL, fieldname);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldCheck(filename, gridname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(fieldname, "Bidirectional_Reflectance_Factor[2]");
  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(fieldname, "Bidirectional_Reflectance_Factor[2][9]");
  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_INVALID_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(fieldname, "Bidirectional_Reflectance_Factor[2][3][1]");
  status =MtkFileGridFieldCheck(filename, gridname, fieldname);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
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
