/*===========================================================================
=                                                                           =
=                       MtkFileGridToFieldList_test                         =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  char filename[200];		/* HDF-EOS filename */
  char gridname[200];		/* HDF-EOS gridname */
  char *fieldlist_expected[] = {"AveSceneElev", "StdDevSceneElev",
				"StdDevSceneElevRelSlp", "PtElev",
				"GeoLatitude", "GeoLongitude",
				"SurfaceFeatureID", "AveSurfNormAzAng",
				"AveSurfNormZenAng" };
  char *fieldlist_expected1[] = {"Red Radiance/RDQI", "Red Radiance",
				 "Red RDQI", "Red DN", 
				 "Red Equivalent Reflectance", "Red Brf" };
  char *fieldlist_expected2[] = { "LandHDRF", "Raw LandHDRF",
				  "Flag LandHDRF", "LandHDRFUnc",
				  "Raw LandHDRFUnc", "RDQI", "LandBHR",
				  "Raw LandBHR", "LandBHRRelUnc",
				  "Raw LandBHRRelUnc", "LandBRF",
				  "Raw LandBRF", "Flag LandBRF",
				  "LandDHR", "Raw LandDHR", "BRFModParam1",
				  "Raw BRFModParam1", "BRFModParam2",
				  "Raw BRFModParam2", "BRFModParam3",
				  "Raw BRFModParam3", "BRFModFitResid",
				  "Raw BRFModFitResid", "NDVI",
				  "Raw NDVI", "BiomeBestEstimate",
				  "BiomeBestEstimateQA", "LAIBestEstimate",
				  "LAIBestEstimateQA", "FPARBestEstimate",
				  "BHRPAR", "DHRPAR", "LAIMean1",
				  "Raw LAIMean1", "LAIDelta1",
				  "Flag LAIDelta1", "LAINumGoodFit1",
				  "LAIMean2", "Raw LAIMean2", "LAIDelta2",
				  "Flag LAIDelta2", "LAINumGoodFit2",
				  "LAIQA", "SubrVar", "Raw SubrVar" };
  char *fieldlist_expected3[] = {"BlueConversionFactor",
                                 "GreenConversionFactor",
                                 "RedConversionFactor",
				 "NIRConversionFactor"};
  char *fieldlist_expected4[] = {
    "X_Dim",
    "Y_Dim",
    "Block_Number",
    "Block_Start_X_Index",
    "Block_Start_Y_Index",
    "Time",
    "Camera_Dim",
    "Band_Dim",
    "Biome_Type_Dim",
    "Latitude",
    "Longitude",
    "Hemispherical_Directional_Reflectance_Factor",
    "Hemispherical_Directional_Reflectance_Factor_Uncertainty",
    "Bi-Hemispherical_Reflectance",
    "Bi-Hemispherical_Reflectance_Relative_Uncertainty",
    "Bidirectional_Reflectance_Factor",
    "Directional_Hemispherical_Reflectance",
    "Normalized_Difference_Vegetation_Index",
    "Biome_Best_Estimate",
    "Leaf_Area_Index_Best_Estimate",
    "Leaf_Area_Index_Best_Estimate_QA",
    "Fractional_Absorbed_Photosynthetically_Active_Radiation_Best_Estimate",
    "Photosynthetically_Active_Radiation_Integrated_Bi-Hemispherical_Reflectance",
    "Photosynthetically_Active_Radiation_Integrated_Directional_Hemispherical_Reflectance",
    "Leaf_Area_Index_QA",
    "AUXILIARY/BRF_HDRF_Interpolation_Flag",
    "AUXILIARY/mRPV_Model_r0",
    "AUXILIARY/mRPV_Model_k",
    "AUXILIARY/mRPV_Model_b",
    "AUXILIARY/mRPV_Model_Fit_Residual",
    "AUXILIARY/Mean_Leaf_Area_Index_Test_1",
    "AUXILIARY/Leaf_Area_Index_Merit_Function_Test_1",
    "AUXILIARY/Number_Passing_LAI_Values_Test_1",
    "AUXILIARY/Mean_Leaf_Area_Index_Test_2",
    "AUXILIARY/Leaf_Area_Index_Merit_Function_Test_2",
    "AUXILIARY/Number_Passing_LAI_Values_Test_2",
    "AUXILIARY/Equivalent_Reflectance_Subregion_Variability",
    "AUXILIARY/AGP_Surface_Type",
    "AUXILIARY/Suitable_For_Surface_Retrieval"
  };
  
  char **fieldlist;
  int num_fields;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridToFieldList");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (num_fields != sizeof(fieldlist_expected) /
                       sizeof(*fieldlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_fields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(num_fields, &fieldlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "BRF Conversion Factors");

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (num_fields != sizeof(fieldlist_expected3) /
                       sizeof(*fieldlist_expected3))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_fields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected3[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }

    MtkStringListFree(num_fields, &fieldlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "RedBand");

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (num_fields != sizeof(fieldlist_expected1) /
                       sizeof(*fieldlist_expected1))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_fields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected1[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(num_fields, &fieldlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (num_fields != sizeof(fieldlist_expected2) /
                       sizeof(*fieldlist_expected2))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_fields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected2[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(num_fields, &fieldlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileGridToFieldList(NULL,gridname,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,NULL,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,gridname,NULL,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,NULL);
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
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,&fieldlist);
  if (status == MTK_SUCCESS) {
    int num_fields_expect = sizeof(fieldlist_expected4) / (sizeof(char *));
    if (num_fields != num_fields_expect) {
      data_ok = MTK_FALSE;
    }

    for (i = 0; i < num_fields; ++i) {
      if (strcmp(fieldlist[i],fieldlist_expected4[i]) != 0)
      {
        data_ok = MTK_FALSE;
      }
    }
    MtkStringListFree(num_fields, &fieldlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,NULL,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,gridname,NULL,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToFieldList(filename,gridname,&num_fields,NULL);
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
