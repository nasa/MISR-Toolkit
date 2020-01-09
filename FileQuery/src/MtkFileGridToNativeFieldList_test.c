/*===========================================================================
=                                                                           =
=                     MtkFileGridToNativeFieldList_test                     =
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
  char *fieldlist_expected1[] = {"Red Radiance/RDQI" };
  char *fieldlist_expected2[] = { "LandHDRF", "LandHDRFUnc",
				  "RDQI", "LandBHR",
				  "LandBHRRelUnc", "LandBRF",
				  "LandDHR", "BRFModParam1",
				  "BRFModParam2",
				  "BRFModParam3",
				  "BRFModFitResid",
				  "NDVI",
				  "BiomeBestEstimate",
				  "BiomeBestEstimateQA", "LAIBestEstimate",
				  "LAIBestEstimateQA", "FPARBestEstimate",
				  "BHRPAR", "DHRPAR", "LAIMean1",
				  "LAIDelta1",
				  "LAINumGoodFit1",
				  "LAIMean2", "LAIDelta2",
				  "LAINumGoodFit2",
				  "LAIQA", "SubrVar" };
  char *fieldlist_expected3[] = {"BlueConversionFactor",
                                 "GreenConversionFactor",
                                 "RedConversionFactor",
				 "NIRConversionFactor"};
  char **fieldlist;
  int num_fields;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridToNativeFieldList");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,&fieldlist);
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

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,&fieldlist);
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

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,&fieldlist);
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

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,&fieldlist);
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
  status = MtkFileGridToNativeFieldList(NULL,gridname,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToNativeFieldList(filename,NULL,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToNativeFieldList(filename,gridname,NULL,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,NULL);
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
  {
    char *fieldlist_expected[] = {
      "X_Dim",
      "Y_Dim",
      "Block_Number",
      "Block_Start_X_Index",
      "Block_Start_Y_Index",
      "Time",
      "Camera_Dim",
      "Mixture_Dim",
      "Spectral_AOD_Scaling_Coeff_Dim",
      "Latitude",
      "Longitude",
      "Elevation",
      "Year",
      "Day_Of_Year",
      "Month",
      "Day",
      "Hour",
      "Minute",
      "Land_Water_Retrieval_Type",
      "Aerosol_Optical_Depth",
      "Aerosol_Optical_Depth_Uncertainty",
      "Angstrom_Exponent_550_860nm",
      "Spectral_AOD_Scaling_Coeff",
      "Absorption_Aerosol_Optical_Depth",
      "Nonspherical_Aerosol_Optical_Depth",
      "Small_Mode_Aerosol_Optical_Depth",
      "Medium_Mode_Aerosol_Optical_Depth",
      "Large_Mode_Aerosol_Optical_Depth",
      "AUXILIARY/Land_Water_Retrieval_Type_Raw",
      "AUXILIARY/Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Aerosol_Optical_Depth_Uncertainty_Raw",
      "AUXILIARY/Angstrom_Exponent_550_860nm_Raw",
      "AUXILIARY/Spectral_AOD_Scaling_Coeff_Raw",
      "AUXILIARY/Absorption_Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Nonspherical_Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Small_Mode_Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Medium_Mode_Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Large_Mode_Aerosol_Optical_Depth_Raw",
      "AUXILIARY/Single_Scattering_Albedo_446nm_Raw",
      "AUXILIARY/Single_Scattering_Albedo_558nm_Raw",
      "AUXILIARY/Single_Scattering_Albedo_672nm_Raw",
      "AUXILIARY/Single_Scattering_Albedo_867nm_Raw",
      "AUXILIARY/Aerosol_Retrieval_Confidence_Index",
      "AUXILIARY/Aerosol_Optical_Depth_Per_Mixture",
      "AUXILIARY/Minimum_Chisq_Per_Mixture",
      "AUXILIARY/Legacy_Aerosol_Retrieval_Success_Flag_Per_Mixture",
      "AUXILIARY/Cloud_Screening_Parameter",
      "AUXILIARY/Cloud_Screening_Parameter_Neighbor_3x3",
      "AUXILIARY/Aerosol_Retrieval_Screening_Flags",
      "AUXILIARY/Column_Ozone_Climatology",
      "AUXILIARY/Ocean_Surface_Wind_Speed_Climatology",
      "AUXILIARY/Ocean_Surface_Wind_Speed_Retrieved",
      "AUXILIARY/Rayleigh_Optical_Depth",
      "AUXILIARY/Lowest_Residual_Mixture",
      "GEOMETRY/Solar_Zenith_Angle",
      "GEOMETRY/Solar_Azimuth_Angle",
      "GEOMETRY/View_Zenith_Angle",
      "GEOMETRY/View_Azimuth_Angle",
      "GEOMETRY/Scattering_Angle",
      "GEOMETRY/Glint_Angle"
    };
    
    data_ok = MTK_TRUE;
    strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
    strcpy(gridname, "4.4_KM_PRODUCTS");

    status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,&fieldlist);
    if (status == MTK_SUCCESS) {
      if (num_fields != sizeof(fieldlist_expected) /
          sizeof(*fieldlist_expected))
        data_ok = MTK_FALSE;
        
      for (i = 0; i < num_fields; ++i) {
        if (strcmp(fieldlist[i],fieldlist_expected[i]) != 0) {
          data_ok = MTK_FALSE;
          break; 
        }
      }
      MtkStringListFree(num_fields, &fieldlist);
    }

    if (status == MTK_SUCCESS && data_ok) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  status = MtkFileGridToNativeFieldList(filename,NULL,&num_fields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToNativeFieldList(filename,gridname,NULL,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToNativeFieldList(filename,gridname,&num_fields,NULL);
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
