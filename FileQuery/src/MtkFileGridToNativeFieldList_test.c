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

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
