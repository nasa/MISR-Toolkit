/*===========================================================================
=                                                                           =
=                             MtkReadConv_test                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReadData.h"
#include "MisrFileQuery.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Region region;		/* Region structure */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				/* Data buffer structure */
  MTKt_MapInfo mapinfo;		/* Map Info structure */
  double ctr_lat_dd;		/* Center latitude in decimal degrees */
  double ctr_lon_dd;		/* Center longitude in decimal degrees */
  double lat_extent;		/* Latitude extent */
  double lon_extent;		/* Longitude extent */
  char filename[200];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int path;			/* Path */
  int sb;			/* Start block */
  int eb;			/* End block */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkReadConv");

  /* Normal test call */
  ctr_lat_dd = 22.5;
  ctr_lon_dd = 28.5;
  lat_extent = 11000.0;
  lon_extent = 11000.0;

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P177_F01_24_conv.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
			     lat_extent,
			     lon_extent,
			     "meters",
			     &region);

  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      dbuf.data.i16[7][7] == 291) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  /* Test for when region is larger than file (sb-1) */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020_conv.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Radiance/RDQI");

  MtkFileToPath(filename, &path);
  MtkFileToBlockRange(filename, &sb, &eb);
  MtkSetRegionByPathBlockRange(path, sb-1, eb, &region);

  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      dbuf.data.u16[1000][347] == 65515 &&
      dbuf.data.u16[1000][348] == 21752) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  /* Test for MISR HR L2 product file which has non-standard dimension order */
  strcpy(filename, "../Mtk_testdata/in/MISR_HR_TIP_20121003_P168_O068050_B110_V2.00-0_GRN.hdf");
  strcpy(gridname, "TIP");
  strcpy(fieldname, "ObsCovarFluxes[0]");

  MtkFileToPath(filename, &path);
  MtkFileToBlockRange(filename, &sb, &eb);
  MtkSetRegionByPathBlockRange(path, sb, eb, &region);

  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      fabs(dbuf.data.f[350][350] - 0.0035493) < 0.000001) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
	MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test calls */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
   
  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  
  
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020_conv.hdf");
  strcpy(gridname, "abcd");
   
  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_HDFEOS_GDATTACH_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  
  
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_LM_P177_O004194_BA_SITE_EGYPTDESERT_F02_0020_conv.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "abcd");
  
  status = MtkReadConv(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_HDFEOS_GDFIELDINFO_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  

  /* Argument checks */
  status = MtkReadConv(NULL, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadConv(filename, NULL, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadConv(filename, gridname, NULL, region, &dbuf, &mapinfo);
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
