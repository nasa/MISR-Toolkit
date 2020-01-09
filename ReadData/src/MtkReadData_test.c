/*===========================================================================
=                                                                           =
=                             MtkReadData_test                              =
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
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

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
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkReadData");

  /* Normal test call */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 110.0;
  lon_extent = 110.0;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
			     lat_extent,
			     lon_extent,
			     "kilometers",
			     &region);

  status = MtkReadData(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS) {
    /*
      {
      int l;
      int s;
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  if (dbuf.data.i32[l][s] < 0) printf(".");
	  else printf("*"); 
	}
	printf("\n");
      }
      printf("%d,%d\n",dbuf.nline,dbuf.nsample);
      }
      */
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 110.0;
  lon_extent = 110.0;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_PP_P037_AN_22.hdf");
  strcpy(gridname, "Projection_Parameters");
  strcpy(fieldname, "Line");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
			     lat_extent,
			     lon_extent,
			     "kilometers",
			     &region);

  status = MtkReadData(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS) {
    /*
      {
      int l;
      int s;
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  if (dbuf.data.i32[l][s] < 0) printf(".");
	  else printf("*"); 
	}
	printf("\n");
      }
      printf("%d,%d\n",dbuf.nline,dbuf.nsample);
      }
      */
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100.0;
  lon_extent = 1100.0;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "GreenBand");
  strcpy(fieldname, "Green Radiance/RDQI");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
			     lat_extent,
			     lon_extent,
			     "kilometers",
			     &region);

  status = MtkReadData(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS) {
    /*
      {
      int l;
      int s;
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  if (dbuf.data.i32[l][s] < 0) printf(".");
	  else printf("*"); 
	}
	printf("\n");
      }
      printf("%d,%d\n",dbuf.nline,dbuf.nsample);
      }
      */
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 29.95;
  ctr_lon_dd = -117.61;
  lat_extent = 17600;
  lon_extent = 17600;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Latitude");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
                             lat_extent,
                             lon_extent,
                             "meters",
                             &region);

  status = MtkReadData(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }


  /* Failure test call */
  status = MtkReadData(NULL, gridname, fieldname, region, &dbuf, &mapinfo);
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
