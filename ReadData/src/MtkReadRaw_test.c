/*===========================================================================
=                                                                           =
=                             MtkReadRaw_test                               =
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
  int cn = 0;			    /* Column number */
  MTKt_Region region;		/* Region structure */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				            /* Data buffer structure */
  MTKt_MapInfo mapinfo;		/* Map Info structure */
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */

  MTK_PRINT_STATUS(cn,"Testing MtkReadRaw");

  MtkSetRegionByPathBlockRange(37, 35, 35, &region);

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "GreenBand");
  strcpy(fieldname, "Green Radiance/RDQI");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      dbuf.data.u16[63][79] == 31040 &&
      dbuf.data.u16[63][63] == 65515) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "GreenBand");
  strcpy(fieldname, "Green Radiance/RDQI[4]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1[1]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1[1]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "Makefile");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1[1]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "badgrid");
  strcpy(fieldname, "LAIDelta1");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_INVALID_GRID) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "badfield");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_INVALID_FIELD) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_MISSING_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1[200]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_INVALID_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LAIDelta1[0][1]");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_EXTRA_FIELD_DIMENSION) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadRaw(NULL, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadRaw(filename, NULL, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadRaw(filename, gridname, NULL, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  MtkSetRegionByPathBlockRange(39, 59, 59, &region);
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(gridname, "4.4_KM_PRODUCTS");
  strcpy(fieldname, "Latitude");

  status = MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      dbuf.data.f[0][1] == 40.289031982421875 &&
      dbuf.data.f[1][0] == 40.253570556640625 &&
      dbuf.data.f[31][127] == 38.4030723571777343750) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadRaw(filename, NULL, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadRaw(filename, gridname, NULL, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
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
