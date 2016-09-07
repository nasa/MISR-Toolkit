/*===========================================================================
=                                                                           =
=                           MtkReadL2TCCloud_test                           =
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
  char filename[200];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkReadL2TCCloud");

  MtkSetRegionByPathBlockRange(110, 45, 45, &region);

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_1.1_km");
  strcpy(fieldname, "CloudTopHeight");

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 642) < 1 &&
      (dbuf.data.i16[0][201] - 647) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_1.1_km");
  strcpy(fieldname, "Raw CloudTopHeight");

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 642) < 1 &&
      (dbuf.data.i16[0][201] - 647) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_1.1_km");
  strcpy(fieldname, "Native CloudTopHeight");
  
  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 642) < 1 &&
      (dbuf.data.i16[0][201] - 647) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_WithoutWindCorrection_1.1_km");
  strcpy(fieldname, "CloudMotionCrossTrack_WithoutWindCorrection");

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.f[0][199] - 7.3499999) < .00001 &&
      (dbuf.data.f[0][201] - 7.32999992) < .00001) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_WithoutWindCorrection_1.1_km");
  strcpy(fieldname, "Raw CloudMotionCrossTrack_WithoutWindCorrection");  

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 735) < 1 &&
      (dbuf.data.i16[0][201] - 733) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_WithoutWindCorrection_1.1_km");
  strcpy(fieldname, "Native CloudMotionCrossTrack_WithoutWindCorrection");  
  
  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 735) < 1 &&
      (dbuf.data.i16[0][201] - 733) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_DIAG_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_Prelim_1.1_km");
  strcpy(fieldname, "ERImageContrast_WithoutWindCorrection");

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.f[0][199] - 15.3921738) < .00001 &&
      (dbuf.data.f[0][201] - 17.7010899) < .00001) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_DIAG_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_Prelim_1.1_km");
  strcpy(fieldname, "Raw ERImageContrast_WithoutWindCorrection");  

  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 11873) < 1 &&
      (dbuf.data.i16[0][201] - 12480) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/"
	 "MISR_AM1_TC_CLOUD_DIAG_P110_O074017_F01_0001.hdf");
  strcpy(gridname, "Stereo_Prelim_1.1_km");
  strcpy(fieldname, "Native ERImageContrast_WithoutWindCorrection");  
  
  status = MtkReadL2TCCloud(filename, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_SUCCESS &&
      (dbuf.data.i16[0][199] - 11873) < 1 &&
      (dbuf.data.i16[0][201] - 12480) < 1) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }



  /* Argument check */
  status = MtkReadL2TCCloud(NULL, gridname, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadL2TCCloud(filename, NULL, fieldname, region, &dbuf, &mapinfo);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadL2TCCloud(filename, gridname, NULL, region, &dbuf, &mapinfo);
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
