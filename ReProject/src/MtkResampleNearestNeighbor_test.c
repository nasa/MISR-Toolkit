/*===========================================================================
=                                                                           =
=                      MtkResampleNearestNeighbor_test                      =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include "MisrSetRegion.h"
#include "MisrMapQuery.h"
#include "MisrReadData.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "MisrWriteData.h"

int main () {

  MTKt_status status;           /* Return status */
  int cn = 0;			/* Column number */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean same = MTK_TRUE; /* Expected comparison status */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
				/* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
				/* Longitude data buffer structure */
  MTKt_DataBuffer line = MTKT_DATABUFFER_INIT;
				/* Line data buffer structure */
  MTKt_DataBuffer sample = MTKT_DATABUFFER_INIT;
				/* Sample data buffer structure */
  int l, lp;			/* Line index */
  int s, sp;			/* Sample index */
  MTKt_Region region = MTKT_REGION_INIT;
				/* Region structure */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
				/* Source data buffer */
  MTKt_DataBuffer destbuf = MTKT_DATABUFFER_INIT;
				/* Destination data buffer */
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
				/* Map info structure */

  MTK_PRINT_STATUS(cn,"Testing MtkResampleNearestNeighbor");

  /* Setup test data */
  MtkSetRegionByPathBlockRange(39, 50, 52, &region);
  MtkReadData("../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf",
	      "Standard", "AveSceneElev", region, &srcbuf, &mapinfo);

  /* Normal test call */
  MtkCreateGeoGrid(51.0, -114.0, 46.0, -106.0, .01, 0.01, &lat, &lon);
  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_SUCCESS) {
    same = MTK_TRUE;
    for (l=0; l < line.nline; l++) {
      for (s=0; s < line.nsample; s++) {
	lp = (int)floorf(line.data.f[l][s] + 0.5);
	sp = (int)floorf(sample.data.f[l][s] + 0.5);
	if (lp >= 0 && lp < destbuf.nline && sp >=0 && sp < destbuf.nsample) {
	  if (destbuf.data.i16[l][s] != srcbuf.data.i16[lp][sp]) 
	    same = MTK_FALSE;
	} else {
	  if (destbuf.data.i16[l][s] != 0)
	    same = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&destbuf);
  } else {
    same = MTK_FALSE;
  }
  if (same) {
    MTK_PRINT_STATUS(cn,".")
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Normal test call */
  MtkCreateLatLon(mapinfo, &lat, &lon);
  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_SUCCESS) {
    same = MTK_TRUE;
    for (l=0; l < line.nline; l++) {
      for (s=0; s < line.nsample; s++) {
	if (destbuf.data.i16[l][s] != srcbuf.data.i16[l][s]) {
	  same = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&destbuf);
  } else {
    same = MTK_FALSE;
  }
  if (same) {
    MTK_PRINT_STATUS(cn,".")
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_float, &line);
  MtkDataBufferAllocate(5, 5, MTKe_float, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_DIMENSION_MISMATCH) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_float, &line);
  MtkDataBufferAllocate(10, 3, MTKe_float, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_DIMENSION_MISMATCH) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_int16, &line);
  MtkDataBufferAllocate(10, 5, MTKe_float, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_DATATYPE_MISMATCH) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_float, &line);
  MtkDataBufferAllocate(10, 5, MTKe_double, &sample);

  status = MtkResampleNearestNeighbor(srcbuf, line, sample, &destbuf);
  if (status == MTK_DATATYPE_MISMATCH) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);

  MtkDataBufferFree(&srcbuf);

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
