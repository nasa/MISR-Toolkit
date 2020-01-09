/*===========================================================================
=                                                                           =
=                        MtkTransformCoordinates_test                       =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  int cn = 0;			/* Column number */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean same = MTK_TRUE;	/* Expected comparison status */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
				/* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
				/* Longitude data buffer structure */
  MTKt_DataBuffer line = MTKT_DATABUFFER_INIT;
				/* Line data buffer structure */
  MTKt_DataBuffer sample = MTKT_DATABUFFER_INIT;
				/* Sample data buffer structure */
  int l;			/* Line index */
  int s;			/* Sample index */
  float line_expected;		/* Expected line value */
  float sample_expected;	/* Expected sample value */
  MTKt_Region region = MTKT_REGION_INIT;
				/* Region structure */
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
				/* Map info structure */

  MTK_PRINT_STATUS(cn,"Testing MtkTransformCoordinates");

  /* Normal test call */
  MtkSetRegionByPathBlockRange(39, 51, 52, &region);
  MtkSnapToGrid(39, 1100, region, &mapinfo);
  MtkCreateLatLon(mapinfo, &lat, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_SUCCESS) {
    same = MTK_TRUE;
    for (l=0; l < line.nline; l++) {
      for (s=0; s < line.nsample; s++) {
	if (fabs(lat.data.d[l][s] - -200.0) < .000001 &&
	    fabs(lon.data.d[l][s] - -200.0) < .000001) {
	  if (fabs(line.data.f[l][s] - -1.0) > .000001 ||
	      fabs(sample.data.f[l][s] - -1.0) > .000001) {
	    same = MTK_FALSE;
	  }
	} else {
	  if (fabs(line.data.f[l][s] - l) > .000001 ||
	      fabs(sample.data.f[l][s] - s) > .000001) {
	    same = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
  } else {
    same = MTK_FALSE;
  }
  if (same) {
    MTK_PRINT_STATUS(cn,".")
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Normal test call */
  MtkSetRegionByPathBlockRange(39, 51, 52, &region);
  MtkSnapToGrid(39, 275, region, &mapinfo);
  // MtkCreateGeoGrid(51.0, -114.0, 46.0, -106.0, 0.02, 0.02, &lat, &lon);
  MtkCreateGeoGrid(49.0, -113.0, 47.5, -114.0, 0.02, 0.02, &lat, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_SUCCESS) {
    same = MTK_TRUE;
    for (l=0; l < line.nline; l++) {
      for (s=0; s < line.nsample; s++) {
	MtkLatLonToLS(mapinfo, lat.data.d[l][s], lon.data.d[l][s],
		      &line_expected, &sample_expected);
	if (fabs(line.data.f[l][s] - line_expected) > .000001 ||
	    fabs(sample.data.f[l][s] - sample_expected) > .000001) {
	  same = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
  } else {
    same = MTK_FALSE;
  }
  if (same) {
    MTK_PRINT_STATUS(cn,".")
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_double, &lat);
  MtkDataBufferAllocate(5, 5, MTKe_double, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_DIMENSION_MISMATCH) {
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_double, &lat);
  MtkDataBufferAllocate(10, 3, MTKe_double, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_DIMENSION_MISMATCH) {
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_float, &lat);
  MtkDataBufferAllocate(10, 5, MTKe_double, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_DATATYPE_MISMATCH) {
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  /* Failure test call */
  MtkDataBufferAllocate(10, 5, MTKe_double, &lat);
  MtkDataBufferAllocate(10, 5, MTKe_int16, &lon);

  status = MtkTransformCoordinates(mapinfo, lat, lon, &line, &sample);
  if (status == MTK_DATATYPE_MISMATCH) {
    MtkDataBufferFree(&line);
    MtkDataBufferFree(&sample);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
