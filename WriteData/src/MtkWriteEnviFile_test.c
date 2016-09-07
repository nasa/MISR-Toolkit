/*===========================================================================
=                                                                           =
=                          MtkWriteEnviFile_test                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrWriteData.h"
#include "MisrSetRegion.h"
#include "MisrReadData.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main() {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Region region;		/* Region structure */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
				/* Map Info structure */
  double ctr_lat_dd;		/* Center latitude in decimal degrees */
  double ctr_lon_dd;		/* Center longitude in decimal degrees */
  double lat_extent_meters;     /* Latitude extent in meters */
  double lon_extent_meters;     /* Longitude extent in meters */
  char filename[300];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS fieldname */
  char fieldname[80];		/* HDF-EOS fieldname */
  char outfilename[300];	/* Binary filename */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkWriteEnviFile");

  /* Normal test call */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent_meters = 110000.0;
  lon_extent_meters = 110000.0;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");
  strcpy(outfilename, "../Mtk_testdata/out/MISR_AM1_AGP_P037_F01_24");

  MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
			     lat_extent_meters,
			     lon_extent_meters,
			     "meters",
			     &region);
  MtkReadRaw(filename, gridname, fieldname, region, &dbuf, &mapinfo);

  status = MtkWriteEnviFile(outfilename, dbuf, mapinfo,
			    filename, gridname, fieldname);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  MtkDataBufferFree(&dbuf);

  status = MtkWriteEnviFile(NULL, dbuf, mapinfo,
			    filename, gridname, fieldname);
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
