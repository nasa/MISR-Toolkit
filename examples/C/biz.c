/*===========================================================================
=                                                                           =
=                                  biz                                      =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _MSC_VER // Visual Studio '08 does not include round (part of C99) in math.h
static double round(double val)
{    
    return floor(val + 0.5);
}
#endif


int biz( ) {

  MTKt_status status;		/* For routine calls */
  MTKt_status status_code;	/* For error handler */
  char *err_msg[] = MTK_ERR_DESC; /* Error message descriptions */

  double lat_dd = 32.2, lon_dd = -114.5;
  double lat, lon;
  double lat_extent = 200, lon_extent = 100;
  double somx, somy;
  int path, res;
  float line, sample;
  int block;
  float blk_line, blk_sample;
  int l, s;
  int latdeg, londeg, latmin, lonmin;
  double latsec, lonsec;
  MTKt_Region region = MTKT_REGION_INIT;
  MTKt_TimeMetaData timemeta = MTKT_TIME_METADATA_INIT;
  char datetime[MTKd_DATETIME_LEN];
  char testroot[200];
  char filename[200];
  char gridname[200];
  char fieldname[200];
  MTKt_DataBuffer brf_databuf = MTKT_DATABUFFER_INIT;
  MTKt_MapInfo brf_mapinfo = MTKT_MAPINFO_INIT;
  MTKt_DataBuffer hdrf_databuf = MTKT_DATABUFFER_INIT;
  MTKt_MapInfo hdrf_mapinfo = MTKT_MAPINFO_INIT;

  /* ---------------------------------------------------------- */
  /* Set a file root from prompt or environment variable        */
  /* ---------------------------------------------------------- */
  if( getenv("MTKHOME") ) {
	  strcpy(testroot, getenv("MTKHOME"));
	  strcat(testroot, "/../Mtk_testdata");
  } else {
	  printf("MTKHOME environment variable not found.\n" \
               "Please enter the path to Mtk_testdata directory" \
               "(e.g. C:\\Mtk_testdata or /tmp/Mtk_testdata ):");
	  scanf("%s",testroot);
  }

  /* ---------------------------------------------------------- */
  /* Set a region given center latitude/longitude and an extent */
  /* ---------------------------------------------------------- */

  status = MtkSetRegionByLatLonExtent(lat_dd, lon_dd,
				      lat_extent, lon_extent, "km", &region);
  MTK_ERR_COND_JUMP(status);

  printf("region center lat/lon (dd) = (%f, %f)\n",
	 region.geo.ctr.lat, region.geo.ctr.lon);
  printf("         region extent (m) = (%f, %f)\n",
	      region.hextent.xlat * 2.0, region.hextent.ylon * 2.0);

  /* --------------------------------------------------------- */
  /* Read data in the region from filename/gridname/fieldname, */
  /* do some coordinate conversions, query data buffer and     */
  /* get pixel time                                            */
  /* --------------------------------------------------------- */

  strcpy(filename,testroot);
  strcat(filename, "/in/");
  strcat(filename, "MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Brf");

  status = MtkReadData(filename, gridname, fieldname, region,
		       &brf_databuf, &brf_mapinfo);
  MTK_ERR_COND_JUMP(status);

  status = MtkLatLonToLS(brf_mapinfo, lat_dd, lon_dd, &line, &sample);
  MTK_ERR_COND_JUMP(status);

  l = (int)round(line);
  s = (int)round(sample);

  status = MtkFileToPath(filename, &path);
  MTK_ERR_COND_JUMP(status);

  status = MtkFileGridToResolution(filename, gridname, &res);
  MTK_ERR_COND_JUMP(status);

  status = MtkLatLonToBls(path, res, lat_dd, lon_dd,
			  &block, &blk_line, &blk_sample);
  MTK_ERR_COND_JUMP(status);

  status = MtkLSToLatLon(brf_mapinfo, line, sample, &lat, &lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkDdToDegMinSec(lat, &latdeg, &latmin, &latsec);
  MTK_ERR_COND_JUMP(status);

  status = MtkDdToDegMinSec(lon, &londeg, &lonmin, &lonsec);
  MTK_ERR_COND_JUMP(status);

  status = MtkLSToSomXY(brf_mapinfo, line, sample, &somx, &somy);
  MTK_ERR_COND_JUMP(status);

  status = MtkTimeMetaRead(filename, &timemeta);
  MTK_ERR_COND_JUMP(status);

  status = MtkPixelTime(timemeta, somx, somy, datetime);
  MTK_ERR_COND_JUMP(status);

  printf("\n              Reading file = %s\n",filename);
  printf("              Reading grid = %s\n",gridname);
  printf("             Reading field = %s\n",fieldname);
  printf("   number of lines/samples = (%d, %d)\n",
	 brf_mapinfo.nline, brf_mapinfo.nsample);
  printf("        input lat/lon (dd) = (%f, %f)\n", lat_dd, lon_dd);
  printf("               line/sample = (%f, %f)\n", line, sample);
  printf("             Brf[%d][%d] = %f\n", l, s, brf_databuf.data.f[l][s]), 
  printf("         block/line/sample = (%d, %f, %f)\n",
	 block, blk_line, blk_sample);
  printf("              lat/lon (dd) = (%f, %f)\n", lat, lon);
  printf("             lat/lon (dms) = (%d:%d:%f, %d:%d:%f)\n",
	 latdeg, latmin, latsec, londeg, lonmin, lonsec);
  printf("                   SOM x/y = (%f, %f)\n", somx, somy);
  printf("      pixel time[%d][%d] = %s\n", l, s, datetime);

  MtkDataBufferFree(&brf_databuf);

  /* --------------------------------------------------------- */
  /* Read data in the region from filename/gridname/fieldname  */
  /* with extra dimensions, do some coordinate conversions and */
  /* query data buffer                                         */
  /* --------------------------------------------------------- */

  strcpy(filename,testroot);
  strcat(filename, "/in/");
  strcat(filename, "MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandHDRF[3][4]");

  status = MtkReadData(filename, gridname, fieldname, region,
		       &hdrf_databuf, &hdrf_mapinfo);
  MTK_ERR_COND_JUMP(status);

  status = MtkLatLonToLS(hdrf_mapinfo, lat_dd, lon_dd, &line, &sample);
  MTK_ERR_COND_JUMP(status);

  l = (int)round(line);
  s = (int)round(sample);

  status = MtkFileToPath(filename, &path);
  MTK_ERR_COND_JUMP(status);

  status = MtkFileGridToResolution(filename, gridname, &res);
  MTK_ERR_COND_JUMP(status);

  status = MtkLatLonToBls(path, res, lat_dd, lon_dd,
			  &block, &blk_line, &blk_sample);
  MTK_ERR_COND_JUMP(status);

  status = MtkLSToLatLon(hdrf_mapinfo, line, sample, &lat, &lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkDdToDegMinSec(lat, &latdeg, &latmin, &latsec);
  MTK_ERR_COND_JUMP(status);

  status = MtkDdToDegMinSec(lon, &londeg, &lonmin, &lonsec);
  MTK_ERR_COND_JUMP(status);

  status = MtkLatLonToSomXY(path, lat_dd, lon_dd, &somx, &somy);
  MTK_ERR_COND_JUMP(status);

  printf("\n              Reading file = %s\n",filename);
  printf("              Reading grid = %s\n",gridname);
  printf("             Reading field = %s\n",fieldname);
  printf("   number of lines/samples = (%d, %d)\n",
	 hdrf_mapinfo.nline, hdrf_mapinfo.nsample);
  printf("        input lat/lon (dd) = (%f, %f)\n", lat_dd, lon_dd);
  printf("               line/sample = (%f, %f)\n", line, sample);
  printf("              HDRF[%d][%d] = %f\n", l, s, hdrf_databuf.data.f[l][s]), 
  printf("         block/line/sample = (%d, %f, %f)\n",
	 block, blk_line, blk_sample);
  printf("              lat/lon (dd) = (%f, %f)\n", lat, lon);
  printf("             lat/lon (dms) = (%d:%d:%f, %d:%d:%f)\n",
	 latdeg, latmin, latsec, londeg, lonmin, lonsec);
  printf("                   SOM x/y = (%f, %f)\n", somx, somy);

  MtkDataBufferFree(&hdrf_databuf);

  return MTK_SUCCESS;
 ERROR_HANDLE:
  printf("Error: %s\n",err_msg[status_code]);
  MtkDataBufferFree(&hdrf_databuf);
  return status_code;
}
