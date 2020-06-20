/*===========================================================================
=                                                                           =
=                            MtkCreateGeoGrid                               =
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
#include "MisrError.h"
#include "MisrUtil.h"

/** \brief Creates a regularly spaced geographic 2-D grid consisting of a latitude buffer and a longitude buffer in decimal degrees, given upper left and lower right latitude/longitude coordinates and latitude/longitude cell size.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we create a regularly spaced geographic 2-D grid of lat/lon spanning the upper left latitude 40.0, longitude -120.0 and lower right latitude 30.0, longitude -110.0 with a cell size of 0.25 degrees in both lat/lon.
 *
 *  \code
 *  status = MtkCreateGeoGrid(40.0, -120.0, 30.0, -110.0, 0.25, 0.25, &latbuf, &lonbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by latbuf and lonbuf
*/

MTKt_status MtkCreateGeoGrid(
  double ulc_lat_dd,       /**< [IN] Upper left corner latitude */
  double ulc_lon_dd,       /**< [IN] Upper left corner longitude */
  double lrc_lat_dd,       /**< [IN] Lower right corner latitude */
  double lrc_lon_dd,       /**< [IN] Lower right corner longitude */
  double lat_cellsize_dd,  /**< [IN] Latitude cell size in decimal degrees */
  double lon_cellsize_dd,  /**< [IN] Longitude cell size in decimal degrees */
  MTKt_DataBuffer *latbuf, /**< [OUT] Latitude data buffer */
  MTKt_DataBuffer *lonbuf  /**< [OUT] Longitue data buffer */ )
{
  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
                                /* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
                                /* Longitude data buffer structure */
  double lat_extent_dd;         /* Latitude extent */
  double lon_extent_dd;         /* Longitude extent */
  double lontmp;		/* Temporary longitude */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int l;			/* Line index */
  int s;			/* Sample index */

  /* Check latitude bounds */
  if (ulc_lat_dd > 90.0 || ulc_lat_dd < -90.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lrc_lat_dd > 90.0 || lrc_lat_dd < -90.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (ulc_lat_dd < lrc_lat_dd)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check longitude bounds */
  if (ulc_lon_dd > 180.0 || ulc_lon_dd < -180.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lrc_lon_dd > 180.0 || lrc_lon_dd < -180.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check latitude cell size bounds */
  if (lat_cellsize_dd <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check longitude cell size bounds */
  if (lon_cellsize_dd <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check for date line crossing */
  lat_extent_dd = ulc_lat_dd - lrc_lat_dd;
  if (ulc_lon_dd > lrc_lon_dd) {
    lon_extent_dd = 360 + lrc_lon_dd - ulc_lon_dd;      /* Date line crossed */
  } else {
    lon_extent_dd = lrc_lon_dd - ulc_lon_dd;
  }

  nline = (int) (lat_extent_dd / lat_cellsize_dd) + 1;
  nsample = (int) (lon_extent_dd / lon_cellsize_dd) + 1;

  /* --------------------------------------------------- */
  /* Allocate buffers of the latitude and longitude data */
  /* --------------------------------------------------- */

  status = MtkDataBufferAllocate(nline, nsample, MTKe_double, &lat);
  MTK_ERR_COND_JUMP(status);
  status = MtkDataBufferAllocate(nline, nsample, MTKe_double, &lon);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------ */
  /* Compute Geographic Grid  */
  /* ------------------------ */

  if (ulc_lon_dd > lrc_lon_dd) {
    for (l=0; l < nline; l++) {	                        /* Date line crossed */
      for (s=0; s < nsample; s++) {
	lat.data.d[l][s] = ulc_lat_dd - lat_cellsize_dd * l;
	lontmp =  ulc_lon_dd + lon_cellsize_dd * s;
	lon.data.d[l][s] = lontmp > 180 ? lontmp - 360.0 : lontmp;
      }
    }
  } else {
    for (l=0; l < nline; l++) {
      for (s=0; s < nsample; s++) {
	lat.data.d[l][s] = ulc_lat_dd - lat_cellsize_dd * l;
	lon.data.d[l][s] = ulc_lon_dd + lon_cellsize_dd * s;
      }
    }
  }

  *latbuf = lat;
  *lonbuf = lon;

  return MTK_SUCCESS;
ERROR_HANDLE:
  MtkDataBufferFree(&lat);
  MtkDataBufferFree(&lon);
  return status_code;
}
