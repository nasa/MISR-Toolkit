/*===========================================================================
=                                                                           =
=                              MtkCreateLatLon                              =
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
#include "MisrProjParam.h"
#include "MisrMapQuery.h"
#include "MisrCoordQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"

/** \brief Creates a 2-D latitude buffer and a 2-D longitude buffer in decimal degrees corresponding to the data plane described by the mapinfo argument
 *         given a mapinfo structure.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we define a region centered at lat 35.0, lon -115.0 with a lat extent of 110.0km and lon extent 110.0km. We then read the Regional Best Estimate Spectral Optical Depth and create the corresponding latutide and longitude buffers for this dataplane described by mapinfo. This routine computes the latitude and longitude for every pixel in the dataplane for which there is MISR data. It does not preclude the use of the MapQuery routines MtkLSToLatLon or MtkLatLonToLS that compute latitude and longitude.  In fact this routine makes use of those routines.
 *
 *  \code
 *   MTKt_Region region = MTKT_REGION_INIT;
 *   MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;
 *   MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
 *   MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;
 *   MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
 *   status = MtkSetRegionByLatLonExtent(35.0, -115.0, 110.0, 110.0, "km", &region);
 *   status = MtkReadData("MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf", "RegParamsAer", "RegBestEstimateSpectralOptDepth", 
 *                        region, &databuf, &mapinfo);
 *   status = MtkCreateLatLon(mapinfo, &latbuf, &lonbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by databuf, latbuf and lonbuf
 */

MTKt_status MtkCreateLatLon(
  MTKt_MapInfo mapinfo,     /**< [IN] Mapinfo */
  MTKt_DataBuffer *latbuf,  /**< [OUT] Latitude data buffer */
  MTKt_DataBuffer *lonbuf   /**< [OUT] Longitue data buffer */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
                                /* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
                                /* Longitude data buffer structure */
  float line;			/* Data plane line index */
  float sample;			/* Data plane sample index */
  double somx;			/* SOM X */
  double somy;			/* SOM Y */
  int b;                        /* Block index */
  int l;                        /* Block line index */
  int s;                        /* Block sample index */
  int lp;                       /* Data plane integer line index */
  int sp;                       /* Data plane integer sample index */

  /* --------------------------------------------------- */
  /* Allocate buffers of the latitude and longitude data */
  /* --------------------------------------------------- */

  status = MtkDataBufferAllocate(mapinfo.nline, mapinfo.nsample,
				 MTKe_double, &lat);
  MTK_ERR_COND_JUMP(status);
  status = MtkDataBufferAllocate(mapinfo.nline, mapinfo.nsample,
				 MTKe_double, &lon);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------ */
  /* Fill with a default fill value */
  /* ------------------------------ */

  for (l = 0; l < mapinfo.nline; l++) {
    for (s = 0; s < mapinfo.nsample; s++) {
      lat.data.d[l][s] = -200.0;
      lon.data.d[l][s] = -200.0;
    }
  }

  /* --------------------------------------------------------------- */
  /* Compute the lat/lon for each block/line/sample in the swath     */
  /* that intersect with the data plane defined my mapinfo           */
  /* --------------------------------------------------------------- */

  for (b = mapinfo.start_block; b <= mapinfo.end_block; b++) {
    for (l = 0; l < mapinfo.pp.nline; l++) {
      for (s = 0; s < mapinfo.pp.nsample; s++) {
	MtkBlsToSomXY(mapinfo.path, mapinfo.resolution, b, (float)l, (float)s,
		      &somx, &somy);
	MtkSomXYToLS(mapinfo, somx, somy, &line, &sample);
	lp = (int)line;
	sp = (int)sample;
	if (lp >= 0 && lp < mapinfo.nline &&
	    sp >= 0 && sp < mapinfo.nsample) {
	  MtkSomXYToLatLon(mapinfo.path, somx, somy, 
			   &(lat.data.d[lp][sp]), &(lon.data.d[lp][sp]));
	}
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
