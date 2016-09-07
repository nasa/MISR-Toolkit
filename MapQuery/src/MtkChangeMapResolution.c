/*===========================================================================
=                                                                           =
=                          MtkChangeMapResolution                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrMapQuery.h"
#include "MisrUtil.h"
#include "MisrCoordQuery.h"
#include <stdlib.h>
#include <math.h>

/** \brief Change resolution of an MTKt_MapInfo structure.
 *
 *  New resolution must be an integer multiple of the base MISR
 *  resolution, defined by MAXRESOLUTION.
 *
 *  The edges of the output map will be anchored at the same locations
 *  as the input map.  Therefore, the size of input map must evenly divisible
 *  by the new resolution.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkChangeMapResolution()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkChangeMapResolution(
  const MTKt_MapInfo *Map_info_in,  /**< [IN] Input map information */
  int New_resolution,            /**< [IN] New resolution. */
  MTKt_MapInfo *Map_info_out  /**< [OUT] Output map information. */
)
{
  MTKt_status status;           /* Return status of called routines. */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_MapInfo map_info_out_tmp = MTKT_MAPINFO_INIT;
				/* Map information */
  int size_x;			/* Size of map in meters. */
  int size_y;			/* Size of map in meters. */
  double center_offset; 	/* Offset to shift pixel center at corners. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info_in = NULL                                 */
  /*                 Map_info_in->nline < 1                             */
  /*                 Map_info_in->nsample < 1                           */
  /*                 Map_info_in->resolution < 1                        */
  /* ------------------------------------------------------------------ */

  if (Map_info_in == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info_in");
  }
  if (Map_info_in->nline < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Map_info_in->nline < 1");
  }
  if (Map_info_in->nsample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Map_info_in->nsample < 1");
  }
  if (Map_info_in->resolution < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Map_info_in->resolution < 1");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   New_resolution < 1                                               */
  /*   New_resolution % MAXRESOLUTION != 0                              */
  /* ------------------------------------------------------------------ */

  if (New_resolution < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"New_resolution < 1");
  }
  if (New_resolution % MAXRESOLUTION != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"New_resolution % MAXRESOLUTION != 0");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info_out = NULL                                */
  /* ------------------------------------------------------------------ */

  if (Map_info_out == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info_out == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Initialize output map info to value of the input map info.         */
  /* ------------------------------------------------------------------ */

  map_info_out_tmp = *Map_info_in;

  /* ------------------------------------------------------------------ */
  /* Calculate the size of the ouput map.                               */
  /*                                                                    */
  /* Argument check: size_x % New_resolution != 0                       */
  /*                 size_y % New_resolution != 0                       */
  /* ------------------------------------------------------------------ */

  size_x = ( Map_info_in->nline * Map_info_in->resolution );
  size_y = ( Map_info_in->nsample * Map_info_in->resolution );

  if (size_x % New_resolution != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"size_x % New_resolution != 0");
  }
  if (size_y % New_resolution != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"size_y % New_resolution != 0");
  }
  
  map_info_out_tmp.nline = size_x / New_resolution;
  map_info_out_tmp.nsample = size_y / New_resolution;

  /* ------------------------------------------------------------------ */
  /* Calculate output map resolution factor.                            */
  /* ------------------------------------------------------------------ */

  map_info_out_tmp.resfactor = New_resolution / MAXRESOLUTION;

  /* ------------------------------------------------------------------ */
  /* Set output map resolution.                                         */
  /* ------------------------------------------------------------------ */

  map_info_out_tmp.resolution = New_resolution;

  /* ------------------------------------------------------------------ */
  /* Recalculate pixel centers at map corners.                          */
  /* ------------------------------------------------------------------ */

  center_offset = (Map_info_in->resolution - New_resolution) / (double)2.0;
  map_info_out_tmp.som.ulc.x -= center_offset;
  map_info_out_tmp.som.ulc.y -= center_offset;
  map_info_out_tmp.som.lrc.x += center_offset;
  map_info_out_tmp.som.lrc.y += center_offset;
  
  status = MtkSomXYToLatLon(map_info_out_tmp.path,
			    map_info_out_tmp.som.ulc.x,
			    map_info_out_tmp.som.ulc.y,
			    &map_info_out_tmp.geo.ulc.lat,
			    &map_info_out_tmp.geo.ulc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map_info_out_tmp.path,
			    map_info_out_tmp.som.lrc.x,
			    map_info_out_tmp.som.lrc.y,
			    &map_info_out_tmp.geo.lrc.lat,
			    &map_info_out_tmp.geo.lrc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map_info_out_tmp.path,
			    map_info_out_tmp.som.ulc.x,
			    map_info_out_tmp.som.lrc.y,
			    &map_info_out_tmp.geo.urc.lat,
			    &map_info_out_tmp.geo.urc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map_info_out_tmp.path,
			    map_info_out_tmp.som.lrc.x,
			    map_info_out_tmp.som.ulc.y,
			    &map_info_out_tmp.geo.llc.lat,
			    &map_info_out_tmp.geo.llc.lon);
  MTK_ERR_COND_JUMP(status);

  /* --------------------------------------------------------------------- */
  /* Set the projection parameters in mapinfo for this path and resolution */
  /* --------------------------------------------------------------------- */

  status = MtkPathToProjParam(map_info_out_tmp.path, New_resolution, &(map_info_out_tmp.pp));
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Map_info_out = map_info_out_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}


