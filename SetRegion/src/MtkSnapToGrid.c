/*===========================================================================
=                                                                           =
=                              MtkSnapToGrid                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrSetRegion.h"
#include "MisrCoordQuery.h"
#include "MisrMapQuery.h"
#include "MisrError.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>

/** \brief Snap a region to a MISR grid based on path number and resolution
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we snap a region to a SOM plane for path 37 at resolution 1100 meters.
 *
 *  \code
 *  status = MtkSnapToGrid(37, 1100, region, &mapinfo);
 *  \endcode
 *
 *  \par Special Note:
 *  Typically this function is not called directly.  Instead use MtkReadData().
 */

MTKt_status MtkSnapToGrid(
  int Path,             /**< [IN] Path */
  int Resolution,       /**< [IN] Resolution */
  MTKt_Region Region,   /**< [IN] Region */
  MTKt_MapInfo *Map_info /**< [OUT] Map Info */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
				/* Map info structure */
  MTKt_SomCoord som_min;	/* Som coordinate */
  MTKt_SomCoord som_max;	/* Som coordinate */
  MTKt_SomCoord corner;		/* Som coordinate */
  MTKt_SomCoord center;		/* Som coordinate */

  /* ------------------------------------------------------------------ */
  /* Argument check: mapinfo == NULL                                    */
  /* ------------------------------------------------------------------ */
  
  if (Map_info == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* ----------------------- */
  /* Set path and resolution */
  /* ----------------------- */

  map.path = Path;
  map.som.path = Path;
  map.resolution = Resolution;
  map.resfactor = map.resolution / MAXRESOLUTION;
  map.pixelcenter = MTK_TRUE;

  /* -------------------------------------------------------------- */
  /* Snap center geo coordinate of region to som grid for this path */
  /* -------------------------------------------------------------- */
  
  status = MtkLatLonToSomXY(Path, Region.geo.ctr.lat, Region.geo.ctr.lon,
			    &center.x, &center.y);
  MTK_ERR_COND_JUMP(status);

  som_min.x = center.x - Region.hextent.xlat;
  som_min.y = center.y - Region.hextent.ylon;
  som_max.x = center.x + Region.hextent.xlat;
  som_max.y = center.y + Region.hextent.ylon;

  status = MtkBlsToSomXY(Path, MINRESOLUTION, 1, 0, 0, &corner.x, &corner.y);
  MTK_ERR_COND_JUMP(status);

  corner.x -= MINRESOLUTION / (double)2.0;
  corner.y -= MINRESOLUTION / (double)2.0;
  som_min.x = (floor((som_min.x - corner.x) / MINRESOLUTION)) * MINRESOLUTION + corner.x;
  som_min.y = (floor((som_min.y - corner.y) / MINRESOLUTION)) * MINRESOLUTION + corner.y;
  som_max.x = (ceil((som_max.x - corner.x) / MINRESOLUTION)) * MINRESOLUTION + corner.x;
  som_max.y = (ceil((som_max.y - corner.y) / MINRESOLUTION)) * MINRESOLUTION + corner.y;
  
  map.som.ctr.x = (som_max.x + som_min.x) / 2.0;
  map.som.ctr.y = (som_max.y + som_min.y) / 2.0;

  map.nline = (int) (som_max.x - som_min.x) / map.resolution;
  map.nsample = (int) (som_max.y - som_min.y) / map.resolution;

  /* ------------------------- */
  /* Compute upper left corner */
  /* ------------------------- */

  map.som.ulc.x = som_min.x + (map.resolution / (double)2.0);
  map.som.ulc.y = som_min.y + (map.resolution / (double)2.0);

  /* --------------------------- */
  /* Compute lower right corner  */
  /* --------------------------- */

  map.som.lrc.x = som_max.x - (map.resolution / (double)2.0);
  map.som.lrc.y = som_max.y - (map.resolution / (double)2.0);

  /* ------------------------------------------- */
  /* Compute geographic coordinates for map area */
  /* ------------------------------------------- */

  status = MtkSomXYToLatLon(map.som.path, map.som.ulc.x, map.som.ulc.y,
			    &map.geo.ulc.lat, &map.geo.ulc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map.som.path, map.som.ulc.x, map.som.lrc.y,
			    &map.geo.urc.lat, &map.geo.urc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map.som.path, map.som.ctr.x, map.som.ctr.y,
			    &map.geo.ctr.lat, &map.geo.ctr.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map.som.path, map.som.lrc.x, map.som.lrc.y,
			    &map.geo.lrc.lat, &map.geo.lrc.lon);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(map.som.path, map.som.lrc.x, map.som.ulc.y,
			    &map.geo.llc.lat, &map.geo.llc.lon);
  MTK_ERR_COND_JUMP(status);

  /* --------------------------------------------------------------------- */
  /* Set the projection parameters in mapinfo for this path and resolution */
  /* --------------------------------------------------------------------- */

  status = MtkPathToProjParam(Path, Resolution, &(map.pp));
  MTK_ERR_COND_JUMP(status);

  /* ----------------------------------------------------- */
  /* Determine start block and end block of this map plane */
  /* ----------------------------------------------------- */

  map.start_block = (int)((map.som.ulc.x - map.pp.ulc[0])/
			  (map.pp.nline * map.pp.resolution)) + 1;
  map.end_block = (int)((map.som.lrc.x - map.pp.ulc[0])/
			(map.pp.nline * map.pp.resolution)) + 1;

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Map_info = map;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
