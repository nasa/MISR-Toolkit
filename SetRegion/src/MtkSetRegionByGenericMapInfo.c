/*===========================================================================
=                                                                           =
=                          MtkSetRegionByGenericMapInfo                     =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrSetRegion.h"
#include "MisrUtil.h"
#include <stdlib.h>
#include <math.h>
#include "MisrToolkit.h"  /* Definition of inv_init */
#include <proj.h> 		/* Definition of MAXPROJ */

/** \brief  Create an MtkRegion structure that contains the given map.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkSetRegionByGenericMapInfo()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkSetRegionByGenericMapInfo(
  const MTKt_GenericMapInfo *Map_info, /**< [IN] Map information. */
  const MTKt_GCTPProjInfo *Proj_info,  /**< [IN] Projection information. */
  int Path,			       /**< [IN] Orbit path number. */
  MTKt_Region *Region /**< [OUT] Region */
) 
{
  MTKt_status status_code = MTK_FAILURE;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_Region region_tmp = MTKT_REGION_INIT; 
				/* Region */
  double corner_lat[4]; 	/* Latitude coordinates for each corner of the
                                   target map. */
  double corner_lon[4]; 	/* Longitude coordinates for each corner of the
                                   target map. */
  double min_som_x = 0.0;	/* Minimum SOM X coordinate. */
  double min_som_y = 0.0;	/* Minimum SOM Y coordinate. */
  double max_som_x = 0.0;	/* Maximum SOM X coordinate. */
  double max_som_y = 0.0;	/* Maximum SOM Y coordinate. */
  int i; 			/* Loop iterator */
  double rad2deg = 180 / acos(-1); /* For converting Radians to degress */

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /* ------------------------------------------------------------------ */

  if (Map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /*                 Proj_info->proj_code > MAXPROJ                     */
  /* ------------------------------------------------------------------ */

  if (Proj_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Proj_info == NULL");
  }
  if (Proj_info->proj_code > MAXPROJ) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Proj_info->proj_code > MAXPROJ");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Region == NULL                                     */
  /* ------------------------------------------------------------------ */

  if (Region == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Region == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Calculate lat/lon coordinates for each corner of the target map.   */
  /* ------------------------------------------------------------------ */

  { int iflg;   		/* Status flag returned by GCTP. */
    int (*inv_trans[MAXPROJ+1])(); /* Array of transformation functions 
				   returned by GCTP. */
    
    inv_init(Proj_info->proj_code, Proj_info->zone_code, 
	     (double *)Proj_info->proj_param, 
	     Proj_info->sphere_code, 0, 0, &iflg, inv_trans);
    if (iflg) {
      MTK_ERR_MSG_JUMP("Trouble with GCTP inv_init routine\n");
    }

    inv_trans[Proj_info->proj_code](Map_info->min_x, Map_info->min_y, 
				   &corner_lon[0], &corner_lat[0]);
    inv_trans[Proj_info->proj_code](Map_info->min_x, Map_info->max_y, 
				   &corner_lon[1], &corner_lat[1]);
    inv_trans[Proj_info->proj_code](Map_info->max_x, Map_info->min_y, 
				   &corner_lon[2], &corner_lat[2]);
    inv_trans[Proj_info->proj_code](Map_info->max_x, Map_info->max_y, 
				   &corner_lon[3], &corner_lat[3]);
  }
  
  /* ------------------------------------------------------------------ */
  /* Calculate SOM coordinates for each corner of the target map.       */
  /* Find the minimum and maximum SOM coords                            */
  /* ------------------------------------------------------------------ */

  for (i = 0 ; i < 4 ; i++) {
    double som_x;	      /* SOM X coordinate for this corner. */ 
    double som_y;	      /* SOM Y coordinate for each corner of the
				 target map. */

    status = MtkLatLonToSomXY(Path,
			      corner_lat[i]*rad2deg,
			      corner_lon[i]*rad2deg,
			      &som_x, &som_y);
    MTK_ERR_COND_JUMP(status);

    if (i == 0 || som_x < min_som_x) {
      min_som_x = som_x;
    }
    if (i == 0 || som_y < min_som_y) {
      min_som_y = som_y;
    }
    if (i == 0 || som_x > max_som_x) {
      max_som_x = som_x;
    }
    if (i == 0 || som_y > max_som_y) {
      max_som_y = som_y;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Create region containing the target map area.                      */
  /* ------------------------------------------------------------------ */

  {
    double som_x_extent = max_som_x - min_som_x;
    double som_y_extent = max_som_y - min_som_y;
    
    double center_som_x = min_som_x + (max_som_x - min_som_x) / 2.0;
    double center_som_y = min_som_y + (max_som_y - min_som_y) / 2.0;
    double center_lat;
    double center_lon;

    status = MtkSomXYToLatLon(Path, center_som_x, center_som_y, &center_lat,
			      &center_lon);
    MTK_ERR_COND_JUMP(status);

    status = MtkSetRegionByLatLonExtent(center_lat, center_lon, 
					som_x_extent, som_y_extent, "meters",
					&region_tmp);
    MTK_ERR_COND_JUMP(status);
  }


  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Region = region_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
