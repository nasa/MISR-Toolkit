/*===========================================================================
=                                                                           =
=                          MtkGCTPCreateLatLon                              =
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
#include "MisrCoordQuery.h" /* definitino of inv_init  */
#include <stdlib.h>
#include <math.h>

/** \brief Create an array of latitude and longitude values corresponding to each pixel in the given map.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkGCTPCreateLatLon()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkGCTPCreateLatLon(
  const MTKt_GenericMapInfo *Map_info,  /**< [IN] Map information. */
  const MTKt_GCTPProjInfo *Proj_info, /**< [IN] Projection information. */
  MTKt_DataBuffer *Latitude,  /**< [OUT] Latitude values. */
  MTKt_DataBuffer *Longitude /**< [OUT] Longitude values. */
)
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_DataBuffer latitude_tmp = MTKT_DATABUFFER_INIT;
				/* Latitude values in degrees */
  MTKt_DataBuffer longitude_tmp = MTKT_DATABUFFER_INIT;
				/* Longitude values in degrees */
  int iflg;			/* Status flag returned by GCTP library. */
  int (*inv_trans[1000])(); 	/* Array of transform functions returned by GCTP. */
  int iline;			/* Loop iterator. */
  int isample;			/* Loop iterator. */
  double rad2deg = 180.0 / acos(-1);

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /*                 Map_info->size_x < 1                               */
  /*                 Map_info->size_y < 1                               */
  /* ------------------------------------------------------------------ */

  if (Map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info == NULL");
  }
  if (Map_info->size_line < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info->size_line < 1");
  }
  if (Map_info->size_sample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info->size_sample < 1");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Proj_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Proj_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Latitude == NULL                                   */
  /* ------------------------------------------------------------------ */

  if (Latitude == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Latitude == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Longitude == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Longitude == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Longitude == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Allocate memory for latitude and longitude.                        */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(Map_info->size_line, Map_info->size_sample, 
				 MTKe_double, &latitude_tmp);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(Map_info->size_line, Map_info->size_sample, 
				 MTKe_double, &longitude_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Initialize GCTP library.                                           */
  /* ------------------------------------------------------------------ */

  inv_init(Proj_info->proj_code, Proj_info->zone_code, Proj_info->proj_param,
           Proj_info->sphere_code, 0, 0, &iflg, inv_trans);
  if (iflg) {
    printf("iflg = %d\n",iflg);
    MTK_ERR_CODE_JUMP(MTK_GCTP_INVERSE_INIT_FAILED);
  }

  /* ------------------------------------------------------------------ */
  /* For each pixel...                                                  */
  /* ------------------------------------------------------------------ */

  for (iline = 0 ; iline < Map_info->size_line ; iline++) {
    for (isample = 0 ; isample < Map_info->size_sample ; isample++) {
      double x,y;  		/* Map coordinates */
      double lat,lon; 		/* Lat/lon coordinates */
      double line, sample; 	/* Pixel coordinates. */

  /* ------------------------------------------------------------------ */
  /* Calculate map coordinate for this pixel.                           */
  /* ------------------------------------------------------------------ */

      line = (((iline - Map_info->tline[2]) * Map_info->tline[3]) - 
	   Map_info->tline[1]) / Map_info->tline[0];
      sample = (((isample - Map_info->tsample[2]) * Map_info->tsample[3]) - 
	   Map_info->tsample[1]) / Map_info->tsample[0];

  /* ------------------------------------------------------------------ */
  /* Calculate lat/lon values at this map coordinate.                   */
  /* Mapping from X, Y to line, sample depends on the origin code as    */
  /* follows:                                                           */
  /*                                                                    */
  /*   origin_code   line   sample                                      */
  /*   UL              Y      X                                         */
  /*   UR              X      Y                                         */
  /*   LL              X      Y                                         */
  /*   LR              Y      X                                         */
  /*                                                                    */
  /* Argument check: Origin_code not recognized                         */
  /* ------------------------------------------------------------------ */

      switch(Map_info->origin_code) {
      case MTKe_ORIGIN_UL:
      case MTKe_ORIGIN_LR:
	x = sample;
	y = line; 
	break;
      case MTKe_ORIGIN_UR:
      case MTKe_ORIGIN_LL:
	x = line;
	y = sample; 
	break;
      default:
	MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Origin_code not recognized");
      }
      
      inv_trans[Proj_info->proj_code](x, y, &lon, &lat);
       
  /* ------------------------------------------------------------------ */
  /* Set lat/lon for this pixel.                                        */
  /* ------------------------------------------------------------------ */

       latitude_tmp.data.d[iline][isample] = lat * rad2deg;
       longitude_tmp.data.d[iline][isample] = lon * rad2deg;

  /* ------------------------------------------------------------------ */
  /* End loop for each pixel.                                           */
  /* ------------------------------------------------------------------ */

    }
  }
  
  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Latitude = latitude_tmp;
  *Longitude = longitude_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&latitude_tmp);
  MtkDataBufferFree(&longitude_tmp);
  return status_code;
}


