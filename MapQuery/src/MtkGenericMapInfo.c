/*===========================================================================
=                                                                           =
=                          MtkGenericMapInfo                                =
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
#include <stdlib.h>
#include <math.h>

/** \brief  Initialize a MTKt_GenericMapInfo structure.
 *
 *  Min_x and Min_y contains the minimum x,y coordinates at the
 *  edge of the map.  
 *
 *  Resolution_x is the size of a pixel along the X-axis.
 *  Resolution_y is the size of a pixel along the Y-axis.
 *
 *  Number_pixel_x and number_pixel_y specifies the size of the map in pixels
 *  along the respective axes.
 *
 *  Origin_code defines the corner of the map at which pixel 0,0 is located. 
 *  Possible values are:
 *            MTKe_ORIGIN_UL - Upper Left (min X, max Y)
 *            MTKe_ORIGIN_UR - Upper Right (max X, max Y)
 *            MTKe_ORIGIN_LL - Lower Left (min X, min Y)
 *            MTKe_ORIGIN_LR - Lower Right (max X, min Y)
 *
 *  Pix_reg_code defines whether a pixel value is related to the corner or
 *  center of the corresponding area of that pixel on the map.  If the
 *  corner is used, then it is always the corner corresponding to the
 *  corner of the origin.  Possible values are:
 *            MTKe_PIX_REG_CENTER - Center
 *            MTKe_PIX_REG_CORNER - Corner
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkGenericMapInfo()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkGenericMapInfo(
  double Min_x,  /**< [IN] Minimum X value in map units */
  double Min_y,  /**< [IN] Minimum Y value in map units */
  double Resolution_x,  /**< [IN] Size of a pixel along X-axis, in map units */
  double Resolution_y,  /**< [IN] Size of a pixel along Y-axis, in map units */
  int Number_pixel_x,   /**< [IN] Number of pixels along X-axis. */
  int Number_pixel_y,   /**< [IN] Number of pixels along Y-axis. */
  MTKt_OriginCode Origin_code,  /**< [IN] Corner of the map at which pixel 0,0 is located. */
  MTKt_PixRegCode Pix_reg_code, /**< [IN] Flag indicating if pixel values correspond to the center or corner of the area of the pixel on the map. */
  MTKt_GenericMapInfo *Map_info /**< [OUT] Map information. */
)
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_GenericMapInfo map_info_tmp = MTKT_GENERICMAPINFO_INIT;
				/* Map information */

  /* ------------------------------------------------------------------ */
  /* Argument check: Resolution_x <= 0.0                                */
  /*                 Resolution_y <= 0.0                                */
  /* ------------------------------------------------------------------ */

  if (Resolution_x <= 0.0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Resolution_x <= 0.0");
  }
  if (Resolution_y <= 0.0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Resolution_y <= 0.0");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Number_pixel_x < 1                                 */
  /*                 Number_pixel_y < 1                                 */
  /* ------------------------------------------------------------------ */

  if (Number_pixel_x < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Number_pixel_x < 1");
  }
  if (Number_pixel_y < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Number_pixel_y < 1");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Initialize map information.                                        */
  /* ------------------------------------------------------------------ */

  map_info_tmp.min_x = Min_x;
  map_info_tmp.min_y = Min_y;
  map_info_tmp.max_x = Min_x + Number_pixel_x * Resolution_x;
  map_info_tmp.max_y = Min_y + Number_pixel_y * Resolution_y;
  map_info_tmp.resolution_x = Resolution_x;
  map_info_tmp.resolution_y = Resolution_y;
  map_info_tmp.pix_reg_code = Pix_reg_code;
  map_info_tmp.origin_code = Origin_code;

  /* ------------------------------------------------------------------ */
  /* Setup transform coefficients for converting between map            */
  /* coordinates and pixel coordinates.                                 */
  /*                                                                    */
  /* Argument_check: Unsupported Origin_code                            */
  /* Argument_check: Unsupported Pix_reg_code                           */
  /* ------------------------------------------------------------------ */

  switch(Origin_code) {
  case MTKe_ORIGIN_UL: 
    map_info_tmp.size_line = Number_pixel_y;
    map_info_tmp.size_sample = Number_pixel_x;
    map_info_tmp.tsample[0] = 1; 
    map_info_tmp.tsample[1] = -map_info_tmp.min_x;
    map_info_tmp.tline[0] = -1;
    map_info_tmp.tline[1] = map_info_tmp.max_y;
    map_info_tmp.tline[3] = map_info_tmp.resolution_y;
    map_info_tmp.tsample[3] = map_info_tmp.resolution_x;
    break;
  case MTKe_ORIGIN_UR:
    map_info_tmp.size_line = Number_pixel_x;
    map_info_tmp.size_sample = Number_pixel_y;
    map_info_tmp.tline[0] = -1; 
    map_info_tmp.tline[1] = map_info_tmp.max_x;
    map_info_tmp.tsample[0] = -1;
    map_info_tmp.tsample[1] = map_info_tmp.max_y;
    map_info_tmp.tline[3] = map_info_tmp.resolution_x;
    map_info_tmp.tsample[3] = map_info_tmp.resolution_y;
    break;
  case MTKe_ORIGIN_LL:
    map_info_tmp.size_line = Number_pixel_x;
    map_info_tmp.size_sample = Number_pixel_y;
    map_info_tmp.tline[0] = 1; 
    map_info_tmp.tline[1] = -map_info_tmp.min_x;
    map_info_tmp.tsample[0] = 1;
    map_info_tmp.tsample[1] = -map_info_tmp.min_y;
    map_info_tmp.tline[3] = map_info_tmp.resolution_x;
    map_info_tmp.tsample[3] = map_info_tmp.resolution_y;
    break;
  case MTKe_ORIGIN_LR:
    map_info_tmp.size_line = Number_pixel_y;
    map_info_tmp.size_sample = Number_pixel_x;
    map_info_tmp.tsample[0] = -1; 
    map_info_tmp.tsample[1] = map_info_tmp.max_x;
    map_info_tmp.tline[0] = 1;
    map_info_tmp.tline[1] = -map_info_tmp.min_y;
    map_info_tmp.tline[3] = map_info_tmp.resolution_y;
    map_info_tmp.tsample[3] = map_info_tmp.resolution_x;
    break;
  default:
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Unsupported Origin_code");
  }

  switch(Pix_reg_code) {
  case MTKe_PIX_REG_CORNER: 
    map_info_tmp.tline[2] = map_info_tmp.tsample[2] = 0.0;
    break;
  case MTKe_PIX_REG_CENTER:
    map_info_tmp.tline[2] = map_info_tmp.tsample[2] = -0.5;
    break;
  default:
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Unsupported Pix_reg_code");
  }

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Map_info = map_info_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}


