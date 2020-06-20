/*===========================================================================
=                                                                           =
=                          MtkGenericMapInfoRead                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/
#include "MisrToolkit.h"
#include <stdlib.h>
#include <math.h>
#ifndef _MSC_VER
#include <strings.h> 		/* for strncasecmp */
#endif

#define MAX_LENGTH 1000

/** \brief  Initialize a MTKt_GenericMapInfo structure using data from an external file.
 *
 *  The external file must be simple text format with parameter_name = value pairs.
 *  
 *  Parameter names recognized by this routine are as follows:
 *    min_corner_x is the minimum X coordinate at the edge of the map.
 *    min_corner_y is the minimum Y coordinate at the edge of the map.
 *    resolution_x is the size of a pixel along the X-axis.
 *    resolution_y is the size of a pixel along the Y-axis.
 *    number_pixel_x is the size of the map in pixels, along the X-axis.
 *    number_pixel_y is the size of the map in pixels, along the Y-axis.
 *    origin_code defines the corner of the map at which pixel 0,0 is located. 
 *    pix_reg_code defines whether a pixel value is related to the corner or
 *      center of the corresponding area of that pixel on the map.  If the
 *      corner is used, then it is always the corner corresponding to the
 *      corner of the origin.
 *
 *  Possible values for origin_code are:
 *    UL - Upper Left (min X, max Y)
 *    UR - Upper Right (max X, max Y)
 *    LL - Lower Left (min X, min Y)
 *    LR - Lower Right (max X, min Y)
 *
 *  Possible values are:
 *    CENTER - Center
 *    CORNER - Corner
 *
 *  Unrecognized parameter names are ignored.  
 *  Lines starting with a '#' character are ignored.
 *  Comments may be placed on the same line, after a name = value pair as well.
 *
 *  Example map information file:
 *  # this is a comment
 *  min_corner_x = 10000.0
 *  min_corner_y = 20000.0
 *  resolution_x = 250.0
 *  resolution_y = 250.0
 *  number_pixel_x = 1000
 *  number_pixel_y = 2000
 *  origin_code = UL       # min x, max y
 *  pix_reg_code = CENTER
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkGenericMapInfoRead()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkGenericMapInfoRead(
  const char *Filename,  /**< [IN] Filename */
  MTKt_GenericMapInfo *Map_info /**< [OUT] Map information. */
)
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_GenericMapInfo map_info_tmp = MTKT_GENERICMAPINFO_INIT;
				/* Map information */
  double min_corner_x = 0.0;
  double min_corner_y = 0.0;
  double resolution_x = 0.0;
  double resolution_y = 0.0;
  int number_pixel_x = 0;
  int number_pixel_y = 0;
  MTKt_OriginCode origin_code = -1;
  MTKt_PixRegCode pix_reg_code = -1;
  FILE *fp = NULL;
  char line[MAX_LENGTH]; 	/* Line buffer for input file. */
  char origin_code_string[3] = {0,0,0};  	
				/* Buffer for origin_code string. */
  char pix_reg_code_string[7] = {0,0,0,0,0,0,0};  
				/* Buffer for pix_reg_code string. */
  int min_corner_x_found = 0;
  int min_corner_y_found = 0;
  int resolution_x_found = 0;
  int resolution_y_found = 0;
  int number_pixel_x_found = 0;
  int number_pixel_y_found = 0;
  int origin_code_found = 0;
  int pix_reg_code_found = 0;

  /* ------------------------------------------------------------------ */
  /* Argument check: Filename == NULL                                   */
  /* ------------------------------------------------------------------ */

  if (Filename == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Filename == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Open file.                                                         */
  /* ------------------------------------------------------------------ */

  fp = fopen(Filename,"r");
  if (fp == NULL) {
    MTK_ERR_CODE_JUMP(MTK_FILE_OPEN);
  }

  /* ------------------------------------------------------------------ */
  /* Scan input file for map parameters.                                */
  /* ------------------------------------------------------------------ */

  while (NULL != fgets(line,MAX_LENGTH,fp)) {
    if (line[0] == '#') continue;

    if (1 == sscanf(line, "min_corner_x = %lf",&min_corner_x)) {
      min_corner_x_found = 1;
    } else if (1 == sscanf(line, "min_corner_y = %lf",&min_corner_y)) {
      min_corner_y_found = 1;
    } else if (1 == sscanf(line, "resolution_x = %lf",&resolution_x)) {
      resolution_x_found = 1;
    } else if (1 == sscanf(line, "resolution_y = %lf",&resolution_y)) {
      resolution_y_found = 1;
    } else if (1 == sscanf(line, "number_pixel_x = %d",&number_pixel_x)) {
      number_pixel_x_found = 1;
    } else if (1 == sscanf(line, "number_pixel_y = %d",&number_pixel_y)) {
      number_pixel_y_found = 1;
    } else if (1 == sscanf(line, "origin_code = %2s",origin_code_string)) {
      origin_code_found = 1;
    } else if (1 == sscanf(line, "pix_reg_code = %6s",pix_reg_code_string)) {
      pix_reg_code_found = 1;
    } else {
      /* Skip unrecognized input */
    }
  } 

  /* ------------------------------------------------------------------ */
  /* Close file.                                                        */
  /* ------------------------------------------------------------------ */

  fclose(fp);
  fp = NULL;

  /* ------------------------------------------------------------------ */
  /* Check that all required paramters were found.                      */
  /* Argument check: min_corner_x not found                             */
  /*                 min_corner_y not found                             */
  /*                 resolution_x not found                             */
  /*                 resolution_y not found                             */
  /*                 number_pixel_x not found                           */
  /*                 number_pixel_y not found                           */
  /*                 origin_code not found                              */
  /*                 pix_reg_code not found                             */
  /* ------------------------------------------------------------------ */

  if (!min_corner_x_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"min_corner_x not found");
  }
  if (!min_corner_y_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"min_corner_y not found");
  }
  if (!resolution_x_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"resolution_x not found");
  }
  if (!resolution_y_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"resolution_y not found");
  }
  if (!number_pixel_x_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"number_pixel_x not found");
  }
  if (!number_pixel_y_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"number_pixel_y not found");
  }
  if (!origin_code_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"origin_code not found");
  }
  if (!pix_reg_code_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"pix_reg_code not found");
  }

  /* ------------------------------------------------------------------ */
  /* Parse origin_code                                                  */
  /* ------------------------------------------------------------------ */

  if (0 == strncasecmp("UL",origin_code_string,2)) {
    origin_code = MTKe_ORIGIN_UL;
  } else if (0 == strncasecmp("UR",origin_code_string,2)) {
    origin_code = MTKe_ORIGIN_UR;
  } else if (0 == strncasecmp("LL",origin_code_string,2)) {
    origin_code = MTKe_ORIGIN_LL;
  } else if (0 == strncasecmp("LR",origin_code_string,2)) {
    origin_code = MTKe_ORIGIN_LR;
  } else {
    fprintf(stderr,"Unrecognized origin_code: %2s\n",origin_code_string);
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* ------------------------------------------------------------------ */
  /* Parse pix_reg_code                                                 */
  /* ------------------------------------------------------------------ */

  if (0 == strncasecmp("CENTER",pix_reg_code_string,6)) {
    pix_reg_code = MTKe_PIX_REG_CENTER;
  } else if (0 == strncasecmp("CORNER",pix_reg_code_string,6)) {
    pix_reg_code = MTKe_PIX_REG_CORNER;
  } else {
    fprintf(stderr,"Unrecognized pix_reg_code: %6s\n",pix_reg_code_string);
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* ------------------------------------------------------------------ */
  /* Initialize map information.                                        */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfo(min_corner_x, min_corner_y, resolution_x, resolution_y, 
		    number_pixel_x, number_pixel_y, origin_code,
		    pix_reg_code, &map_info_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Map_info = map_info_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fp != NULL) {
    fclose(fp);
  }
  return status_code;
}


