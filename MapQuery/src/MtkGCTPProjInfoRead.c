/*===========================================================================
=                                                                           =
=                          MtkGCTPProjInfoRead                              =
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

#define MAX_LENGTH 1000

/** \brief  Initialize a MTKt_GCTPProjInfo structure using data from an external file.
 *
 *  The external file must be simple text format with parameter_name = value pairs.
 *  
 *  Parameter names recognized by this routine are as follows:
 *    proj_code is the GCTP projection code.
 *    utm_zone is the UTM zone number for UTM projections only.
 *    sphere_code is GCTP spheroid code.
 *    proj_param(n) is the nth GCTP projection parameter.  (1 <= n <= 15)
 *
 *  Unrecognized parameter names are ignored.  
 *  Lines starting with a '#' character are ignored.
 *  Everything after the first name = value on a line is ignored.
 *
 *  The default value for proj_param(n) which are not specified is zero.
 *  Utm_zone is only required if proj_code is UTM (1).
 *
 *  Example map information file:
 *  # this is a comment
 *  proj_code = 3   # Albers
 *  #utm_zone = -1
 *  sphere_code = 8   # GRS80 
 *  proj_param(1) = 0.0   # Semi-major axis of ellipsoid. (superceded by sphere code)
 *  proj_param(2) = 0.0   # Semi-minor axis of ellipsoid. (superceded by sphere code)
 *  proj_param(3) = 29030000.0  # Latitude of the first standard parallel. (DDDMMMSSS.SS)
 *  proj_param(4) = 45030000.0  # Latitude of the second standard parallel. (DDDMMMSSS.SS)
 *  proj_param(5) = -96000000.0 # Longitude of the central meridian. (DDDMMMSSS.SS)
 *  proj_param(6) = 23000000.0  # Latitude of the projection origin. (DDDMMMSSS.SS)
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkGCTPProjInfoRead()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkGCTPProjInfoRead(
  const char *Filename,  /**< [IN] Filename */
  MTKt_GCTPProjInfo *Proj_info /**< [OUT] Proj information. */
)
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_GCTPProjInfo proj_info_tmp = MTKT_GCTPPROJINFO_INIT;
				/* Proj information */
  int proj_code;
  int utm_zone = -1;
  int sphere_code;
  double proj_param[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
			   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  int proj_code_found = 0;
  int utm_zone_found = 0;
  int sphere_code_found = 0;

  int i;
  double value;
  
  FILE *fp = NULL;
  char line[MAX_LENGTH]; 	/* Line buffer for input file. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Filename == NULL                                   */
  /* ------------------------------------------------------------------ */

  if (Filename == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Filename == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Proj_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Proj_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Open file.                                                         */
  /* ------------------------------------------------------------------ */

  fp = fopen(Filename,"r");
  if (fp == NULL) {
    MTK_ERR_CODE_JUMP(MTK_FILE_OPEN);
  }

  /* ------------------------------------------------------------------ */
  /* Scan input file for proj parameters.                                */
  /* ------------------------------------------------------------------ */

  while (NULL != fgets(line,MAX_LENGTH,fp)) {
    if (line[0] == '#') continue;

    if (1 == sscanf(line, "proj_code = %d",&proj_code)) {
      proj_code_found = 1;
    } else if (1 == sscanf(line, "utm_zone = %d",&utm_zone)) {
      utm_zone_found = 1;
    } else if (1 == sscanf(line, "sphere_code = %d",&sphere_code)) {
      sphere_code_found = 1;
    } else if (2 == sscanf(line, "proj_param(%d) = %lf",&i,&value)) {
      if (i > 0 && i < 16) {
	proj_param[i-1] = value;
      }
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
  /* Argument check: proj_code not found                                */
  /*                 proj_code == UTM && utm_zone not found             */
  /*                 sphere_code not found                              */
  /* ------------------------------------------------------------------ */

  if (!proj_code_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"proj_code not found");
  }
  if (proj_code == 1 && !utm_zone_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"utm_zone not found");
  }
  if (!sphere_code_found) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NOT_FOUND,"sphere_code not found");
  }

  /* ------------------------------------------------------------------ */
  /* Initialize proj information.                                        */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfo(proj_code, sphere_code, utm_zone, proj_param,
			   &proj_info_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Proj_info = proj_info_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fp != NULL) {
    fclose(fp);
  }
  return status_code;
}


