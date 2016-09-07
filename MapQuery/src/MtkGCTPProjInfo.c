/*===========================================================================
=                                                                           =
=                          MtkGCTPProjInfo                                 =
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

/** \brief  Initialize a MTKt_GCTPProjInfo structure.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  status = MtkGCTPProjInfo()
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkGCTPProjInfo(
  int Proj_code,  /**< [IN] Projection code. */
  int Sphere_code, /**< [IN] Spheroid code. */
  int Zone_code,  /**< [IN] UTM zone code (only applicable if Projection code is UTM */
  double Proj_param[15], /**< [IN] Projection parameters. */
  MTKt_GCTPProjInfo *Proj_info /**< [OUT] Projection information. */
)
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_GCTPProjInfo proj_info_tmp = MTKT_GCTPPROJINFO_INIT;
				/* Projection information */
  int i;

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /* ------------------------------------------------------------------ */
  
  if (Proj_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Proj_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Initialize projection information.                                 */
  /* ------------------------------------------------------------------ */

  proj_info_tmp.proj_code = Proj_code;
  proj_info_tmp.sphere_code = Sphere_code;
  proj_info_tmp.zone_code = Zone_code;
  for (i = 0 ; i < 15; i++) {
    proj_info_tmp.proj_param[i] = Proj_param[i];
  }

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Proj_info = proj_info_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}


