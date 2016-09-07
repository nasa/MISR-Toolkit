/*===========================================================================
=                                                                           =
=                            MtkPathToProjParam                             =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrUnitConv.h"
#include "MisrProjParam.h"
#include <stddef.h>
/* M_PI is not defined in math.h in Linux unless __USE_BSD is defined */
/* and you can define it at the gcc command-line if -ansi is set */
#ifndef __USE_BSD
# define __USE_BSD
#endif
#include <math.h>

/** \brief Get projection parameters
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the projection parameters for path 1 at 1100 meter resolution.
 *
 *  \code
 *  MTKt_MisrProjParam pp = MTKT_MISRPROJPARAM_INIT;
 *  status = MtkPathToProjParam(1, 1100, &pp);
 *  \endcode
 */

MTKt_status MtkPathToProjParam(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  MTKt_MisrProjParam *pp /**< [OUT] Projection Parameters */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_MisrProjParam pp_temp = MTKT_MISRPROJPARAM_INIT;
				/* MISR projection parameters */
  MTKt_status status; 		/* Return status */
  double asclong;		/* Long of ascending node in dms */
  double lambda0;		/* Path 1 long of ascending node (radians) */
  double lambda;		/* Current path long of ascending node (rad) */
  int resolution;		/* Desired resolution */
  float resfactor;		/* Resolution scale factor */
  int i;			/* Loop index */

  if (pp == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Use default resolution if NULL is passed */
  if (resolution_meters == 0)
    resolution = RESOLUTION;
  else
    resolution = resolution_meters;

  /* Check the bounds of path */
  if (path < 1 || path > NPATH)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Compute resolution scale factor */
  resfactor = pp_temp.resolution / (float)resolution;

  /* Compute longitude of ascending node for this path from first path */
  status = MtkDmsToRad(PP5_ASCLONG, &lambda0);
  MTK_ERR_COND_JUMP(status);

  lambda = lambda0 - (2.0 * M_PI / NPATH) * (path - 1);

  status = MtkRadToDms(lambda, &asclong);
  MTK_ERR_COND_JUMP(status);

  /* Set path and resolution dependent projection parameters. */
  pp_temp.path = path;
  pp_temp.projparam[4] = asclong;
  pp_temp.nline = (int)(pp_temp.nline * resfactor);
  pp_temp.nsample = (int)(pp_temp.nsample * resfactor);
  pp_temp.resolution = resolution;

  for (i = 0; i < NBLOCK - 2; i++) {
    pp_temp.reloffset[i] *= resfactor;
  }

  *pp = pp_temp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
