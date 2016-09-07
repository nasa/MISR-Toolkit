/*===========================================================================
=                                                                           =
=                               MtkDmsToRad                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUnitConv.h"
#include "MisrError.h"
/* M_PI is not defined in math.h in Linux unless __USE_BSD is defined */
/* and you can define it at the gcc command-line if -ansi is set */
#ifndef __USE_BSD
# define __USE_BSD
#endif
#include <math.h>

/** \brief Convert packed degrees, minutes, seconds to Radians
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130004058.23 dms to radians.
 *
 *  \code
 *  status = MtkDmsToRad(130004058.23, &rad);
 *  \endcode
 */

MTKt_status MtkDmsToRad(
  double dms, /**< [IN] Packed degrees, minutes, seconds */
  double *rad /**< [OUT] Radians */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  double dd;			/* Decimal degrees */

  if (rad == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkDmsToDd( dms, &dd );
  MTK_ERR_COND_JUMP(status);

  *rad = dd * (M_PI / 180.0);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
