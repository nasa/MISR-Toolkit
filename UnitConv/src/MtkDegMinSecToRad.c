/*===========================================================================
=                                                                           =
=                            MtkDegMinSecToRad                              =
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

/** \brief Convert unpacked degrees, minutes, seconds to radians
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130 degress, 4 minutes, and 58.23 seconds to radians.
 *
 *  \code
 *  status = MtkDegMinSecToRad(130, 4, 58.23, &rad);
 *  \endcode
 */

MTKt_status MtkDegMinSecToRad(
  int deg,    /**< [IN] Degrees */
  int min,    /**< [IN] Minutes */
  double sec, /**< [IN] Seconds */
  double *rad /**< [OUT] Radians */ )
{
  MTKt_status status_code;      /* Returen status of this function */
  MTKt_status status;           /* Return status */
  double dd;			/* Decimal degrees */

  if (rad == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkDegMinSecToDd( deg, min, sec, &dd );
  MTK_ERR_COND_JUMP(status);

  *rad = dd * (M_PI / 180.0);

  return status;

ERROR_HANDLE:
  return status_code;
}
