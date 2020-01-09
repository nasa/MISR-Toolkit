/*===========================================================================
=                                                                           =
=                             MtkDegMinSecToDd                              =
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
#include <math.h>
#include <stdlib.h>

/** \brief Convert unpacked degrees, minutes, seconds to decimal degrees.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130 degress, 4 minutes, and 58.23 seconds to decimal degrees.
 *
 *  \code
 *  status = MtkDegMinSecToDd(130, 4, 58.23, &dd);
 *  \endcode
 */

MTKt_status MtkDegMinSecToDd(
  int deg,    /**< [IN] Degrees */
  int min,    /**< [IN] Minutes */
  double sec, /**< [IN] Seconds */
  double *dd  /**< [OUT] Decimal degrees */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int sgn;                      /* Sign of input */
  
  if (dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* determine sign */

  if (deg < 0) {
    sgn = -1;
  } else {
    sgn = 1;
  }
  deg = abs(deg);

  /* Check degrees, minutes, seconds bounds */

  if (deg > 360) return MTK_OUTBOUNDS;
  if (min > 60) return MTK_OUTBOUNDS;
  if (sec > 60.0) return MTK_OUTBOUNDS;

  *dd = sgn * (deg * 3600.0 + min * 60.0 + sec) / 3600.0;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
