/*===========================================================================
=                                                                           =
=                             MtkDdToDegMinSec                              =
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

/** \brief Convert decimal degrees to unpacked degrees, minutes, seconds
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130.08284167 degrees to unpacked degrees, minutes, seconds.
 *
 *  \code
 *  status = MtkDdToDegMinSec(130.08284167, &deg, &min, &sec);
 *  \endcode
 */

MTKt_status MtkDdToDegMinSec(
  double dd,  /**< [IN] Decimal degrees */
  int *deg,   /**< [OUT] Degrees */
  int *min,   /**< [OUT] Minutes */
  double *sec /**< [OUT] Seconds */ )
{
  MTKt_status status_code;      /* Retrun status of this function */
  MTKt_status status;           /* Return status */
  double dms;			/* Packed degrees, minutes seconds */

  if (deg == NULL || min == NULL || sec == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkDdToDms(dd, &dms);
  MTK_ERR_COND_JUMP(status);

  status = MtkDmsToDegMinSec(dms, deg, min, sec);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
