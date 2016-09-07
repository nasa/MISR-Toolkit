/*===========================================================================
=                                                                           =
=                              MtkTaiJdToTai                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include "MisrError.h"

/** \brief Convert TAI Julian date to TAI93 time
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert a TAI Julian date to a TAI93 time.
 *
 *  \code
 *  status = MtkTaiJdToTai(jdTAI, &secTAI93);
 *  \endcode
 */

MTKt_status MtkTaiJdToTai(
  double jdTAI[2], /**< [IN] TAI Julian date */
  double *secTAI93 /**< [OUT] TAI93 Time */ )
{
  MTKt_status status_code; /* Return code of this function */

  if (secTAI93 == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *secTAI93 = (jdTAI[0] - EPOCH_DAY) * SECONDSperDAY +
              (jdTAI[1] - EPOCH_DAY_FRACTION) * SECONDSperDAY;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
