/*===========================================================================
=                                                                           =
=                              MtkTaiToTaiJd                                =
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
#include <math.h>

/** \brief Convert TAI93 to TAI Julian date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from a TAI93 time to a TAI Julian date. 
 *
 *  \code
 *  status = MtkTaiToTaiJd(429030246.630996, jdTAI);
 *  \endcode
 */

MTKt_status MtkTaiToTaiJd(
  double secTAI93, /**< [IN] TAI93 Time */
  double jdTAI[2] /**< [OUT] TAI Julian date */ )
{
  double dayFraction;

  dayFraction = fmod(secTAI93,SECONDSperDAY);
  jdTAI[0] = EPOCH_DAY + ((secTAI93 - dayFraction) / SECONDSperDAY);
  jdTAI[1] = dayFraction / SECONDSperDAY + EPOCH_DAY_FRACTION;

  if (jdTAI[1] >= 1.0)
  {
    jdTAI[0] += 1.0;
    jdTAI[1] -= 1.0;
  }
  else if (jdTAI[1] < 0)
  {
    jdTAI[0] -= 1.0;
    jdTAI[1] += 1.0;
  }

  return MTK_SUCCESS;
}
