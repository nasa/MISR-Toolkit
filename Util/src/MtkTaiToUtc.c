/*===========================================================================
=                                                                           =
=                               MtkTaiToUtc                                 =
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

/** \brief Convert TAI93 to UTC
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the TAI93 time to UTC time.
 *
 *  \code
 *  status = MtkTaiToUtc(429030246.630996, utc_datetime);
 *  \endcode
 */

MTKt_status MtkTaiToUtc(
  double secTAI93, /**< [IN] TAI93 Time */
  char utc_datetime[MTKd_DATETIME_LEN] /**< [OUT] UTC Date time */ )
{
  double jdTAI[2];	  /* TAI Julian Date equivalent of input time */
  double jdUTC[2];	  /* UTC Julian Date equivalent of input time */
  
  MtkTaiToTaiJd(secTAI93, jdTAI);
  MtkTaiJdToUtcJd(jdTAI, jdUTC);
  MtkUtcJdToUtc(jdUTC, utc_datetime);
 
  return MTK_SUCCESS;
}
