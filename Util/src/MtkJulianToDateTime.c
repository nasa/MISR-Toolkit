/*===========================================================================
=                                                                           =
=                           MtkJulianToDateTime                             =
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
#include <string.h>

/** \brief Convert Julian date to date and time (ISO 8601)
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the julian date 2453728.27313 to date and time (ISO 8601).
 *
 *  \code
 *  status = MtkJulianToDateTime(2453728.27313, datetime);
 *  \endcode
 */

MTKt_status MtkJulianToDateTime(
  double jd, /**< [IN] Julian date */
  char datetime[MTKd_DATETIME_LEN] /**< [OUT] Date and time (ISO 8601) */ )
{
  MTKt_status status; /* Return status */
  MTKt_status status_code; /* Return code of this function */
  int year, month, day, hour, min, sec;

  if (datetime == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (jd < 1721119.5)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkJulianToCal(jd,&year,&month,&day,&hour,&min,&sec);
  MTK_ERR_COND_JUMP(status);

  sprintf(datetime,"%04d-%02d-%02dT%02d:%02d:%02dZ",year,month,day,hour,min,sec);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
