/*===========================================================================
=                                                                           =
=                           MtkDateTimeToJulian                             =
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
#include <stdio.h>

/** \brief Convert date and time (ISO 8601) to Julian date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the date and time 2002-05-02 02:00:00 to Julian date.
 *
 *  \code
 *  status = MtkDateTimeToJulian("2002-05-02T02:00:00Z", &julian);
 *  \endcode
 */

MTKt_status MtkDateTimeToJulian(
  const char *datetime, /**< [IN] Date and time (ISO 8601) */
  double *jd /**< [OUT] Julian date */ )
{
  MTKt_status status; /* Return status */
  MTKt_status status_code; /* Return code of this function */
  int year, month, day, hour, min, sec;
  int num_values;

  if (datetime == NULL || jd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  num_values = sscanf(datetime,"%4d-%2d-%2dT%2d:%2d:%2d",&year,&month,
                      &day,&hour,&min,&sec);

  if (num_values != 6)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkCalToJulian(year,month,day,hour,min,sec,jd);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
