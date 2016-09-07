/*===========================================================================
=                                                                           =
=                              MtkCalToJulian                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include "MisrError.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*  LEAP_GREGORIAN  --  Is a given year in the Gregorian calendar a leap year ? */

static int leap_gregorian(int year)
{
    return ((year % 4) == 0) &&
            (!(((year % 100) == 0) && ((year % 400) != 0)));
}

/*  GREGORIAN_TO_JD  --  Determine Julian day number from Gregorian calendar date */

#define GREGORIAN_EPOCH 1721425.5

static double gregorian_to_jd(int year, int month, int day)
{
    return (GREGORIAN_EPOCH - 1.0) +
           (365.0 * (year - 1.0)) +
           floor((year - 1.0) / 4.0) +
           (-floor((year - 1.0) / 100.0)) +
           floor((year - 1.0) / 400.0) +
           floor((((367.0 * month) - 362.0) / 12.0) +
           ((month <= 2) ? 0 :
                               (leap_gregorian(year) ? -1 : -2)
           ) +
           day);
}

/** \brief Convert calendar date to Julian date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the calendar date 2002-05-02 02:00:00 to Julian date.
 *
 *  \code
 *  status = MtkCalToJulian(2002, 5, 2, 2, 0, 0, &julian);
 *  \endcode
 */

MTKt_status MtkCalToJulian(
  int year, /**< [IN] Year */
  int month, /**< [IN] Month */
  int day, /**< [IN] Day */
  int hour, /**< [IN] Hour */
  int min, /**< [IN] Minutes */
  int sec, /**< [IN] Seconds */
  double *jd /**< [OUT] Julian date */ )
{
  MTKt_status status_code; /* Return code of this function */

  if (jd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (year == 0 || month <= 0 || day <= 0 || hour < 0 || min < 0 || sec < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  *jd = gregorian_to_jd(year, month, day) +
               (floor(sec + 60 * (min + 60 * hour) + 0.5) / 86400.0);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
