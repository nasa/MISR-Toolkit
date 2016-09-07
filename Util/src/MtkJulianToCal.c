/*===========================================================================
=                                                                           =
=                              MtkJulianToCal                               =
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

/*  MOD  --  Modulus function which works for non-integers.  */

static double mod(double a, double b)
{
    return a - (b * floor(a / b));
}


/*  JD_TO_GREGORIAN  --  Calculate Gregorian calendar date from Julian day */

static void jd_to_gregorian(double jd, int *y, int *m, int *d)
{
  double wjd;
  double depoch;
  double quadricent;
  double dqc;
  int cent;
  double dcent;
  double quad;
  double dquad;
  int yindex;
  int year;
  int month;
  int day;
  double yearday;
  int leapadj;

  wjd = floor(jd - 0.5) + 0.5;
  depoch = wjd - GREGORIAN_EPOCH;
  quadricent = floor(depoch / 146097);
  dqc = mod(depoch, 146097);
  cent = (int)floor(dqc / 36524);
  dcent = mod(dqc, 36524);
  quad = floor(dcent / 1461);
  dquad = mod(dcent, 1461);
  yindex = (int)floor(dquad / 365);
  year = (int)((quadricent * 400) + (cent * 100) + (quad * 4) + yindex);
  if (!((cent == 4) || (yindex == 4))) {
    year++;
  }
  yearday = wjd - gregorian_to_jd(year, 1, 1);
  leapadj = ((wjd < gregorian_to_jd(year, 3, 1)) ? 0 :
	     (leap_gregorian(year) ? 1 : 2));
  month = (int)floor((((yearday + leapadj) * 12) + 373) / 367);
  day = (int)((wjd - gregorian_to_jd(year, month, 1)) + 1);

  *y = year;
  *m = month;
  *d = day;
}

/*  JHMS  --  Convert Julian time to hour, minutes, and seconds. */

static void jhms(double j, int *hour, int *min, int *sec)
{
    int ij;

    j += 0.5;                 /* Astronomical to civil */
    ij = (int)(((j - floor(j)) * 86400.0) + 0.5);
    *hour = (int)floor(ij / 3600);
    *min = (int)floor((ij / 60) % 60);
    *sec = (int)floor(ij % 60);
}

/** \brief Convert Julian date to calendar date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the julian date 2453728.27313 to calendar date.
 *
 *  \code
 *  status = MtkJulianToCal(2453728.27313, &year, &month, &day, &hour, &min, &sec);
 *  \endcode
 * 
 *  \note Julian date must be >= 1721119.5
 */

MTKt_status MtkJulianToCal(
  double jd, /**< [IN] Julian date */
  int *year, /**< [OUT] Year */
  int *month, /**< [OUT] Month */
  int *day, /**< [OUT] Day */
  int *hour, /**< [OUT] Hour */
  int *min, /**< [OUT] Minutes */
  int *sec /**< [OUT] Seconds */ )
{
  MTKt_status status_code; /* Return code of this function */

  if (year == NULL || month == NULL || day == NULL ||
      hour == NULL || min == NULL || sec == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (jd < 1721119.5)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  jd_to_gregorian(jd,year,month,day);
  jhms(jd,hour,min,sec);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
