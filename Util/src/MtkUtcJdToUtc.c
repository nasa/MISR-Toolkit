/*===========================================================================
=                                                                           =
=                               MtkUtcJdToUtc                               =
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

static double* JulianDateSplit(
  double inputJD[2],
  double outputJD[2] )
{
    double remainder;
    double temp; 
	
    outputJD[0] = inputJD[0];
    outputJD[1] = inputJD[1];

    /* the Julian date must be of the form:
       outputJD[0] = (Julian Day #) + 0.5
       outputJD[1] = Julian Day fraction (0 <= outputJD[1] < 1) */

    /* Make sure outputJD[0] is greater in magnitude than outputJD[1] */

    if (fabs(outputJD[1]) > fabs(outputJD[0]))
    {
	  temp = outputJD[0];
	  outputJD[0] = outputJD[1];
	  outputJD[1] = temp;
    }

    /* Make sure outputJD[0] is half integral */

    if ((remainder=fmod(outputJD[0],1.0)) != 0.5)
    {
	  outputJD[0] = outputJD[0] - remainder + 0.5;
	  outputJD[1] = outputJD[1] + remainder - 0.5;
    }

    /* Make sure magnitude of outputJD[1] is less than 1.0 */
    if (fabs(outputJD[1]) >= 1.0)
    {
	  remainder=fmod(outputJD[1],1.0);
	  outputJD[0] += outputJD[1] - remainder;
	  outputJD[1] = remainder;
    }

    /* Make sure outputJD[1] is greater than or equal to 0.0 */

    if (outputJD[1] < 0)
    {
	  if ((outputJD[1] + 1.0) < 1.0)
	  {
	    outputJD[0] -= 1.0;
	    outputJD[1] += 1.0;
	  }
	  else
	    outputJD[1] = 0.0;
    }

    return outputJD;
}

static void calday(                 /* converts Julian day to calendar components */
  int julianDayNum, /* Julian day */
  int *year,        /* calendar year */
  int *month,       /* calendar month */
  int *day)         /* calendar day (of month) */
{ 
    long         l;            /* intermediate variable */
    long         n;            /* intermediate variable */

    l = julianDayNum + 68569; 
    n = 4 * l / 146097; 
    l = l - (146097 * n + 3) / 4;  
    *year = (int)(4000 * (l + 1) / 1461001); 
    l = l - 1461 * (*year) / 4 + 31; 
    *month = (int)(80 * l / 2447L); 
    *day = (int)(l - 2447 * (*month) / 80);  
    l = *month / 11;  
    *month = (int)(*month + 2 - 12 * l);  
    *year = (int)(100 * (n - 49) + *year + l);
     
    return;  
}


/** \brief Convert calendar date to Julian date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from the calendar date 2002-05-02 02:00:00 to Julian date.
 *
 *  \code
 *  status = MtkUtcJdToUtc(2002, 5, 2, 2, 0, 0, &julian);
 *  \endcode
 */

MTKt_status MtkUtcJdToUtc(
  double jdUTCin[2], /**< [IN] UTC Julian date */
  char utc_datetime[MTKd_DATETIME_LEN] /**< [OUT] UTC Date time */ )
{
  double  jdUTC[2];
  int year;             /* year portion of date */
  int month;            /* month portion of date */
  int day;              /* day portion of date */
  int          hours;           /* hour of day */
  int          minutes;         /* minute of hour */
  double  seconds;         /* seconds of minute */
  int          intSecs;   /* integer number of UTC seconds of latest minute */
  int          fracSecs;  /* integer number of microseconds of latest second */
  double  dayFractionSecs;
    
  JulianDateSplit(jdUTCin,jdUTC);

  calday((int)(jdUTC[0] + 0.5), &year, &month, &day);

  dayFractionSecs = jdUTC[1] * SECONDSperDAY + 5.0e-7;

  hours = (int) (dayFractionSecs / 3600.0);
  minutes = (int) ((dayFractionSecs - hours * 3600.0) / 60.0);
  seconds = (dayFractionSecs - hours * 3600.0 - minutes * 60.0);
  intSecs = (int) seconds;
  fracSecs = (int) (fmod(dayFractionSecs,1.0) * 1.0e6);
    
    /* sometimes jdUTC[1] is right on the edge and is just shy of 1 when it
       should be 1 and "tip over" into the next day */

  if (hours == 24)
  {
	hours = 0;
	calday((int)(jdUTC[0] + 1.5), &year, &month, &day);
  }
    
  sprintf(utc_datetime,"%04d-%02d-%02dT%02d:%02d:%02d.%06dZ", (int)year, 
	      (int)month, (int)day, hours, minutes, intSecs, fracSecs);
	    
  return MTK_SUCCESS;
}
