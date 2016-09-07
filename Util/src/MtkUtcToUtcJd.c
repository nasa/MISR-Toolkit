/*===========================================================================
=                                                                           =
=                               MtkUtcToUtcJd                               =
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

/* Convert calendar day to Julian day */
static int julday(int year, int month, int day)
{ 
    long j1;    /*  Scratch Variable */
    long j2;    /*  Scratch Variable */
    long j3;    /*  Scratch Variable */
    
    j1 = 1461L * (year + 4800L + (month - 14L) / 12L) / 4L;  
    j2 = 367L * (month - 2L - (month - 14L) / 12L * 12L) / 12L;
    j3 = 3L * ((year + 4900L + (month - 14L) / 12L) / 100L) / 4L;  
    return (int)(day - 32075L + j1 + j2 - j3);  
} 


/** \brief Convert UTC date to UTC Julian date
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert the UTC date 2006-08-06T15:04:00.630996Z to a UTC Julian date.
 *
 *  \code
 *  status = MtkUtcToUtcJd("2006-08-06T15:04:00.630996Z", jdUTC);
 *  \endcode
 */

MTKt_status MtkUtcToUtcJd(
  char utc_datetime[MTKd_DATETIME_LEN], /**< [IN] UTC Date time */
  double jdUTC[2] /**< [OUT] UTC Julian date */ )
{
  MTKt_status status_code; /* Return code of this function */
  int          scanCheck;        /* checks the return value of sscanf call */
  int          year;             /* year portion of date */
  int          month;            /* month portion of date */
  int          day;              /* day portion of date */
  int          hours;            /* hours of the given date */
  int          minutes;          /* minutes of the given date */
  double       seconds;
    
  scanCheck = sscanf(utc_datetime,"%4d-%2d-%2dT%2d:%2d:%lfZ",
		             &year, &month, &day, &hours, &minutes, &seconds);
  if (scanCheck != 6)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
    
  if (month < 1 || month > 12)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
  
  if (hours > 23)
     MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
  
  if (minutes > 59)
     MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
  
  if (seconds > 60.99999999)
  {
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
  }
  else if (seconds >= 60.0 && (minutes != 59 || hours != 23))
     MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT); 	


  if (seconds >= 60.0)
    seconds -= 1.0;

  jdUTC[1] = (hours * SECONDSperHOUR + minutes * SECONDSperMINUTE + seconds) /
               SECONDSperDAY;
  
  jdUTC[0] = julday(year, month, day) - 0.5;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
