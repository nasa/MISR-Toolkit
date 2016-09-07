/*===========================================================================
=                                                                           =
=                               MtkUtcToTai                                 =
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

/** \brief Convert UTC to TAI93
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert from UTC 2006-08-06T15:04:00.630996Z to TAI93.
 *
 *  \code
 *  status = MtkUtcToTai("2006-08-06T15:04:00.630996Z" &tai93);
 *  \endcode
 */

MTKt_status MtkUtcToTai(
  char utc_datetime[MTKd_DATETIME_LEN], /**< [IN] UTC Date time */
  double *secTAI93 /**< [OUT] TAI Time */ )
{
  MTKt_status status_code; /* Return code of this function */
  double jdTAI[2];     /* TAI Julian Date equivalent of input time */
  double jdUTC[2];     /* UTC Julian Date equivalent of input time */
  int scanCheck;        /* checks the return value of sscanf call */
  int year;             /* year portion of date */
  int month;            /* month portion of date */
  int day;              /* day portion of date */
  int hours;            /* hours of the given date */
  int minutes;          /* minutes of the given date */
  double seconds;
  
  if (secTAI93 == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
    
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
  
  MtkUtcToUtcJd(utc_datetime, jdUTC);
  MtkUtcJdToTaiJd(jdUTC, jdTAI);
  MtkTaiJdToTai(jdTAI, secTAI93);
 
  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
