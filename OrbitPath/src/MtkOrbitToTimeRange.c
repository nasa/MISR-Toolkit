/*===========================================================================
=                                                                           =
=                           MtkOrbitToTimeRange                             =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/** \brief Given a orbit number return time
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  Time Format: YYYY-MM-DDThh:mm:ssZ (ISO 8601)
 *
 *  \par Example:
 *  In this example, we get the time for orbit number 26000
 *
 *  \code
 *  status = MtkOrbitToTimeRange(26000, start_time, end_time);
 *  \endcode
 */

MTKt_status MtkOrbitToTimeRange(
  int orbit, /**< [IN] Orbit Number */
  char start_time[MTKd_DATETIME_LEN], /**< [OUT] Start Time */
  char end_time[MTKd_DATETIME_LEN] /**< [OUT] End Time */ )
{
  MTKt_status status;	   /* Return status */
  MTKt_status status_code; /* Return code of this function */
  const double JNref[] = MISR_ORBIT_REF;
  const double JNref_995 = MISR_ORBIT_REF_995;  
  double jn;
  int year;
  int month;
  int day;
  int hour;
  int min;
  int sec;
  int ref_num;

  if (orbit < 995)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  if (start_time == NULL || end_time == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Determine Start Time */
  if (orbit < MISR_ORBIT_REF_DT) {
    jn = JNref_995 + 16.0 / 233.0 * ( orbit - 995 );
  } else {
    ref_num = orbit / MISR_ORBIT_REF_DT;
    if (ref_num > sizeof JNref / sizeof *JNref)
      ref_num = sizeof JNref / sizeof *JNref;

    jn = JNref[ref_num - 1] + 16.0 / 233.0 * ( orbit - ref_num * MISR_ORBIT_REF_DT );
  }
  status = MtkJulianToCal(jn,&year,&month,&day,&hour,&min,&sec);
  MTK_ERR_COND_JUMP(status);

  sprintf(start_time,"%04d-%02d-%02dT%02d:%02d:%02dZ",year,month,day,hour,min,sec);  

  /* Determine End Time */
  if (orbit < MISR_ORBIT_REF_DT) {
    jn = JNref_995 + 16.0 / 233.0 * ( (orbit + 1) - 995 );
  } else {
    ref_num = (orbit + 1) / MISR_ORBIT_REF_DT;
    if (ref_num > sizeof JNref / sizeof *JNref)
      ref_num = sizeof JNref / sizeof *JNref;

    jn = JNref[ref_num - 1] + 16.0 / 233.0 * ( (orbit + 1) - ref_num * MISR_ORBIT_REF_DT );
  }
  status = MtkJulianToCal(jn,&year,&month,&day,&hour,&min,&sec);
  MTK_ERR_COND_JUMP(status);

  sprintf(end_time,"%04d-%02d-%02dT%02d:%02d:%02dZ",year,month,day,hour,min,sec);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}

