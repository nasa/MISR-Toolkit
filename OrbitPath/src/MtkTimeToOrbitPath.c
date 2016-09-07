/*===========================================================================
=                                                                           =
=                            MtkTimeToOrbitPath                             =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/** \brief Given time return orbit number and path number
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  Time Format: YYYY-MM-DDThh:mm:ssZ (ISO 8601)
 *
 *  \par Example:
 *  In this example, we get the orbit and path numbers for the time 2002-05-02 02:00:00 UTC
 *
 *  \code
 *  status = MtkTimeToOrbitPath("2002-05-02T02:00:00Z", &orbit, &path);
 *  \endcode
 */

MTKt_status MtkTimeToOrbitPath(
  const char *datetime, /**< [IN] Date Time */
  int *orbit, /**< [OUT] Orbit Number */
  int *path /**< [OUT] Path */)
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  const double JNref[] = MISR_ORBIT_REF;
  const double JNref_995 = MISR_ORBIT_REF_995;
  double j;
  int year, month, day, hour, min, sec;
  int num_values;
  int ref_num;

  if (datetime == NULL || orbit == NULL || path == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  num_values = sscanf(datetime,"%4d-%2d-%2dT%2d:%2d:%2d",&year,&month,
                      &day,&hour,&min,&sec);

  if (num_values != 6)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  if (year < 2000)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  if (year == 2000 && month < 2)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  if (year == 2000 && month == 2 && day < 24)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkCalToJulian(year,month,day,hour,min,sec,&j);
  MTK_ERR_COND_JUMP(status);

  /* Implement special code for orbits 995-1000 */
  
  if (j < JNref[0]) {
    *orbit = (int)( (233.0 / 16.0 * ( j - JNref_995)) 
                   +(995));
  } else {
    /* The JNref table is to account/adjust for drift in the orbit.
       We need to find the closest entry in this table to our orbit. We do
       this by determining the approximate orbit first and then using it
       to find a more accurate entry in JNref table (ie ref_num = orbit/MISR_ORBIT_REF_DT). 
       Then we compute real orbit number. */

    ref_num = (int)((233.0 / 16.0 * ( j - JNref[0])) + MISR_ORBIT_REF_DT /*Nref*/) 
             / MISR_ORBIT_REF_DT;
    if (ref_num > sizeof JNref / sizeof *JNref)
      ref_num = sizeof JNref / sizeof *JNref;

    *orbit = (int)( (233.0 / 16.0 * ( j - JNref[ref_num - 1])) 
                   +(ref_num * MISR_ORBIT_REF_DT) );
  }

  status = MtkOrbitToPath(*orbit,path);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

 ERROR_HANDLE:
  return status_code;
}

