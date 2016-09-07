/*===========================================================================
=                                                                           =
=                             MtkUtcJdToUtc_test                              =
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
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  double jdUTC[2];
  char utc_datetime[MTKd_DATETIME_LEN];

  MTK_PRINT_STATUS(cn,"Testing MtkUtcJdToUtc");

  /* Normal Call */
  jdUTC[0] = 2453953.5;
  jdUTC[1] = 0.627785;
  
  status = MtkUtcJdToUtc(jdUTC, utc_datetime);
  if (status == MTK_SUCCESS && strcmp("2006-08-06T15:04:00.630996Z", utc_datetime))
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
    
  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
