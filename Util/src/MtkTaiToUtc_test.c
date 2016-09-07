/*===========================================================================
=                                                                           =
=                             MtkTaiToUtc_test                              =
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

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char utc_datetime[MTKd_DATETIME_LEN];

  MTK_PRINT_STATUS(cn,"Testing MtkTaiToUtc");

  /* Normal Call */
  status = MtkTaiToUtc(305996978.644320, utc_datetime);
  if (status == MTK_SUCCESS && strcmp("2002-09-12T15:09:33.644320Z", utc_datetime) == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkTaiToUtc(429030246.630996, utc_datetime);
  if (status == MTK_SUCCESS && strcmp("2006-08-06T15:04:00.630996Z", utc_datetime) == 0)
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
