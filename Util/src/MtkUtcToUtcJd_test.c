/*===========================================================================
=                                                                           =
=                             MtkUtcToUtcJd_test                            =
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
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  double jdUTC[2];

  MTK_PRINT_STATUS(cn,"Testing MtkUtcToUtcJd");

  /* Normal Call */ 
  status = MtkUtcToUtcJd("2006-08-06T15:04:00.630996Z", jdUTC);
  if (status == MTK_SUCCESS && fabs(jdUTC[0] - 2453953.5) < .0000001 &&
      fabs(jdUTC[1] - 0.62778508) < .001)
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
