/*===========================================================================
=                                                                           =
=                             MtkTaiToTaiJd_test                            =
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
  double jdTAI[2];

  MTK_PRINT_STATUS(cn,"Testing MtkTaiToTaiJd");

  /* Normal Call */
  status = MtkTaiToTaiJd(429030246.630996, jdTAI);
  if (status == MTK_SUCCESS && fabs(jdTAI[0] - 2453953.5) < .0000001 &&
      fabs(jdTAI[1] - 0.628167) < .0000001)
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
