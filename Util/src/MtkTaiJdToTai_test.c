/*===========================================================================
=                                                                           =
=                             MtkTaiJdToTai_test                            =
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
  double secTAI93;
  double jdTAI[2];

  MTK_PRINT_STATUS(cn,"Testing MtkTaiJdToTai");

  /* Normal Call */
  jdTAI[0] = 2453953.500000;
  jdTAI[1] = 0.62816703;
  
  status = MtkTaiJdToTai(jdTAI, &secTAI93);
  if (status == MTK_SUCCESS && fabs(secTAI93 - 429030246.630996) < .001)
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
