/*===========================================================================
=                                                                           =
=                             MtkUtcJdToTaiJd_test                          =
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
  double jdTAI[2];

  MTK_PRINT_STATUS(cn,"Testing MtkUtcJdToTaiJd");

  /* Normal Call */
  jdUTC[0] = 2452529.5;   
  jdUTC[1] = 0.63163940;
  
  status = MtkUtcJdToTaiJd(jdUTC, jdTAI);
  if (status == MTK_SUCCESS && fabs(jdTAI[0] - 2452529.5) < .0000001 && fabs(jdTAI[1] - 0.63200977) < .0000001)
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
