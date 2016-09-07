/*===========================================================================
=                                                                           =
=                             MtkDmsToRad_test                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUnitConv.h"
#include "MisrError.h"
#include <math.h>
#include <stdio.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double dms;			/* Packed degrees, minutes, seconds */
  double rad;			/* Radians */
  double rad_expected;		/* Expected radians */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDmsToRad");

  /* Test normal call with positive test case */
  dms = 130004058.23;
  rad_expected = 2.270373886;

  status = MtkDmsToRad(dms, &rad);
  if (status == MTK_SUCCESS &&
      fabs(rad - rad_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  dms = -98030024.99;
  rad_expected = -1.719270468;

  status = MtkDmsToRad(dms, &rad);
  if (status == MTK_SUCCESS &&
      fabs(rad - rad_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDmsToRad(dms, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
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
