/*===========================================================================
=                                                                           =
=                             MtkRadToDms_test                              =
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
  double dms_expected;		/* Expected dms */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRadToDms");

  /* Test normal call with positive test case */
  rad = 2.270373886;
  dms_expected = 130004058.23;

  status = MtkRadToDms(rad, &dms);
  if (status == MTK_SUCCESS &&
      fabs(dms - dms_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  rad = -1.719270468;
  dms_expected = -98030024.99;

  status = MtkRadToDms(rad, &dms);
  if (status == MTK_SUCCESS &&
      fabs(dms - dms_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkRadToDms(rad, NULL);
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
