/*===========================================================================
=                                                                           =
=                         MtkRadToDegMinSec_test                            =
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
#include <stdlib.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double rad;			/* Radians */
  int deg;			/* Degrees */
  int min;			/* Minutes */
  double sec;			/* Seconds */
  int deg_expected;		/* Expected degrees */
  int min_expected;		/* Expected minutes */
  double sec_expected;		/* Expected seconds */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRadToDegMinSec");

  /* Test normal call with positive test case */
  rad = 2.270373886;
  deg_expected = 130;
  min_expected = 4;
  sec_expected = 58.23;

  status = MtkRadToDegMinSec(rad, &deg, &min, &sec);
  if (status == MTK_SUCCESS &&
      abs(deg - deg_expected) == 0 &&
      abs(min - min_expected) == 0 &&
      fabs(sec - sec_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  rad = -1.719270468;
  deg_expected = -98;
  min_expected = 30;
  sec_expected = 24.99;

  status = MtkRadToDegMinSec(rad, &deg, &min, &sec);
  if (status == MTK_SUCCESS &&
      abs(deg - deg_expected) == 0 &&
      abs(min - min_expected) == 0 &&
      fabs(sec - sec_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkRadToDegMinSec(rad, NULL, &min, &sec);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkRadToDegMinSec(rad, &deg, NULL, &sec);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkRadToDegMinSec(rad, &deg, &min, NULL);
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
