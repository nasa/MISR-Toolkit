/*===========================================================================
=                                                                           =
=                              MtkRadToDd_test                              =
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
  double rad;			/* Radians */
  double dd;			/* Decimal degrees */
  double dd_expected;		/* Expected decimal degrees */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRadToDd");

  /* Test normal call with positive test case */
  rad = 2.270373886;
  dd_expected = 130.08284167;

  status = MtkRadToDd(rad, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  rad = -1.719270468;
  dd_expected = -98.50694167;

  status = MtkRadToDd(rad, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkRadToDd(rad, NULL);
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
