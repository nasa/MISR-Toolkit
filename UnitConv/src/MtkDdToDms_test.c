/*===========================================================================
=                                                                           =
=                             MtkDdToDms_test                               =
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
  double dd;			/* Decimal degrees */
  double dms_expected;		/* Expected dms */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDdToDms");

  /* Test normal call with positive test case */
  dd = 130.08284167;
  dms_expected = 130004058.23;

  status = MtkDdToDms(dd, &dms);
  if (status == MTK_SUCCESS &&
      fabs(dms - dms_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  dd = -98.50694167;
  dms_expected = -98030024.99;

  status = MtkDdToDms(dd, &dms);
  if (status == MTK_SUCCESS &&
      fabs(dms - dms_expected) < 0.01) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for degress out of bounds */
  dd = -365.99;

  status = MtkDdToDms(dd, &dms);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDdToDms(dd, NULL);
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
