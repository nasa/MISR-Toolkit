/*===========================================================================
=                                                                           =
=                             MtkDmsToDd_test                               =
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
#include <float.h>
#include <stdio.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double dms;			/* Packed degrees, minutes, seconds */
  double dd;			/* Decimal degrees */
  double dd_expected;		/* Expected dd */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDmsToDd");

  /* Test normal call with positive test case */
  dms = 130004058.23;
  dd_expected = 130.08284167;
   
  status = MtkDmsToDd(dms, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  dms = -98030024.99;
  dd_expected = -98.50694167;

  status = MtkDmsToDd(dms, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for degress out of bounds */
  dms = -498030024.99;

  status = MtkDmsToDd(dms, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for minutes out of bounds */
  dms = 1080024.99;

  status = MtkDmsToDd(dms, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for seconds out of bounds */
  dms = 1008060.01;

  status = MtkDmsToDd(dms, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDmsToDd(dms, NULL);
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
