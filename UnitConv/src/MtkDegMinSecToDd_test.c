/*===========================================================================
=                                                                           =
=                          MtkDegMinSecToDd_test                            =
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
  int deg;			/* Degrees */
  int min;			/* Minutes */
  double sec;			/* Seconds */
  double dd;			/* Decimal degrees */
  double dd_expected;		/* Expected dd */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDegMinSecToDd");

  /* Test normal call with positive test case */
  deg = 130;
  min = 4;
  sec = 58.23;
  dd_expected = 130.08284167;

  status = MtkDegMinSecToDd(deg, min, sec, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test normal call with negative test case */
  deg = -98;
  min = 30;
  sec = 24.99;
  dd_expected = -98.50694167;

  status = MtkDegMinSecToDd(deg, min, sec, &dd);
  if (status == MTK_SUCCESS &&
      fabs(dd - dd_expected) < 0.0000001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for degress out of bounds */
  deg = -498;
  min = 30;
  sec = 24.99;

  status = MtkDegMinSecToDd(deg, min, sec, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for minutes out of bounds */
  deg = 1;
  min = 80;
  sec = 24.99;

  status = MtkDegMinSecToDd(deg, min, sec, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Test for seconds out of bounds */
  deg = 1;
  min = 8;
  sec = 60.01;

  status = MtkDegMinSecToDd(deg, min, sec, &dd);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDegMinSecToDd(deg, min, sec, NULL);
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
