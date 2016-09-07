/*===========================================================================
=                                                                           =
=                         MtkDateTimeToJulian_test                          =
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
  double julian;

  MTK_PRINT_STATUS(cn,"Testing MtkDateTimeToJulian");

  /* Normal Call */
  status = MtkDateTimeToJulian("2002-05-02T02:00:00Z",&julian);
  if (status == MTK_SUCCESS && fabs(julian - 2452396.58333) < 0.0001)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkDateTimeToJulian("2002-05-02T02:00:00",&julian);
  if (status == MTK_SUCCESS && fabs(julian - 2452396.58333) < 0.0001)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkDateTimeToJulian("2002-05-02T",&julian);
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkDateTimeToJulian(NULL,&julian);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkDateTimeToJulian("2002-05-02T02:00:00",NULL);
  if (status == MTK_NULLPTR)
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
