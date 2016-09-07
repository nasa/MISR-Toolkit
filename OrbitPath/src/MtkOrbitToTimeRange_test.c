/*===========================================================================
=                                                                           =
=                         MtkOrbitToTimeRange_test                          =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrError.h"
#include <stdio.h>
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char start_time[MTKd_DATETIME_LEN];
  char end_time[MTKd_DATETIME_LEN];

  MTK_PRINT_STATUS(cn,"Testing MtkOrbitToTimeRange");

  /* Normal Call */
  status = MtkOrbitToTimeRange(30000,start_time,end_time);
  if (status == MTK_SUCCESS && strcmp("2005-08-08T10:10:00Z",start_time) == 0 &&
      strcmp("2005-08-08T11:48:53Z",end_time) == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkOrbitToTimeRange(60000,start_time,end_time);
  if (status == MTK_SUCCESS)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkOrbitToTimeRange(996,start_time,end_time);
  if (status == MTK_SUCCESS && strcmp("2000-02-24T18:11:53Z",start_time) == 0 &&
      strcmp("2000-02-24T19:50:46Z",end_time) == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkOrbitToTimeRange(994,start_time,end_time);
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkOrbitToTimeRange(30000,NULL,end_time);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkOrbitToTimeRange(30000,start_time,NULL);
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
