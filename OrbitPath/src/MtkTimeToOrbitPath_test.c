/*===========================================================================
=                                                                           =
=                         MtkTimeToOrbitPath_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrError.h"
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  int orbit;
  int path;

  MTK_PRINT_STATUS(cn,"Testing MtkTimeToOrbitPath");

  /* Normal Call */
  status = MtkTimeToOrbitPath("2001-08-07T04:00:00Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 8705 && path == 123)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2007-02-15T23:47:05Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 38105 && path == 96)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2007-02-15T22:07:12Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 38103 && path == 64)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2007-02-16T00:00:00Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 38105 && path == 96)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2007-02-17T00:00:00Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 38119 && path == 87)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2006-09-24T08:56:59Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 35999 && path == 185)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2006-09-24T10:35:00Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 35999 && path == 185)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeToOrbitPath("2006-09-24T10:35:30Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 36000 && path == 201)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Normal Call */
  status = MtkTimeToOrbitPath("2000-02-24T18:21:53Z",&orbit,&path);
  if (status == MTK_SUCCESS && orbit == 996 && path == 36)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkTimeToOrbitPath("2001-08-07T04:00",&orbit,&path);
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeToOrbitPath(NULL,&orbit,&path);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeToOrbitPath("2001-08-07T04:00:00Z",NULL,&path);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeToOrbitPath("2001-08-07T04:00:00Z",&orbit,NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeToOrbitPath("1999-08-07T04:00:00Z",&orbit,&path);
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeToOrbitPath("2000-02-07T04:00:00Z",&orbit,&path);
  if (status == MTK_BAD_ARGUMENT)
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
