/*===========================================================================
=                                                                           =
=                            MtkVersion_test                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include <stdio.h>

int main () {

  MTKt_boolean pass = MTK_TRUE; /* Test status */
  char *version = NULL;		/* MisrToolkit version */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkVersion");

  /* Normal test call */
  version = MtkVersion();

  if (version != NULL) {
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
