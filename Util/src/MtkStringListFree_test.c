/*===========================================================================
=                                                                           =
=                          MtkStringListFree_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int strcnt;			/* String count */
  char **strlist = NULL;	/* String list */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkStringListFree");

  /* Normal test call */
  strcnt = 3;
  strlist = (char **)calloc(strcnt, sizeof(char *));
  strlist[0] = (char *)malloc(10);
  strlist[1] = (char *)malloc(20);
  strlist[2] = (char *)malloc(30);

  status = MtkStringListFree(strcnt, &strlist);
  if (status == MTK_SUCCESS && strlist == NULL) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkStringListFree(strcnt, NULL);
  if (status == MTK_SUCCESS) {
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
