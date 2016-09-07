/*===========================================================================
=                                                                           =
=                        MtkCoreMetaDataFree_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MtkCoreMetaData metadata = MTK_CORE_METADATA_INIT;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkCoreMetaDataFree");

  /* Normal test call */
  metadata.data.s = (char **)calloc(3, sizeof(char*));
  metadata.data.s[0] = (char*)malloc(10);
  metadata.data.s[1] = (char*)malloc(20);
  metadata.data.s[2] = (char*)malloc(30);
  metadata.dataptr = metadata.data.s;
  metadata.num_values = 3;

  status = MtkCoreMetaDataFree(&metadata);
  if (status == MTK_SUCCESS && metadata.data.s == NULL &&
      metadata.dataptr == NULL && metadata.num_values == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkCoreMetaDataFree(NULL);
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
