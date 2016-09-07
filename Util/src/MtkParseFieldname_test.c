/*===========================================================================
=                                                                           =
=                          MtkParseFieldname_test                           =
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
#include "math.h"
#include <float.h>
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  char fieldname[MAXSTR];	/* HDF-EOS fieldname */
  char fieldname_expected[MAXSTR]; /* HDF-EOS fieldname */
  char *basefieldname;		/* HDF-EOS fieldname */
  int ndim;			/* Number of dimensions */
  int *dimlist;			/* Dimension list */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkParseFieldname");

  /* Normal test call */
  strcpy(fieldname, "AveSceneElev");
  strcpy(fieldname_expected, "AveSceneElev");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "AveSceneElev[6]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 1 &&
      dimlist[0] == 6) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "AveSceneElev[3][4]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 2 &&
      dimlist[0] == 3 &&
      dimlist[1] == 4) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Raw AveSceneElev[3][4]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 2 &&
      dimlist[0] == 3 &&
      dimlist[1] == 4) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Flag AveSceneElev[3][4]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 2 &&
      dimlist[0] == 3 &&
      dimlist[1] == 4) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call - [a] is ignored */
  strcpy(fieldname, "AveSceneElev[a]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call - [] is ignored */
  strcpy(fieldname, "AveSceneElev[]");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, fieldname_expected) == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red DN");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red Equivalent Reflectance");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red Brf");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red Radiance/RDQI");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red Radiance");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "Red RDQI");

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
  if (status == MTK_SUCCESS &&
      strcmp(basefieldname, "Red Radiance/RDQI") == 0 &&
      ndim == 0) {
    free(basefieldname);
    free(dimlist);
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

   /* Normal test call */
   strcpy(fieldname, "RegBestEstimateSpectralOptDepth[2]");
 
   status = MtkParseFieldname(fieldname, &basefieldname, &ndim, &dimlist);
   if (status == MTK_SUCCESS &&
       strcmp(basefieldname, "RegBestEstimateSpectralOptDepth") == 0 &&
       ndim == 1 &&
       dimlist[0] == 2) {
     free(basefieldname);
     free(dimlist);
     MTK_PRINT_STATUS(cn,".");
   } else {
     MTK_PRINT_STATUS(cn,"*");
     pass = MTK_FALSE;
   }
 
  /* Argument Checks */
  status = MtkParseFieldname(NULL, &basefieldname, &ndim, &dimlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkParseFieldname(fieldname, NULL, &ndim, &dimlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkParseFieldname(fieldname, &basefieldname, NULL, &dimlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkParseFieldname(fieldname, &basefieldname, &ndim, NULL);
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
