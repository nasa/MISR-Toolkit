/*===========================================================================
=                                                                           =
=                       MtkFileGridToResolution_test                        =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  int resolution;		/* Resolution */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridToResolution");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");

  status = MtkFileGridToResolution(filename, gridname, &resolution);
  if (status == MTK_SUCCESS && resolution == 1100) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(filename, "");
  strcpy(gridname, "");

  status = MtkFileGridToResolution(filename, gridname, &resolution);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "");

  status = MtkFileGridToResolution(filename, gridname, &resolution);
  if (status == MTK_HDFEOS_GDATTACH_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileGridToResolution(NULL, gridname, &resolution);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToResolution(filename, NULL, &resolution);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToResolution(filename, gridname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(gridname, "4.4_KM_PRODUCTS");

  status = MtkFileGridToResolution(filename, gridname, &resolution);
  if (status == MTK_SUCCESS && resolution == 4400) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(gridname, "");
  status = MtkFileGridToResolution(filename, gridname, &resolution);
  if (status == MTK_NETCDF_READ_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToResolution(filename, NULL, &resolution);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridToResolution(filename, gridname, NULL);
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
