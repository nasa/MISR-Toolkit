/*===========================================================================
=                                                                           =
=                            MtkNCVarId_test                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2017, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include "MisrError.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  char filename[80];    /* HDF-EOS filename */
  int cn = 0;     /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkNCVarId");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");

  int ncid;
  int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
  if (nc_status == NC_NOERR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  int group_id;
  nc_status = nc_inq_grp_ncid(ncid, "4.4_KM_PRODUCTS", &group_id);
  if (nc_status == NC_NOERR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MTKt_ncvarid var;
  status = MtkNCVarId(group_id, "AUXILIARY/Rayleigh_Optical_Depth", &var);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  nc_status = nc_close(ncid);
  if (nc_status == NC_NOERR) {
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
