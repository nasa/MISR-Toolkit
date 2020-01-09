/*===========================================================================
=                                                                           =
=                           MtkFileVersion_test                             =
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
  int cn = 0;			/* Column number */
  char filename[80];		/* HDF filename */
  char fileversion[100]; 	/* File version */

  MTK_PRINT_STATUS(cn,"Testing MtkFileVersion");

  /* Normal test call */
  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  status = MtkFileVersion(filename, fileversion);
  if (status == MTK_SUCCESS && strcmp(fileversion,"F01_24") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_GRP_RCCM_GM_P037_O012049_AN_F01_0011.hdf");
  status = MtkFileVersion(filename, fileversion);
  if (status == MTK_SUCCESS && strcmp(fileversion,"F01_0011") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_TC_ALBEDO_P037_O029058_F04_0007.hdf");
  status = MtkFileVersion(filename, fileversion);
  if (status == MTK_SUCCESS && strcmp(fileversion,"F04_0007") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  status = MtkFileVersion(filename, fileversion);
  if (status == MTK_SUCCESS && strcmp(fileversion,"F06_0017") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */ 
  strcpy(filename,"../Mtk_testdata/in/abcd.hdf"); 
  status = MtkFileVersion(filename, fileversion); 
  if (status == MTK_HDF_SDSTART_FAILED) { 
    MTK_PRINT_STATUS(cn,"."); 
  } else { 
    MTK_PRINT_STATUS(cn,"*"); 
    pass = MTK_FALSE; 
  } 

  /* Argument Checks */
  status = MtkFileVersion(NULL, fileversion);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename,"../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  status = MtkFileVersion(filename, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  status = MtkFileVersion(filename, fileversion);
  if (status == MTK_SUCCESS && strcmp(fileversion,"F13_23") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileVersion(filename, NULL);
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
