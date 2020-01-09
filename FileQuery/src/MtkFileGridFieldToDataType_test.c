/*===========================================================================
=                                                                           =
=                     MtkFileGridFieldToDataType_test                       =
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
  char fieldname[80];		/* HDF-EOS fieldname */
  MTKt_DataType datatype;	/* Datatype */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridFieldToDataType");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  status =MtkFileGridFieldToDataType(filename, gridname, fieldname, &datatype);
  if (status == MTK_SUCCESS && datatype == MTKe_int16) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(filename, "");
  strcpy(gridname, "");
  strcpy(fieldname, "");

  status =MtkFileGridFieldToDataType(filename, gridname, fieldname, &datatype);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "");
  strcpy(fieldname, "");

  status =MtkFileGridFieldToDataType(filename, gridname, fieldname, &datatype);
  if (status == MTK_HDFEOS_GDATTACH_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P039_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "");

  status =MtkFileGridFieldToDataType(filename, gridname, fieldname, &datatype);
  if (status == MTK_HDFEOS_GDFIELDINFO_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileGridFieldToDataType(NULL, gridname, fieldname, &datatype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, NULL, fieldname, &datatype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, gridname, NULL, &datatype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, gridname, fieldname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Bi-Hemispherical_Reflectance");

  status =MtkFileGridFieldToDataType(filename, gridname, fieldname, &datatype);
  if (status == MTK_SUCCESS && datatype == MTKe_uint8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, NULL, fieldname, &datatype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, gridname, NULL, &datatype);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDataType(filename, gridname, fieldname, NULL);
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
