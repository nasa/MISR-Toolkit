/*===========================================================================
=                                                                           =
=                           MtkFieldAttrList_test                           =
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
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  int num_attrs;                /* Number of attributes */
  char **attrlist;              /* Attribute List */
  char filename[80];		/* HDF-EOS filename */
  char fieldname[80]; /* HDF SDS Field Name */
  int cn = 0;			/* Column number */
  int i;
  
  char *attrlist_expected[] = {"_FillValue","scale_factor",
  "add_offset","valid_min",
  "valid_max","units",
  "long_name"};

  MTK_PRINT_STATUS(cn,"Testing MtkFieldAttrList");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(fieldname, "CloudMotionCrossTrack");

  status = MtkFieldAttrList(filename, fieldname, &num_attrs, &attrlist);
  if (status == MTK_SUCCESS)
  {
    if (num_attrs != sizeof(attrlist_expected) / sizeof(*attrlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_attrs; ++i)
      if (strcmp(attrlist[i],attrlist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
	break;
      }
    MtkStringListFree(num_attrs, &attrlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* File doesn't exists */
  strcpy(filename, "abcd.hdf");

  status = MtkFieldAttrList(filename, fieldname, &num_attrs, &attrlist);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  status = MtkFieldAttrList(NULL, fieldname, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrList(filename, NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrList(filename, fieldname, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrList(filename, fieldname, &num_attrs, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */

  {
    char *attrlist_expected[] = {"_FillValue","coordinates",
                                 "units", "standard_name", "long_name"};

    strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
    strcpy(fieldname, "Latitude");
    
    status = MtkFieldAttrList(filename, fieldname, &num_attrs, &attrlist);
    if (status == MTK_SUCCESS) {
      if (num_attrs != sizeof(attrlist_expected) / sizeof(*attrlist_expected))
        data_ok = MTK_FALSE;
        
      for (i = 0; i < num_attrs; ++i)
        if (strcmp(attrlist[i],attrlist_expected[i]) != 0) {
          data_ok = MTK_FALSE;
          break;
        }
      MtkStringListFree(num_attrs, &attrlist);
    }
    
    if (status == MTK_SUCCESS && data_ok) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  /* Argument Checks */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  status = MtkFieldAttrList(NULL, fieldname, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrList(filename, NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrList(filename, fieldname, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrList(filename, fieldname, &num_attrs, NULL);
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
