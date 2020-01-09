/*===========================================================================
=                                                                           =
=                           MtkGridAttrList_test                            =
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
  char gridname[80];		/* HDF-EOS attrname */
  int cn = 0;			/* Column number */
  int i;
  char *attrlist_expected[] = {"Block_size.resolution_x",
			       "Block_size.resolution_y",
			       "Block_size.size_x",
			       "Block_size.size_y",
			       "Scale factor",
			       "std_solar_wgted_height",
			       "SunDistanceAU"};

  MTK_PRINT_STATUS(cn,"Testing MtkGridAttrList");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(gridname, "RedBand");

  status = MtkGridAttrList(filename, gridname, &num_attrs, &attrlist);
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

  status = MtkGridAttrList(filename, gridname, &num_attrs, &attrlist);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkGridAttrList(NULL, gridname, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  status = MtkGridAttrList(filename, NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  status = MtkGridAttrList(filename, gridname, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkGridAttrList(filename, gridname, &num_attrs, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  {
    char *attrlist_expected[] = {
      "GCTP_projection_parameters",
      "resolution_in_meters",
      "block_size_in_lines",
      "block_size_in_samples"
    };

    strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
    strcpy(gridname, "4.4_KM_PRODUCTS");

    status = MtkGridAttrList(filename, gridname, &num_attrs, &attrlist);
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

  status = MtkGridAttrList(filename, NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkGridAttrList(filename, gridname, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkGridAttrList(filename, gridname, &num_attrs, NULL);
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
