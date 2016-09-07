/*===========================================================================
=                                                                           =
=                           MtkFileAttrList_test                            =
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
  int cn = 0;			/* Column number */
  int i;
  char *attrlist_expected[] = {"HDFEOSVersion", "StructMetadata.0",
                               "Path_number", "AGP_version_id",
                               "DID_version_id", "Number_blocks",
                               "Ocean_blocks_size", "Ocean_blocks.count",
			       "Ocean_blocks.numbers",
			       "SOM_parameters.som_ellipsoid.a",
			       "SOM_parameters.som_ellipsoid.e2",
			       "SOM_parameters.som_orbit.aprime",
			       "SOM_parameters.som_orbit.eprime",
			       "SOM_parameters.som_orbit.gama",
			       "SOM_parameters.som_orbit.nrev",
			       "SOM_parameters.som_orbit.ro",
			       "SOM_parameters.som_orbit.i",
			       "SOM_parameters.som_orbit.P2P1",
			       "SOM_parameters.som_orbit.lambda0",
			       "Origin_block.ulc.x",
			       "Origin_block.ulc.y",
			       "Origin_block.lrc.x",
			       "Origin_block.lrc.y",
			       "Start_block", "End block",
			       "Cam_mode", "Num_local_modes",
			       "Local_mode_site_name",
			       "Orbit_QA", "Camera", "coremetadata"};

  MTK_PRINT_STATUS(cn,"Testing MtkFileAttrList");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");

  status = MtkFileAttrList(filename, &num_attrs, &attrlist);
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

  status = MtkFileAttrList(filename, &num_attrs, &attrlist);
  if (status == MTK_HDF_SDSTART_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileAttrList(NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  status = MtkFileAttrList(filename, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrList(filename, &num_attrs, NULL);
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
