/*===========================================================================
=                                                                           =
=                      MtkFileGridFieldToDimList_test                       =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  char filename[MAXSTR];	/* HDF-EOS filename */
  char gridname[MAXSTR];	/* HDF-EOS gridname */
  char fieldname[MAXSTR];	/* HDF-EOS fieldname */
  char *dimlist_expected[] = { "NBandDim", "NCamDim" };
  char *dimlist_expected2[] = { "Band_Dim", "Camera_Dim" };
  int dimsize_expected[] = { 4, 9 };
  char **dimlist;
  int *dimsize;
  int num_dims;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileGridFieldToDimList");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(gridname, "SubregParamsLnd");
  strcpy(fieldname, "LandHDRF");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS)
  {
    if (num_dims != sizeof(dimlist_expected) /
                       sizeof(*dimlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_dims; ++i)
      if (strcmp(dimlist[i],dimlist_expected[i]) != 0 ||
	  dimsize[i] != dimsize_expected[i])
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(num_dims, &dimlist);
    free(dimsize);
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

  /* Normal test call */
  strcpy(fieldname, "NDVI");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS &&
      num_dims == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(fieldname, "blah");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_HDFEOS_GDFIELDINFO_FAILED &&
      num_dims == -1)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Radiance/RDQI");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS &&
      num_dims == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Radiance");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS &&
      num_dims == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red RDQI");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS &&
      num_dims == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  strcpy(gridname, "RedBand");
  strcpy(fieldname, "Red Brf");

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS &&
      num_dims == 0)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileGridFieldToDimList(NULL,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,NULL,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,NULL,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,NULL,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     NULL,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist, NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Hemispherical_Directional_Reflectance_Factor");
  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_SUCCESS) {
    int num_dims_expect = sizeof(dimlist_expected2) / sizeof(*dimlist_expected2);
    if (num_dims != num_dims_expect) {
      data_ok = MTK_FALSE;
    }

    for (i = 0; i < num_dims; ++i) {
      if (strcmp(dimlist[i],dimlist_expected2[i]) != 0 ||
          dimsize[i] != dimsize_expected[i])
      {
        data_ok = MTK_FALSE;
        break;
      }
    }
    MtkStringListFree(num_dims, &dimlist);
    free(dimsize);
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

  status = MtkFileGridFieldToDimList(filename,NULL,fieldname,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,NULL,&num_dims,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,NULL,
                                     &dimlist,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     NULL,&dimsize);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileGridFieldToDimList(filename,gridname,fieldname,&num_dims,
                                     &dimlist, NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
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
