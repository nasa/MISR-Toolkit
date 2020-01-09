/*===========================================================================
=                                                                           =
=                          MtkFileToGridList_test                           =
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
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  char filename[80];		/* HDF-EOS filename */
  char *gridlist_expected[] = {"BlueBand", "GreenBand", "RedBand", "NIRBand",
			       "BRF Conversion Factors", "GeometricParameters"};
  int num_grids;
  char **gridlist;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileToGridList");

  /* Normal Call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf");
  status = MtkFileToGridList(filename,&num_grids,&gridlist);
  if (status == MTK_SUCCESS)
  {
    if (num_grids != sizeof(gridlist_expected) /
                       sizeof(*gridlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_grids; ++i)
      if (strcmp(gridlist[i],gridlist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(num_grids, &gridlist);
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
  status = MtkFileToGridList("abcd.hdf",&num_grids,&gridlist);
  if (status == MTK_HDFEOS_GDOPEN_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileToGridList(NULL,&num_grids,&gridlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileToGridList(filename,NULL,&gridlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileToGridList(filename,&num_grids,NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  {
    char *gridlist_expected[] = {"4.4_KM_PRODUCTS"};

    strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
    status = MtkFileToGridList(filename,&num_grids,&gridlist);
    if (status == MTK_SUCCESS) {
      if (num_grids != sizeof(gridlist_expected) /
          sizeof(*gridlist_expected))
        data_ok = MTK_FALSE;
      
      for (i = 0; i < num_grids; ++i)
        if (strcmp(gridlist[i],gridlist_expected[i]) != 0) {
          data_ok = MTK_FALSE;
          break;
        }
      MtkStringListFree(num_grids, &gridlist);
    }
    
    if (status == MTK_SUCCESS && data_ok) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  status = MtkFileToGridList(filename,NULL,&gridlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileToGridList(filename,&num_grids,NULL);
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
