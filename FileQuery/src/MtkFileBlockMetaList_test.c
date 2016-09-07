/*===========================================================================
=                                                                           =
=                        MtkFileBlockMetaList_test                          =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  char filename[200];		/* HDF-EOS filename */
  char *blockmetalist_expected[] = {"PerBlockMetadataCommon",
                                    "PerBlockMetadataRad",
                                    "PerBlockMetadataTime" };
  char *blockmetalist_expected1[] = {"PerBlockMetadataCommon",
                                     "PerBlockMetadataAGP" };
  char *blockmetalist_expected2[] = {"PerBlockMetadataCommon",
  	                                 "Common Per Block Metadata",
                                     "PerBlockMetadataTime" };
  char *blockmetalist_expected3[] = {"PerBlockMetadataCommon",
  	                                 "PerBlockMetadataGeoParm" };                           
  int nblockmeta;
  char **blockmetalist;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileBlockMetaList");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");

  status = MtkFileBlockMetaList(filename,&nblockmeta,&blockmetalist);
  if (status == MTK_SUCCESS)
  {
    if (nblockmeta != sizeof(blockmetalist_expected) /
                       sizeof(*blockmetalist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < nblockmeta; ++i)
      if (strcmp(blockmetalist[i],blockmetalist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nblockmeta, &blockmetalist);
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
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P177_F01_24.hdf");

  status = MtkFileBlockMetaList(filename,&nblockmeta,&blockmetalist);
  if (status == MTK_SUCCESS)
  {
    if (nblockmeta != sizeof(blockmetalist_expected1) /
                       sizeof(*blockmetalist_expected1))
      data_ok = MTK_FALSE;

    for (i = 0; i < nblockmeta; ++i)
      if (strcmp(blockmetalist[i],blockmetalist_expected1[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nblockmeta, &blockmetalist);
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
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");

  status = MtkFileBlockMetaList(filename,&nblockmeta,&blockmetalist);
  if (status == MTK_SUCCESS)
  {
    if (nblockmeta != sizeof(blockmetalist_expected2) /
                       sizeof(*blockmetalist_expected2))
      data_ok = MTK_FALSE;

    for (i = 0; i < nblockmeta; ++i)
      if (strcmp(blockmetalist[i],blockmetalist_expected2[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nblockmeta, &blockmetalist);
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
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");

  status = MtkFileBlockMetaList(filename,&nblockmeta,&blockmetalist);
  if (status == MTK_SUCCESS)
  {
    if (nblockmeta != sizeof(blockmetalist_expected3) /
                       sizeof(*blockmetalist_expected3))
      data_ok = MTK_FALSE;

    for (i = 0; i < nblockmeta; ++i)
      if (strcmp(blockmetalist[i],blockmetalist_expected3[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nblockmeta, &blockmetalist);
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

  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
  
  status = MtkFileBlockMetaList(filename,&nblockmeta,&blockmetalist);
  if (status == MTK_HDF_HDFOPEN_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  } 

  /* Argument Checks */
  status = MtkFileBlockMetaList(NULL,&nblockmeta,&blockmetalist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  status = MtkFileBlockMetaList(filename,NULL,&blockmetalist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaList(filename,&nblockmeta,NULL);
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
