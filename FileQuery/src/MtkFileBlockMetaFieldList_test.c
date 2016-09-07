/*===========================================================================
=                                                                           =
=                      MtkFileBlockMetaFieldList_test                       =
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
  char blockmetaname[200];	/* HDF-EOS gridname */
  char *fieldlist_expected[] = {"Block_number", "Ocean_flag",
			                    "Block_coor_ulc_som_meter.x",
			                    "Block_coor_ulc_som_meter.y",
			                    "Block_coor_lrc_som_meter.x",
			                    "Block_coor_lrc_som_meter.y",
			                    "Data_flag" };
  char *fieldlist_expected1[] = {"Point_elev_offset.x", "Point_elev_offset.y",
  	                             "ULC_latitude", "ULC_longitude",
  	                             "ULC_som_meter.x", "ULC_som_meter.y",
  	                             "ULC_som_pixel.x", "ULC_som_pixel.y",
  	                             "Ave_block_elev" };
  char *fieldlist_expected2[] = {"Geometric DQI" };
  char *fieldlist_expected3[] = {"SunDistance" };                           
  int nfields;
  char **fieldlist;
  int i;
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileBlockMetaFieldList");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(blockmetaname, "PerBlockMetadataCommon");

  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (nfields != sizeof(fieldlist_expected) /
                       sizeof(*fieldlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < nfields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nfields, &fieldlist);
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
  strcpy(blockmetaname, "PerBlockMetadataAGP");
  
  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (nfields != sizeof(fieldlist_expected1) /
                       sizeof(*fieldlist_expected1))
      data_ok = MTK_FALSE;

    for (i = 0; i < nfields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected1[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nfields, &fieldlist);
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
  strcpy(blockmetaname, "Common Per Block Metadata");
  
  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (nfields != sizeof(fieldlist_expected2) /
                       sizeof(*fieldlist_expected2))
      data_ok = MTK_FALSE;

    for (i = 0; i < nfields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected2[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nfields, &fieldlist);
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
  strcpy(blockmetaname, "PerBlockMetadataGeoParm");
  
  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_SUCCESS)
  {
    if (nfields != sizeof(fieldlist_expected3) /
                       sizeof(*fieldlist_expected3))
      data_ok = MTK_FALSE;

    for (i = 0; i < nfields; ++i)
      if (strcmp(fieldlist[i],fieldlist_expected3[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
    MtkStringListFree(nfields, &fieldlist);
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

  /* Failure test calls */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
  
  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_HDF_OPEN_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  strcpy(blockmetaname, "abcd");
  
  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_HDF_VSFIND_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileBlockMetaFieldList(NULL,blockmetaname,&nfields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldList(filename,NULL,&nfields,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldList(filename,blockmetaname,NULL,&fieldlist);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldList(filename,blockmetaname,&nfields,NULL);
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
