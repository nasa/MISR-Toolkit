/*===========================================================================
=                                                                           =
=                          MtkFindFileList_test                             =
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

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char **filenames;
  int file_count;

  MTK_PRINT_STATUS(cn,"Testing MtkFindFileList");

  /* Normal test call */
  status = MtkFindFileList("../Mtk_testdata","GRP.*","DF",".*",
                           "012115","F03_0021",&file_count,&filenames);
  if (status == MTK_SUCCESS)
  {
    MtkStringListFree(file_count, &filenames);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","TC.*",NULL,".*",
                           ".*","F04_0007",&file_count,&filenames);
  if (status == MTK_SUCCESS)
  {
    MtkStringListFree(file_count, &filenames);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","TC.*","",".*",
                           ".*","F04_0007",&file_count,&filenames);
  if (status == MTK_SUCCESS)
  {
    MtkStringListFree(file_count, &filenames);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test - File instead of directory */
  status = MtkFindFileList("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf",
                           "GRP.*","DF",".*","012115","F03_0021",&file_count,&filenames);
  if (status == MTK_FAILURE)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  

  /* Failure test - Direcotry does not exist */
  status = MtkFindFileList("../Mtk_testdata/doesnotexist",
                           "GRP.*","DF",".*","012115","F03_0021",&file_count,&filenames);
  if (status == MTK_FAILURE)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }  
  
  /* Argument Checks */
  status = MtkFindFileList(NULL,"GRP_TERRAIN_GM","DF",".*",
                           "012115","F03_0021",&file_count,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata",NULL,"DF",".*",
                           "012115","F03_0021",&file_count,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","GRP_TERRAIN_GM","DF",NULL,
                           "012115","F03_0021",&file_count,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","GRP_TERRAIN_GM","DF",".*",
                           NULL,"F03_0021",&file_count,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","GRP_TERRAIN_GM","DF",".*",
                           "012115",NULL,&file_count,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","GRP_TERRAIN_GM","DF",".*",
                           "012115","F03_0021",NULL,&filenames);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFindFileList("../Mtk_testdata","GRP_TERRAIN_GM","DF",".*",
                           "012115","F03_0021",&file_count,NULL);
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
