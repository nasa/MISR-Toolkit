/*===========================================================================
=                                                                           =
=                           MtkMakeFilename_test                            =
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char *filename;

  MTK_PRINT_STATUS(cn,"Testing MtkMakeFilename");

  /* Normal Call */
  status = MtkMakeFilename("MisrToolkit","GRP_TERRAIN_GM",
			   "DA",161,12115,"F03_0024",
			   &filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MisrToolkit/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_"
                      "DA_F03_0024.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit","GRP_terrain_GM",
			   "DA",161,12115,"F03_0024",
			   &filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MisrToolkit/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_"
                      "DA_F03_0024.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit","TC_ALBEDO",NULL,37,29058,
                           "F04_0007",&filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MisrToolkit/MISR_AM1_TC_ALBEDO_P037_O029058_"
                      "F04_0007.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit","TC_ALBEDO","",37,29058,
                           "F04_0007",&filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MisrToolkit/MISR_AM1_TC_ALBEDO_P037_O029058_"
                      "F04_0007.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit/","TC_ALBEDO","",37,29058,
                           "F04_0007",&filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MisrToolkit/MISR_AM1_TC_ALBEDO_P037_O029058_"
                      "F04_0007.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("","TC_ALBEDO","",37,29058,
                           "F04_0007",&filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MISR_AM1_TC_ALBEDO_P037_O029058_"
                      "F04_0007.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("","AGP","",37,0,"F01_24",&filename);
  if (status == MTK_SUCCESS &&
      strcmp(filename,"MISR_AM1_AGP_P037_F01_24.hdf") == 0)
  {
    free(filename);
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }



  /* Argument Checks */
  status = MtkMakeFilename(NULL,"GRP_TERRAIN_GM",
			   "DA",161,12115,"F03_0024",
			   &filename);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit",NULL,
			   "DA",161,12115,"F03_0024",
			   &filename);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit","GRP_TERRAIN_GM",
			   "DA",161,12115,NULL,
			   &filename);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkMakeFilename("MisrToolkit","GRP_TERRAIN_GM",
			   "DA",161,12115,"F03_0024",
			   NULL);
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
