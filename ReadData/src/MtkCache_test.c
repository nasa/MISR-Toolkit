/*===========================================================================
=                                                                           =
=                              MtkCache_test                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCache.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf.h>
#include <HdfEosDef.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Cache cache;		/* Cache structure */
  MTKt_uint16 buf;		/* Buffer */
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int cn = 0;			/* Column number */
  int32 fid = FAIL;		/* HDF-EOS File id */
  intn hdfstatus;		/* HDF return status */

  MTK_PRINT_STATUS(cn,"Testing MtkCache");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  /* Open file. */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkCacheInitFid(fid, gridname, fieldname, &cache);
  if (status != MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (fid == cache.fid) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (strcmp(gridname, cache.gridname) == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (strcmp(fieldname, cache.fieldname) == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkCachePixelGet(&cache, -1, 10, 20, (void *)&buf);
  MtkCachePixelGet(&cache, 1, -1, 20, (void *)&buf);
  MtkCachePixelGet(&cache, 1, 10, -1, (void *)&buf);

  {
    int b,l,s;

    for (b = 0; b < NBLOCK; b++) {
      for (l = 100; l < 150; l++) {
	for (s = 100; s < 150; s++) {
	  MtkCachePixelGet(&cache, b, l, s, (void *)&buf);
	}
      }
      /*     
      int j;
      printf("%d",i);
      for(j=0; j < NBLOCK; j++) {
	if (cache.block[j].valid) printf("*");
	else printf(".");
      }
      printf("\n");
      */
    }
  }

  MtkCacheFree(&cache);

  /* Close file. */
  hdfstatus = GDclose(fid);
  if (hdfstatus == -1) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Directional_Hemispherical_Reflectance[2]");

  /* Open file. */
  int ncid;
  int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
  if (nc_status != NC_NOERR) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (fid == FAIL) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  } else {
    MTK_PRINT_STATUS(cn,".");
  }

  status = MtkCacheInitNcid(ncid, gridname, fieldname, &cache);
  if (status != MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (ncid == cache.ncid) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (FAIL == cache.fid) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (strcmp(gridname, cache.gridname) == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (strcmp(fieldname, cache.fieldname) == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkCachePixelGet(&cache, -1, 10, 20, (void *)&buf);
  MtkCachePixelGet(&cache, 1, -1, 20, (void *)&buf);
  MtkCachePixelGet(&cache, 1, 10, -1, (void *)&buf);

  {
    for (int block = 0; block < NBLOCK; block++) {
      for (int i = 100; i < 150; i++) {
        for (int j = 100; j < 150; j++) {
          MtkCachePixelGet(&cache, block, i, j, (void *)&buf);
        }
      }
    }
  }

  MtkCacheFree(&cache);

  /* Close file. */
  {
    int nc_status = nc_close(ncid);
    if (nc_status != NC_NOERR) {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
