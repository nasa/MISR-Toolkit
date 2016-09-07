/*===========================================================================
=                                                                           =
=                             MtkPixelTime_test                             =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrOrbitPath.h"
#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  char pixel_time[MTKd_DATETIME_LEN];
  MTKt_TimeMetaData time_metadata = MTKT_TIME_METADATA_INIT;

  MTK_PRINT_STATUS(cn,"Testing MtkPixelTime");

  /* Normal call */
  status = MtkTimeMetaRead("../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf",&time_metadata);
  if (status != MTK_SUCCESS)
    pass = MTK_FALSE;
  
  /* path res blk line samp
      37  275  20  64  512 */
  status = MtkPixelTime(time_metadata, 10153687.5, 738787.5, pixel_time);
  if (status == MTK_SUCCESS && strcmp(pixel_time,"2005-06-04T18:06:07.656501Z") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* path res blk line samp
      37  275  20  64  1536 */
  status = MtkPixelTime(time_metadata, 10153687.5, 1020387.5, pixel_time);
  if (status == MTK_SUCCESS && strcmp(pixel_time,"2005-06-04T18:06:07.522389Z") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* path res blk line samp
      37  275  20  320 512 */
  status = MtkPixelTime(time_metadata, 10224087.5, 738787.5, pixel_time);
  if (status == MTK_SUCCESS && strcmp(pixel_time,"2005-06-04T18:06:18.002869Z") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* path res blk line samp
      37  275  20  320 1536 */
  status = MtkPixelTime(time_metadata, 10224087.5, 1020387.5, pixel_time);
  if (status == MTK_SUCCESS && strcmp(pixel_time,"2005-06-04T18:06:17.839556Z") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkPixelTime(time_metadata, -1, 1020387.5, pixel_time);
  if (status == MTK_MISR_FORWARD_PROJ_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkPixelTime(time_metadata, 10224087.5, -1, pixel_time);
  if (status == MTK_MISR_FORWARD_PROJ_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkPixelTime(time_metadata, 10224087.5, 1020387.5, NULL);
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
