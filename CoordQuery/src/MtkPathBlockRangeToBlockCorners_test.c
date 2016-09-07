/*===========================================================================
=                                                                           =
=                   MtkPathBlockRangeToBlockCorners_test                    =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrError.h"
#include <math.h>
#include <stdio.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean good = MTK_TRUE; /* Good data flag */
  int path;			/* Path */
  int start_block;		/* Start block */
  int end_block;		/* End block */
  int i;			/* Block index */
  double ulclat;		/* ULC latitude */
  double ulclon;		/* ULC longitude */
  double urclat;		/* URC latitude */
  double urclon;		/* URC longitude */
  double ctrlat;		/* CTR latitude */
  double ctrlon;		/* CTR longitude */
  double lrclat;		/* LRC latitude */
  double lrclon;		/* LRC longitude */
  double llclat;		/* LLC latitude */
  double llclon;		/* LLC longitude */
  MTKt_BlockCorners block_corners = MTKT_BLOCKCORNERS_INIT;
				/* Block corners structure */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkPathBlockRangeToBlockCorners");

  /* Normal call test case */
  path = 37;
  start_block = 25;
  end_block = 50;

  status = MtkPathBlockRangeToBlockCorners(path, start_block, end_block,
					   &block_corners);
  if (status == MTK_SUCCESS ||
      block_corners.path == path ||
      block_corners.start_block == start_block ||
      block_corners.end_block == end_block) {
    good = MTK_TRUE;
    for (i = start_block; i <= end_block; i++) {
      MtkBlsToLatLon(path, 275, i, 0.0, 0.0, &ulclat, &ulclon);
      MtkBlsToLatLon(path, 275, i, 0.0, 2047.0, &urclat, &urclon);
      MtkBlsToLatLon(path, 275, i, 255.5, 1023.5, &ctrlat, &ctrlon);
      MtkBlsToLatLon(path, 275, i, 511.0, 2047.0, &lrclat, &lrclon);
      MtkBlsToLatLon(path, 275, i, 511.0, 0.0, &llclat, &llclon);
      if (block_corners.block[i].block_number != i ||
	  fabs(block_corners.block[i].ulc.lat - ulclat) > .000001 ||
	  fabs(block_corners.block[i].ulc.lon - ulclon) > .000001 ||
	  fabs(block_corners.block[i].urc.lat - urclat) > .000001 ||
	  fabs(block_corners.block[i].urc.lon - urclon) > .000001 ||
	  fabs(block_corners.block[i].ctr.lat - ctrlat) > .000001 ||
	  fabs(block_corners.block[i].ctr.lon - ctrlon) > .000001 ||
	  fabs(block_corners.block[i].lrc.lat - lrclat) > .000001 ||
	  fabs(block_corners.block[i].lrc.lon - lrclon) > .000001 ||
	  fabs(block_corners.block[i].llc.lat - llclat) > .000001 ||
	  fabs(block_corners.block[i].llc.lon - llclon) > .000001 ) {
	good = MTK_FALSE;
      }
    }
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  path = 233;
  start_block = 1;
  end_block = 180;

  status = MtkPathBlockRangeToBlockCorners(path, start_block, end_block,
					   &block_corners);
  if (status == MTK_SUCCESS ||
      block_corners.path == path ||
      block_corners.start_block == start_block ||
      block_corners.end_block == end_block) {
    good = MTK_TRUE;
    for (i = start_block; i <= end_block; i++) {
      MtkBlsToLatLon(path, 275, i, 0.0, 0.0, &ulclat, &ulclon);
      MtkBlsToLatLon(path, 275, i, 0.0, 2047.0, &urclat, &urclon);
      MtkBlsToLatLon(path, 275, i, 255.5, 1023.5, &ctrlat, &ctrlon);
      MtkBlsToLatLon(path, 275, i, 511.0, 2047.0, &lrclat, &lrclon);
      MtkBlsToLatLon(path, 275, i, 511.0, 0.0, &llclat, &llclon);
      if (block_corners.block[i].block_number != i ||
	  fabs(block_corners.block[i].ulc.lat - ulclat) > .000001 ||
	  fabs(block_corners.block[i].ulc.lon - ulclon) > .000001 ||
	  fabs(block_corners.block[i].urc.lat - urclat) > .000001 ||
	  fabs(block_corners.block[i].urc.lon - urclon) > .000001 ||
	  fabs(block_corners.block[i].ctr.lat - ctrlat) > .000001 ||
	  fabs(block_corners.block[i].ctr.lon - ctrlon) > .000001 ||
	  fabs(block_corners.block[i].lrc.lat - lrclat) > .000001 ||
	  fabs(block_corners.block[i].lrc.lon - lrclon) > .000001 ||
	  fabs(block_corners.block[i].llc.lat - llclat) > .000001 ||
	  fabs(block_corners.block[i].llc.lon - llclon) > .000001 ) {
	good = MTK_FALSE;
      }
    }
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkPathBlockRangeToBlockCorners(path, start_block, end_block,
					   NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathBlockRangeToBlockCorners(-1, start_block, end_block,
					   &block_corners);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathBlockRangeToBlockCorners(path, 50, 25,
					   &block_corners);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathBlockRangeToBlockCorners(path, -1, end_block,
					   &block_corners);
  if (status == MTK_OUTBOUNDS) {
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
