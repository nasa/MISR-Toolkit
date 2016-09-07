/*===========================================================================
=                                                                           =
=                           MtkRegionToPathList                             =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrCoordQuery.h"
#include "MisrError.h"
#include "MisrProjParam.h"
#include <stdlib.h>

/** \brief Get list of paths that cover a particular region
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the list of paths covering a particular region of interest.
 *
 *  \code
 *  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for freeing the memory used by \c pathlist
 */

MTKt_status MtkRegionToPathList(
  MTKt_Region region,   /**< [IN] Region */
  int *pathcnt,         /**< [OUT] Path Count */
  int **pathlist        /**< [OUT] Path List */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  int b;			/* Block */
  float l;			/* Line */
  float s;			/* Sample */
  int path;			/* Loop index */
  int pathlist_tmp[NPATH];	/* Temp pathlist */
  int *list = NULL;		/* Temp list */
  int cnt = 0;			/* Path count */
  int i;			/* Corner index */
  float lines[4];		/* Block corner lines */
  float samples[4];		/* Block corner samples */
  MTKt_boolean found = MTK_FALSE; /* Found a corner in the region flag */
  double lat_hextent_deg; /* Lat extent in degrees */
  double lon_hextent_deg; /* Lon exten in degrees */
  double ulclat; /* Lat of ULC of Region */
  double ulclon; /* Lon of ULC of Region */
  double lrclat; /* Lat of LRC of Region */
  double lrclon; /* Lon of LRC of Region */
  double blocklat = 0.0; /* Lat of block */
  double blocklon = 0.0; /* Lon of block */
  double deg_per_meter = 360.0 / 40007821.0; /* Average earth circumference (40075004m + 39940638m) / 2.0 = 40007821m */

  if (pathcnt == NULL || pathlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* -------------------------------------------- */
  /* Set lines and samples for four block corners */
  /* -------------------------------------------- */

  lines[0] = 0.0;
  samples[0] = 0.0;
  lines[1] = 0.0;
  samples[1] = MAXNSAMPLE - 1;
  lines[2] = MAXNLINE - 1 ;
  samples[2] = MAXNSAMPLE - 1;
  lines[3] = MAXNLINE - 1;
  samples[3] = 0.0;

  for (path = 1; path <= NPATH; path++) {

    /* ------------------------------------------------------------------- */
    /* If center of region is in a block then we have found a path - done. */
    /* ------------------------------------------------------------------- */

    status = MtkLatLonToBls(path, MAXRESOLUTION, region.geo.ctr.lat,
                            region.geo.ctr.lon, &b, &l, &s);

    if (status == MTK_SUCCESS) { /* --- Center coordinate is in a block in this path --- */

      found = MTK_TRUE;

    } else { /* --- Search block corners for a coordinate in this region --- */
      lat_hextent_deg = (region.hextent.xlat * deg_per_meter);
      lon_hextent_deg = (region.hextent.ylon * deg_per_meter);
      ulclat = region.geo.ctr.lat - lat_hextent_deg;
      ulclon = region.geo.ctr.lon - lon_hextent_deg;
      lrclat = region.geo.ctr.lat + lat_hextent_deg;
      lrclon = region.geo.ctr.lon + lon_hextent_deg;

      /* ------------------------------------------------------------------------ */
      /* Determine Lat/Lon coordinates of the corners of the region for this path */
      /* ------------------------------------------------------------------------ */
      status = MtkLatLonToBls(path, MAXRESOLUTION,
                              region.geo.ctr.lat - lat_hextent_deg,
                              region.geo.ctr.lon - lon_hextent_deg,
                              &b, &l, &s);
      
      /* --------------------------------------------------------------- */
      /* Loop over each block of the to find a corner inside the region  */
      /* --------------------------------------------------------------- */

      for (b = 1; b <= NBLOCK; b++) {
        for (i = 0; i < 4; i++) {          
          status = MtkBlsToLatLon(path, MAXRESOLUTION,b, lines[i], samples[i],
                                  &blocklat, &blocklon);
          if (blocklat >= ulclat && blocklat <= lrclat && blocklon >= ulclon && blocklon <= lrclon) {
            found = MTK_TRUE;
            // printf("Found block: %i in path: %i\n",b,path);
          }
        }
        if (found) break;
      }
    } /* --- End of block corner search --- */

    /* -------------------------------------------------- */
    /* Add path to pathlist only if it crosses the region */
    /* -------------------------------------------------- */

    if (found) {
      pathlist_tmp[cnt++] = path;
      found = MTK_FALSE;
    } // else { printf("No block found for path: %i\n", path); }
  } /* --- End of path loop --- */

  if (cnt == 0) MTK_ERR_CODE_JUMP(MTK_NOT_FOUND);

  list = (int *)malloc(cnt * sizeof(int));
  if (list == NULL) 
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (path = 0; path < cnt; path++) {
    list[path] = pathlist_tmp[path];
  }

  *pathcnt = cnt;
  *pathlist = list;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  return status_code;
}
