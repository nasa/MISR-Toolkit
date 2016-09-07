/*===========================================================================
=                                                                           =
=                           MtkLatLonToPathList                             =
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

/** \brief Get list of paths that cover a particular latitude and longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the list of paths covering decimal degrees latitude -75.345 and longitude 169.89.
 *
 *  \code
 *  status = MtkLatLonToPathList(-75.345, 169.89, &pathcnt, &pathlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for freeing the memory used by \c pathlist
 */

MTKt_status MtkLatLonToPathList(
  double lat_dd, /**< [IN] Latitude */
  double lon_dd, /**< [IN] Longitude */
  int *pathcnt,  /**< [OUT] Path Count */
  int **pathlist /**< [OUT] Path List */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  int block;			/* Block */
  float line;			/* Line */
  float sample;			/* Sample */
  int path;			/* Loop index */
  int pathlist_tmp[NPATH];	/* Temp pathlist */
  int *list = NULL;		/* Temp list */
  int cnt = 0;			/* Path count */

  if (pathcnt == NULL || pathlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  for (path = 1; path <= NPATH; path++) {
    status = MtkLatLonToBls(path, MAXRESOLUTION, lat_dd, lon_dd, 
		   &block, &line, &sample);
    
    if (status == MTK_SUCCESS)
      pathlist_tmp[cnt++] = path;
  }

  if (cnt == 0)
    MTK_ERR_CODE_JUMP(MTK_NOT_FOUND);

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
