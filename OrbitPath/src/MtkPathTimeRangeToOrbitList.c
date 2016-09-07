/*===========================================================================
=                                                                           =
=                       MtkPathTimeRangeToOrbitList                         =
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
#include "MisrError.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/** \brief Given path and time range return list of orbits on path
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  Time Format: YYYY-MM-DDThh:mm:ssZ (ISO 8601)
 *
 *  \par Example:
 *  In this example, we get the list of orbits covering path 78 from 2002-02-02 02:00:00 UTC to 2002-05-02 02:00:00 UTC
 *
 *  \code
 *  status = MtkPathTimeRangeToOrbitList(78, "2002-02-02T02:00:00Z", "2002-05-02T02:00:00Z", &orbitcnt, &orbitlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for freeing the memory used by \c orbitlist
 */

MTKt_status MtkPathTimeRangeToOrbitList(
  int path, /**< [IN] Path */
  const char *start_time, /**< [IN] Start Time */
  const char *end_time, /**< [IN] End Time */
  int *orbitcnt, /**< [OUT] Orbit Count */
  int **orbitlist /**< [OUT] Orbit List */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;		/* Return status */
  int start_orbit;
  int start_path;
  int end_orbit;
  int end_path;
  int i;
  int num_orbits;
  int num_paths = 0;
  int ret_path;
  int *olist = NULL;

  if (path < 1 || path > 233)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  if (start_time == NULL || end_time == NULL || orbitcnt == NULL ||
      orbitlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkTimeToOrbitPath(start_time,&start_orbit,&start_path);
  if (status == MTK_BAD_ARGUMENT)
  {
    status = MtkTimeToOrbitPath("2000-03-03T00:00:00Z",&start_orbit,&start_path);
    MTK_ERR_COND_JUMP(status);
  }
  else if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status)  

  status = MtkTimeToOrbitPath(end_time,&end_orbit,&end_path);
  MTK_ERR_COND_JUMP(status)

  if (start_orbit > end_orbit)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  num_orbits = end_orbit - start_orbit + 1;
  olist = (int*)malloc((size_t)ceil(num_orbits / 233.0) * sizeof(int));
  if (olist == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (i = 0; i < num_orbits; ++i)
  {
    status = MtkOrbitToPath(start_orbit + i,&ret_path);
    if (status != MTK_SUCCESS)
      MTK_ERR_CODE_JUMP(status);

    if (ret_path == path)
      olist[num_paths++] = start_orbit + i;
  }

  *orbitlist = (int*)malloc(num_paths * sizeof(int));
  if (*orbitlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  *orbitcnt = num_paths;

  memcpy(*orbitlist,olist,num_paths * sizeof(int));

  free(olist);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (olist != NULL)
    free(olist);

  return status_code;
}
