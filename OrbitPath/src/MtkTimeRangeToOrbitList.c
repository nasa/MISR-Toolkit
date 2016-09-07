/*===========================================================================
=                                                                           =
=                         MtkTimeRangeToOrbitList                           =
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
#include <stdlib.h>

/** \brief Given start time and end time return list of orbits
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  Time Format: YYYY-MM-DDThh:mm:ssZ (ISO 8601)
 *
 *  \par Example:
 *  In this example, we get the list of orbits from 2002-02-02 02:00:00 UTC to 2002-05-02 02:00:00 UTC
 *
 *  \code
 *  status = MtkTimeRangeToOrbitList("2002-02-02T02:00:00Z", "2002-05-02T02:00:00Z", &orbitcnt, &orbitlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for freeing the memory used by \c orbitlist
 */

MTKt_status MtkTimeRangeToOrbitList(
  const char *start_time, /**< [IN] Start Time */
  const char *end_time, /**< [IN] End Time */
  int *orbitcnt, /**< [OUT] Orbit Count */
  int **orbitlist /**< [OUT] Orbit List */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int start_orbit;
  int start_path;
  int end_orbit;
  int end_path;
  int *list = NULL;
  int i;

  if (start_time == NULL || end_time == NULL ||
      orbitcnt == NULL || orbitlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Find start orbit */
  status = MtkTimeToOrbitPath(start_time,&start_orbit,&start_path);
  if (status == MTK_BAD_ARGUMENT)
  {
    status = MtkTimeToOrbitPath("2000-02-25T01:00:00Z",&start_orbit,&start_path);
    MTK_ERR_COND_JUMP(status);
  }
  else if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status)  

  /* Find end orbit */
  status = MtkTimeToOrbitPath(end_time,&end_orbit,&end_path);
  MTK_ERR_COND_JUMP(status);

  if (start_orbit > end_orbit)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  *orbitcnt = end_orbit - start_orbit + 1;

  list = (int*)malloc(*orbitcnt * sizeof(int));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (i = 0; i < *orbitcnt; ++i)
    list[i] = start_orbit + i;

  *orbitlist = list;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  return status_code;
}
