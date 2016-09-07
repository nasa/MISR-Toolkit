/*===========================================================================
=                                                                           =
=                              MtkFileToPath                                =
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
#include <hdf.h>
#include <mfhdf.h>

/** \brief Read path number from file
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the path number from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileToPath("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &path);
 *  \endcode
 */

MTKt_status MtkFileToPath( const char *filename, /**< [IN] File name */
			   int *path             /**< [OUT] Path number */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int32 sid = FAIL;		/* HDF SD file identifier */
  MTKt_status status;      	/* Return status */
  intn hdfstatus;		/* HDF return status */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  sid = SDstart(filename, DFACC_READ);
  if (sid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read path number. */
  status = MtkFileToPathFid(sid, path);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = SDend(sid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sid != FAIL) {
    hdfstatus = SDend(sid);
  }
  return status_code;
}

/** \brief Version of MtkFileToPath that takes an HDF SD identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileToPathFid( int32 sid,            /**< [IN] HDF SD file identifier */
			      int *path             /**< [OUT] Path number */ )
{
  MTKt_status status_code;      /* Return status of this function */
  intn status;			/* HDF return status */
  int32 attr_index;		/* HDF attribute index */
  int32 path_tmp;		/* Temp path number */

  if (path == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  attr_index = SDfindattr(sid, "Path_number");
  if (attr_index == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFINDATTR_FAILED);

  status = SDreadattr(sid, attr_index, &path_tmp);
  if (status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDREADATTR_FAILED);

  *path = path_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
