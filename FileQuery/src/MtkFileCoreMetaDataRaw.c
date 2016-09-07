/*===========================================================================
=                                                                           =
=                          MtkFileCoreMetaDataRaw                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <hdf.h>

/** \brief Read core metadata from a MISR product file into a buffer
 *
 *  \return MTK_SUCCESS if successful.

 *  \par Example:
 *  In this example, we read the core metadata from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileCoreMetaDataRaw("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &coremeta);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using free() to free the memory used by \c coremeta  
 */

MTKt_status MtkFileCoreMetaDataRaw(
  const char *filename,    /**< [IN] File name */
  char **coremeta          /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdf_status;
  int32 sds_id = FAIL;

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_status = sds_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read coremetadata. */
  status = MtkFileCoreMetaDataRawFid(sds_id, coremeta);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF File */
  hdf_status = SDend(sds_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sds_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sds_id != FAIL)
    SDend(sds_id);

  return status_code;
}

/** \brief Version of MtkFileCoreMetaDataRaw that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileCoreMetaDataRawFid(
  int32 sds_id,            /**< [IN] HDF SD file identifier */
  char **coremeta          /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  intn hdf_status;
  int32 attr_index;
  char attr_name[80];
  int32 data_type;
  int32 count;
  char *attr_buf = NULL;

  if (coremeta == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Find attribute index */
  hdf_status = attr_index = SDfindattr(sds_id, "coremetadata");
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFINDATTR_FAILED);

  /* Get attribute information */
  hdf_status = SDattrinfo(sds_id, attr_index, attr_name, &data_type, &count);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDATTRINFO_FAILED);

  /* Allocate Memory */
  attr_buf = (char*)malloc((count + 1) * sizeof(char));
  if (attr_buf == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Read attribute */
  hdf_status = SDreadattr(sds_id, attr_index, (VOIDP)attr_buf);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDREADATTR_FAILED);

  attr_buf[count] = '\0';

  *coremeta = attr_buf;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (attr_buf != NULL)
    free(attr_buf);

  return status_code;
}
