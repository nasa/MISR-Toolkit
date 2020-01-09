/*===========================================================================
=                                                                           =
=                           MtkFileToBlockRange                             =
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

/** \brief Read start and end block numbers from file
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the start and end block numbers from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileToBlockRange("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &start_block, &end_block);
 *  \endcode
 */

MTKt_status MtkFileToBlockRange(
  const char *filename, /**< [IN] File name */
  int *start_block,     /**< [OUT] Start block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileToBlockRangeNC(filename, start_block, end_block); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileToBlockRangeHDF(filename, start_block, end_block); // try HDF
}

MTKt_status MtkFileToBlockRangeNC(
  const char *filename, /**< [IN] File name */
  int *start_block,     /**< [OUT] Start block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file */
  int ncid = 0;
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read grid attribute */
  status = MtkFileToBlockRangeNcid(ncid, start_block, end_block);
  MTK_ERR_COND_JUMP(status);

  /* Close file */
  {
    int nc_status = nc_close(ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_CLOSE_FAILED);
  }
  ncid = 0;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (ncid != 0) nc_close(ncid);
  return status_code;
}

MTKt_status MtkFileToBlockRangeHDF(
  const char *filename, /**< [IN] File name */
  int *start_block,     /**< [OUT] Start block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int32 sid=FAIL;  	        /* HDF SD file identifier */
  intn hdfstatus;		/* HDF return status */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file. */
  sid = SDstart(filename, DFACC_READ);
  if (sid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read block range. */
  status = MtkFileToBlockRangeFid(sid, start_block, end_block);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = SDend(sid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sid != FAIL) SDend(sid);
  return status_code;
}

/** \brief Version of MtkFileToBlockRange that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileToBlockRangeFid(
  int32 sid,            /**< [IN] HDF SD file identifier */
  int *start_block,     /**< [OUT] Start block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status_code;      /* Return status of this function */
  intn status;			/* HDF return status */
  int32 attr_index;		/* HDF attribute index */
  int32 start_block_tmp;	/* Temp start block */
  int32 end_block_tmp;		/* Temp end block */

  if (start_block == NULL || end_block == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  attr_index = SDfindattr(sid, "Start_block");
  if (attr_index == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFINDATTR_FAILED);

  status = SDreadattr(sid, attr_index, &start_block_tmp);
  if (status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDREADATTR_FAILED);

  attr_index = SDfindattr(sid, "End block");
  if (attr_index == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFINDATTR_FAILED);

  status = SDreadattr(sid, attr_index, &end_block_tmp);
  if (status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDREADATTR_FAILED);

  *start_block = start_block_tmp;
  *end_block = end_block_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}

MTKt_status MtkFileToBlockRangeNcid(
  int ncid,               /**< [IN] netCDF File ID */
  int *start_block,     /**< [OUT] Start block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int start_block_tmp;	/* Temp start block */
  int end_block_tmp;		/* Temp end block */

  if (start_block == NULL || end_block == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  {
    int nc_status = nc_get_att_int(ncid, NC_GLOBAL, "Start_block", &start_block_tmp);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }
  {
    int nc_status = nc_get_att_int(ncid, NC_GLOBAL, "End_block", &end_block_tmp);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *start_block = start_block_tmp;
  *end_block = end_block_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
