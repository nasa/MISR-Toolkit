/*===========================================================================
=                                                                           =
=                               MtkFileLGID                                 =
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

/** \brief Determine local granual ID of MISR product file
 *
 *  \return MTK_SUCCESS if successful.

 *  \par Example:
 *  In this example, we read the local granual ID from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileLGID("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &lgid);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using free() to free the memory used by \c lgid
 */

MTKt_status MtkFileLGID(
  const char *filename,    /**< [IN] File name */
  char **lgid              /**< [OUT] Local Granual ID */ )
{
  MTKt_status status;       /* Return status */

  status = MtkFileLGIDNC(filename, lgid); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileLGIDHDF(filename, lgid); // try HDF
}

MTKt_status MtkFileLGIDNC(
  const char *filename,    /**< [IN] File name */
  char **lgid              /**< [OUT] Local Granual ID */ )
{
  MTKt_status status;
  MTKt_status status_code;
  int ncid = 0;

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file */
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Get Local Granual ID */
  status = MtkFileLGIDNcid(ncid,lgid);
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

MTKt_status MtkFileLGIDHDF(
  const char *filename,    /**< [IN] File name */
  char **lgid              /**< [OUT] Local Granual ID */ )
{
  MTKt_status status;
  MTKt_status status_code; /* Return status of this function */
  intn hdf_status;
  int32 sds_id = FAIL;

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_status = sds_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Get Local Granual ID */
  status = MtkFileLGIDFid(sds_id,lgid);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  /* Close HDF File */
  hdf_status = SDend(sds_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sds_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sds_id != FAIL) {
    hdf_status = SDend(sds_id);
  }
  return status_code;
}

/* ----------------------------------------------- */
/* MtkFileLGIDFid                                  */
/* ----------------------------------------------- */

/** \brief Version of MtkFileLGID that takes an HDF SDS ID rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileLGIDFid(
  int32 sds_id,            /**< [IN] HDF SDS ID */
  char **lgid              /**< [OUT] Local Granual ID */ )
{
  MTKt_status status_code; /* Return status of this function */
  intn hdf_status;
  int32 attr_index;
  char attr_name[80];
  int32 data_type;
  int32 count;
  char *attr_buf = NULL;
  char *fn_start;
  char *fn_end;

  if (lgid == NULL)
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

  fn_start = strstr(attr_buf, "MISR_AM1_");
  if (fn_start == NULL) {
	  fn_start = strstr(attr_buf, "MISR_HR_");
	  if (fn_start == NULL)
	    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }
  fn_end = strstr(attr_buf, ".hdf");
  if (fn_end == NULL)
    MTK_ERR_CODE_JUMP(MTK_FAILURE);

  fn_end += 4; /* Move to end of file name */

  *lgid = (char*)malloc((fn_end - fn_start + 1) * sizeof(char));
  strncpy(*lgid,fn_start,fn_end - fn_start);
  (*lgid)[fn_end - fn_start] = '\0';

  free(attr_buf);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (attr_buf != NULL)
    free(attr_buf);

  return status_code;
}

MTKt_status MtkFileLGIDNcid(
  int ncid,            /**< [IN] netCDF file identifier */
  char **lgid              /**< [OUT] Local Granual ID */ )
{
  MTKt_status status_code; /* Return status of this function */
  char *lgid_tmp = NULL;

  if (lgid == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  size_t len;
  {
    int nc_status = nc_inq_attlen(ncid, NC_GLOBAL, "Local_granule_id", &len);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  lgid_tmp = calloc(len+1, sizeof(char));  /* Add 1 for null terminator */

  {
    int nc_status = nc_get_att(ncid, NC_GLOBAL, "Local_granule_id", lgid_tmp);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *lgid = lgid_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (lgid_tmp != NULL)
    free(lgid_tmp);

  return status_code;
}
