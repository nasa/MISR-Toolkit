/*===========================================================================
=                                                                           =
=                             MtkFileVersion                                =
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
#include <string.h>
#include <mfhdf.h>

/** \brief Determine MISR product file version
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the file version from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileVersion("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", fileversion);
 *  \endcode
 */

MTKt_status MtkFileVersion(
  const char *filename, /**< [IN] File name */
  char *fileversion /**< [OUT] File version */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileVersionNC(filename, fileversion); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileVersionHDF(filename, fileversion); // try HDF
}

MTKt_status MtkFileVersionNC(
  const char *filename, /**< [IN] File name */
  char *fileversion /**< [OUT] File version */ )
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
  status = MtkFileVersionNcid(ncid,fileversion);
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

MTKt_status MtkFileVersionHDF(
  const char *filename, /**< [IN] File name */
  char *fileversion /**< [OUT] File version */ )
{
  MTKt_status status_code;
  MTKt_status status;
  int32 hdf_status;        /* HDF-EOS return status */
  int32 sd_id = FAIL;      /* HDF SD file identifier. */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_status = sd_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read MISR product file version */
  status = MtkFileVersionFid(sd_id,fileversion);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF File */
  hdf_status = SDend(sd_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sd_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sd_id != FAIL)
    SDend(sd_id);

  return status_code;
}

/** \brief Version of MtkFileVersion that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileVersionFid(
  int32 sd_id,      /**< [IN] HDF SD file identifier */
  char *fileversion /**< [OUT] File version */ )
{
  MTKt_status status_code;
  MTKt_status status;
  char *lgid;
  char *fn_end;
  char *fn_start;

  if (fileversion == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Get Local Granual ID */
  status = MtkFileLGIDFid(sd_id,&lgid);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  fn_end = strchr(lgid,'b'); /* check if file is a subset */
  if (fn_end != NULL)
    --fn_end; /* remove . before b */
  else /* Remove .hdf from end */
    fn_end = strstr(lgid, ".hdf");

  fn_start = fn_end;
  while (*fn_start != 'F')
    --fn_start;

  
  strncpy(fileversion,fn_start,fn_end - fn_start);
  fileversion[fn_end - fn_start] = '\0';

  free(lgid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}

MTKt_status MtkFileVersionNcid(
  int ncid,               /**< [IN] netCDF File ID */
  char *fileversion /**< [OUT] File version */ )
{
  MTKt_status status_code;
  MTKt_status status;
  char *lgid;
  char *fn_end;
  char *fn_start;

  if (fileversion == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Get Local Granual ID */
  status = MtkFileLGIDNcid(ncid,&lgid);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  fn_end = strstr(lgid,".b"); /* check if file is a subset */
  if (fn_end == NULL)
    fn_end = strstr(lgid, ".nc");

  fn_start = fn_end;
  while (*fn_start != 'F')
    --fn_start;

  strncpy(fileversion,fn_start,fn_end - fn_start);
  fileversion[fn_end - fn_start] = '\0';

  free(lgid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
