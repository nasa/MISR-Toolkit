/*===========================================================================
=                                                                           =
=                         MtkFileGridToResolution                           =
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
#include <mfhdf.h>
#include <HdfEosDef.h>

/** \brief Get resolution of a particular grid
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the resolution of the grid \c BlueBand from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileGridToResolution("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand", &resolution);
 *  \endcode
 */

MTKt_status MtkFileGridToResolution(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  int *resolution  /**< [OUT] Resolution */ )
{
  MTKt_status status;      	/* Return status */

  status = MtkFileGridToResolutionNC(filename, gridname, resolution); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileGridToResolutionHDF(filename, gridname, resolution); // try HDF
}

MTKt_status MtkFileGridToResolutionNC(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  int *resolution  /**< [OUT] Resolution */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file */
  int ncid = 0;
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Get resolution of grid. */
  status = MtkFileGridToResolutionNcid(ncid, gridname, resolution);
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

MTKt_status MtkFileGridToResolutionHDF(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  int *resolution  /**< [OUT] Resolution */ )
{
  MTKt_status status_code;	/* Return code of this function */
  MTKt_status status;      	/* Return status */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 fid = FAIL;		/* HDF-EOS file identifier */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fid = GDopen((char *)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Get resolution of grid. */
  status = MtkFileGridToResolutionFid(fid, gridname, resolution);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkFileGridToResolution that takes an HDF-EOS file id rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridToResolutionFid(
  int32 fid,            /**< [IN] HDF-EOS file identifier */
  const char *gridname, /**< [IN] Grid name */
  int *resolution  /**< [OUT] Resolution */ )
{
  MTKt_status status_code;	/* Return code of this function */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 gid = FAIL;		/* HDF-EOS grid identifier */
  int32 res_x;			/* Resolution in Som X */
  int32 res_y;			/* Resolution in Som Y */

  if (gridname == NULL || resolution == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  gid = GDattach(fid, (char *)gridname);
  if (gid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  hdfstatus = GDreadattr(gid, "Block_size.resolution_x", (void *)&res_x);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDREADATTR_FAILED);

  hdfstatus = GDreadattr(gid, "Block_size.resolution_y", (void *)&res_y);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDREADATTR_FAILED);

  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  if (res_x != res_y) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  *resolution = res_x;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (gid != FAIL) GDdetach(gid);
  return status_code;
}

MTKt_status MtkFileGridToResolutionNcid(
  int ncid,            /**< [IN] netCDF file identifier */
  const char *gridname, /**< [IN] Grid name */
  int *resolution  /**< [OUT] Resolution */ )
{
  MTKt_status status_code;	/* Return code of this function */

  if (gridname == NULL || resolution == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  int group_id;
  {
    int nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  int32 res;
  {
    int nc_status = nc_get_att(group_id, NC_GLOBAL, "resolution_in_meters", &res);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *resolution = res;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  return status_code;
}
