/*===========================================================================
=                                                                           =
=                              MtkReadBlock                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReadData.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdlib.h>
#include <mfhdf.h>
#include <HdfEosDef.h>

/** \brief Read block of data
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read block 26 for the field \c AveSceneElev in the grid \c Standard in the file \c MISR_AM1_AGP_P037_F01_24.hdf 
 *
 *  \code
 *  MTKt_DataBuffer databuf = MTK_DATABUFFER_INIT;
 *  status = MtkReadBlock("MISR_AM1_AGP_P037_F01_24.hdf", "Standard", "AveSceneElev", 26, &databuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by databuf
 *
 *  \note
 *  The MISR Toolkit read functions always return a 2-D data plane buffer.  Some fields in the MISR data products
 *  are multi-dimensional.  In order to read one of these fields, the slice to read needs to be specified.
 *  A bracket notation on the fieldname is used for this purpose.  For example fieldname = "RetrAppMask[0][5]".
 *
 *  \note
 *  Additional dimensions can be determined by the routine MtkFileGridFieldToDimlist() or by the
 *  MISR Data Product Specification (DPS) Document.  The actually definition of the indices are not described in the
 *  MISR product files and thus not described by the MISR Toolkit.  These will have to be looked up in the
 *  MISR DPS.  All indices are 0-based.
 */

MTKt_status MtkReadBlock(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *fieldname,   /**< [IN] Field name */
  int block,               /**< [IN] Block number */
  MTKt_DataBuffer *databuf /**< [OUT] Data buffer */  )
{
  MTKt_status status;       /* Return status */

  status = MtkReadBlockNC(filename, gridname, fieldname, block, databuf); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkReadBlockHDF(filename, gridname, fieldname, block, databuf); // try HDF
}

MTKt_status MtkReadBlockNC(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *fieldname,   /**< [IN] Field name */
  int block,               /**< [IN] Block number */
  MTKt_DataBuffer *databuf /**< [OUT] Data buffer */  )
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

  /* Read block. */
  status = MtkReadBlockNcid(ncid, gridname, fieldname, block, databuf);
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

MTKt_status MtkReadBlockHDF(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *fieldname,   /**< [IN] Field name */
  int block,               /**< [IN] Block number */
  MTKt_DataBuffer *databuf /**< [OUT] Data buffer */  )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 fid = FAIL;		/* HDF-EOS File id */

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file. */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read block. */
  status = MtkReadBlockFid(fid, gridname, fieldname, block, databuf);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkReadBlock that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadBlockFid(
  int32 fid,               /**< [IN] HDF-EOS file identifier */
  const char *gridname,    /**< [IN] Grid name */
  const char *fieldname,   /**< [IN] Field name */
  int block,               /**< [IN] Block number */
  MTKt_DataBuffer *databuf /**< [OUT] Data buffer */  )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 gid = FAIL;		/* HDF-EOS Grid id */
  int32 start[10];		/* HDF-EOS start dimension */
  int32 edge[10];		/* HDF-EOS edge dimension */
  int32 hdf_datatype;		/* HDF-EOS data type */
  int32 rank;			/* HDF-EOS rank */
  int32 dims[10];		/* HDF-EOS dimensions */
  char dimlist[80];		/* HDF-EOS dimension name list */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */
  MTKt_DataBuffer databuf_tmp = MTKT_DATABUFFER_INIT;
				/* Temp data buffer */
  MTKt_DataType datatype;	/* Mtk data type */
  int i;			/* Loop index */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (block < 1 || block > 180) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  gid = GDattach(fid, (char*)gridname);
  if (gid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  hdfstatus = GDfieldinfo(gid, basefield, &rank, dims, &hdf_datatype, dimlist);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDFIELDINFO_FAILED);

  if (rank != nextradims + 3) MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(dims[1], dims[2], datatype, &databuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  start[0] = block - 1;
  start[1] = 0;
  start[2] = 0;

  edge[0] = 1;
  edge[1] = dims[1];
  edge[2] = dims[2];

  for (i = 3; i < rank; i++) {
    start[i] = extradims[i-3];
    edge[i] = 1;
  }

  hdfstatus = GDreadfield(gid, basefield, start, NULL, edge,
			  databuf_tmp.dataptr);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDREADFIELD_FAILED);

  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  *databuf = databuf_tmp;

  free(basefield);
  free(extradims);
  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (gid != FAIL) GDdetach(gid);
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&databuf_tmp);
  return status_code;
}

MTKt_status MtkReadBlockNcid(
  int ncid,                /**< [IN] netCDF file identifier */
  const char *gridname,    /**< [IN] Grid name */
  const char *fieldname,   /**< [IN] Field name */
  int block,               /**< [IN] Block number */
  MTKt_DataBuffer *databuf /**< [OUT] Data buffer */  )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */
  MTKt_DataBuffer databuf_tmp = MTKT_DATABUFFER_INIT;
				/* Temp data buffer */
  MTKt_DataType datatype;	/* Mtk data type */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (block < 1 || block > 180) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  int group_id;
  {
    int nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  int size_x;
  int size_y;
  int start_x;
  int start_y;
  int found = 0;

  {
    int nc_status;
    nc_status = nc_get_att(group_id, NC_GLOBAL, "block_size_in_lines", &size_x);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_get_att(group_id, NC_GLOBAL, "block_size_in_samples", &size_y);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    int bvarid;
    nc_status = nc_inq_varid(group_id, "Block_Number", &bvarid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    int dimid[NC_MAX_DIMS];
    nc_status = nc_inq_var(group_id, bvarid, NULL, NULL, NULL, dimid, NULL);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    size_t number_block;
    nc_status = nc_inq_dimlen(group_id, dimid[0], &number_block);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    int block_numbers[180];
    nc_status = nc_get_var_int(group_id, bvarid, block_numbers);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    size_t block_index;
    for (size_t i = 0 ; i < number_block ; i++) {
      if (block_numbers[i] == block) {
        found = 1;
        block_index = i;
        break;
      }
    }

    if (found == 1) {

      int xvarid;
      nc_status = nc_inq_varid(group_id, "Block_Start_X_Index", &xvarid);
      if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

      int yvarid;
      nc_status = nc_inq_varid(group_id, "Block_Start_Y_Index", &yvarid);
      if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

      nc_status = nc_get_var1_int(group_id, xvarid, &block_index, &start_x);
      if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

      nc_status = nc_get_var1_int(group_id, yvarid, &block_index, &start_y);
      if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    }
  }

  if (found == 0) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);  // Block is outside file extent

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  int rank;
  nc_type nc_datatype;
  MTKt_ncvarid var;
  {
    int nc_status;

    status = MtkNCVarId(group_id, basefield, &var);
    MTK_ERR_COND_JUMP(status);

    nc_status = nc_inq_varndims(var.gid, var.varid, &rank);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_inq_vartype(var.gid, var.varid, &nc_datatype);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  if (rank != nextradims + 2) MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(size_x, size_y, datatype, &databuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  size_t start[10];
  size_t count[10];

  start[0] = start_x;
  start[1] = start_y;

  count[0] = size_x;
  count[1] = size_y;

  for (int i = 2; i < rank; i++) {
    start[i] = extradims[i-2];
    count[i] = 1;
  }

  {
    int nc_status = nc_get_vara(var.gid, var.varid, start, count, databuf_tmp.dataptr);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *databuf = databuf_tmp;

  free(basefield);
  free(extradims);
  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&databuf_tmp);
  return status_code;
}
