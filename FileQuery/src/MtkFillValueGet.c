/*===========================================================================
=                                                                           =
=                             MtkFillValueGet                               =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdlib.h>
#include <mfhdf.h>
#include <HdfEosDef.h>

/** \brief Get fill value
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the fill value from the <tt> Blue Radiance/RDQI </tt> field in the \c BlueBand grid in the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFillValueGet("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand",
 *                "Blue Radiance/RDQI", &fillbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by \c fillbuf
 */

MTKt_status MtkFillValueGet(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *fillbuf /**< [OUT] Fill value */ )
{
  MTKt_status status;           /* Return status */

  status = MtkFillValueGetNC(filename, gridname, fieldname, fillbuf);  // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFillValueGetHDF(filename, gridname, fieldname, fillbuf);  // try HDF
}

MTKt_status MtkFillValueGetNC(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *fillbuf /**< [OUT] Fill value */ )
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

  /* Get fill value */
  status = MtkFillValueGetNcid(ncid, gridname, fieldname, fillbuf);
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

MTKt_status MtkFillValueGetHDF(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *fillbuf /**< [OUT] Fill value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 fid = FAIL;		/* HDF-EOS File id */

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Get fill value */
  status = MtkFillValueGetFid(fid, gridname, fieldname, fillbuf);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkFillValueGet that takes an HDF-EOS file ID rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFillValueGetFid(
  int32 fid,            /**< [IN] HDF-EOS File id */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *fillbuf /**< [OUT] Fill value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 gid = FAIL;		/* HDF-EOS Grid id */
  int32 hdf_datatype;           /* HDF-EOS data type */
  int32 rank;                   /* Not used */
  int32 dims[10];               /* Not used */
  char dimlist[80];             /* Not used */
  MTKt_DataBuffer fillbuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp fill buffer */
  MTKt_DataType datatype;       /* Mtk data type */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;		/* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  if (gridname == NULL ||
      fieldname == NULL || fillbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  gid = GDattach(fid, (char*)gridname);
  if (gid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = GDfieldinfo(gid, basefield, &rank, dims, &hdf_datatype, dimlist);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDFIELDINFO_FAILED);

  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(1, 1, datatype, &fillbuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  hdfstatus = GDgetfillvalue(gid, basefield, fillbuf_tmp.dataptr);
  if (hdfstatus == FAIL) {
	  if (datatype == MTKe_uint8) {
		  fillbuf_tmp.data.u8[0][0] = 253;
	  } else {
		  MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDGETFILLVALUE_FAILED);
	  }
  }

  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  *fillbuf = fillbuf_tmp;

  free(basefield);
  free(extradims);
  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (gid != FAIL) GDdetach(gid);
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&fillbuf_tmp);
  return status_code;
}

MTKt_status MtkFillValueGetNcid(
  int ncid,            /**< [IN] netCDF id */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *fillbuf /**< [OUT] Fill value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  MTKt_DataBuffer fillbuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp fill buffer */
  MTKt_DataType datatype;       /* Mtk data type */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;		/* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  if (gridname == NULL ||
      fieldname == NULL || fillbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  int group_id;
  {
    int nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  MTKt_ncvarid var;
  status = MtkNCVarId(group_id, basefield, &var);
  MTK_ERR_COND_JUMP(status);

  nc_type nc_datatype;
  {
    int nc_status = nc_inq_vartype(var.gid, var.varid, &nc_datatype);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(1, 1, datatype, &fillbuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  {
    int nc_status = nc_get_att(var.gid, var.varid, "_FillValue", fillbuf_tmp.dataptr);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *fillbuf = fillbuf_tmp;

  free(basefield);
  free(extradims);
  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&fillbuf_tmp);
  return status_code;
}
