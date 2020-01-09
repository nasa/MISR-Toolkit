/*===========================================================================
=                                                                           =
=                              MtkGridAttrGet                               =
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
#include <mfhdf.h>
#include <HdfEosDef.h>

/** \brief Get a grid attribute
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the \c Block_size.resolution_x attribute from the \c BlueBand grid in the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkGridAttrGet("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand",
 *                "Block_size.resolution_x", &attrbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by \a attrbuf
 */

MTKt_status MtkGridAttrGet(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status;           /* Return status */

  status = MtkGridAttrGetNC(filename, gridname, attrname, attrbuf); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkGridAttrGetHDF(filename, gridname, attrname, attrbuf); // try HDF
}

MTKt_status MtkGridAttrGetNC(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
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

  /* Read grid attribute */
  status = MtkGridAttrGetNcid(ncid, gridname, attrname, attrbuf);
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

MTKt_status MtkGridAttrGetHDF(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 fid = FAIL;		/* HDF-EOS File id */

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read grid attribute */
  status = MtkGridAttrGetFid(fid, gridname, attrname, attrbuf);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);

  return MTK_SUCCESS;
ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkFileGridAttrGet that takes an HDF-EOS file ID rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkGridAttrGetFid(
  int32 fid,               /**< [IN] HDF-EOS File ID */
  const char *gridname,    /**< [IN] Grid name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 gid = FAIL;		/* HDF-EOS Grid id */
  int32 hdf_datatype;           /* HDF-EOS data type */
  int32 hdf_datasize;		/* HDF-EOS attribute datasize in bytes */
  MTKt_DataBuffer attrbuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp attribute buffer */
  MTKt_DataType datatype;       /* Mtk data type */

  if (gridname == NULL || 
      attrname == NULL || attrbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  gid = GDattach(fid, (char*)gridname);
  if (gid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  hdfstatus = GDattrinfo(gid, (char*)attrname, &hdf_datatype, &hdf_datasize);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTRINFO_FAILED);

  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(1, 1, datatype, &attrbuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  hdfstatus = GDreadattr(gid, (char*)attrname, attrbuf_tmp.dataptr);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDREADATTR_FAILED);

  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  *attrbuf = attrbuf_tmp;

  return MTK_SUCCESS;
ERROR_HANDLE:
  if (gid != FAIL) GDdetach(gid);
  MtkDataBufferFree(&attrbuf_tmp);
  return status_code;
}

MTKt_status MtkGridAttrGetNcid(
  int ncid,               /**< [IN] netCDF File ID */
  const char *gridname,    /**< [IN] Grid name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_status status;           /* Return status */
  MTKt_DataBuffer attrbuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp attribute buffer */
  MTKt_DataType datatype;       /* Mtk data type */

  if (gridname == NULL || 
      attrname == NULL || attrbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  int group_id;
  {
    int nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  nc_type nc_datatype;
  {
    int nc_status = nc_inq_atttype(group_id, NC_GLOBAL, attrname, &nc_datatype);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(1, 1, datatype, &attrbuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  {
    int nc_status = nc_get_att(group_id, NC_GLOBAL, attrname, attrbuf_tmp.dataptr);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *attrbuf = attrbuf_tmp;

  return MTK_SUCCESS;
ERROR_HANDLE:
  MtkDataBufferFree(&attrbuf_tmp);
  return status_code;
}
