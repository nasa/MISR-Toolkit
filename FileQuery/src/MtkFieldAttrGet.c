/*===========================================================================
=                                                                           =
=                              MtkFieldAttrGet                              =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <mfhdf.h>
#include <HdfEosDef.h>

/** \brief Get a file attribute
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the \c _FillValue attribute from the CloudMotionCrossTrack field in the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFieldAttrGet("MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf", "CloudMotionCrossTrack", "_FillValue", &attrbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by \c attrbuf
 */

MTKt_status MtkFieldAttrGet(
  const char *filename,    /**< [IN] File name */
  const char *fieldname,    /**< [IN] Field name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFieldAttrGetNC(filename, fieldname, attrname, attrbuf); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFieldAttrGetHDF(filename, fieldname, attrname, attrbuf); // try HDF
}

MTKt_status MtkFieldAttrGetNC(
  const char *filename,    /**< [IN] File name */
  const char *fieldname,    /**< [IN] Field name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
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
  status = MtkFieldAttrGetNcid(ncid, fieldname, attrname, attrbuf);
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

MTKt_status MtkFieldAttrGetHDF(
  const char *filename,    /**< [IN] File name */
  const char *fieldname,    /**< [IN] Field name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 fid = FAIL;		/* HDF-EOS File id */

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read grid attribute */
  status = MtkFieldAttrGetFid(fid, fieldname, attrname, attrbuf);
  MTK_ERR_COND_JUMP(status);

  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);

  return MTK_SUCCESS;
ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}


/** \brief Version of MtkFieldAttrGet that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFieldAttrGetFid(
  int32 fid,               /**< [IN] HDF-EOS File ID */
  const char *fieldname,    /**< [IN] Field name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdf_status;
  int32 attr_index;
  int32 hdf_datatype;
  int32 count;
  MTKt_DataType datatype;       /* Mtk data type */
  MTKt_DataBuffer attrbuf_tmp = MTKT_DATABUFFER_INIT; /* Temp attribute buffer */
  char attr_name_tmp[MAXSTR];
  int32 HDFfid = FAIL; /* HDF File ID */
  int32 sdInterfaceID = FAIL; /* SD Interface ID (file level) */
  int32 sd_id = FAIL; /* SDS ID (field level) */
  int32 sds_index = 0; /* SDS index/offset for field */

  if (fieldname == NULL || attrname == NULL || attrbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  /* Transform HDF-EOS fid to plain HDF fid and SDS fid for file */
  EHidinfo(fid, &HDFfid, &sdInterfaceID);
  
  sds_index = SDnametoindex(sdInterfaceID, fieldname);
  sd_id = SDselect(sdInterfaceID,sds_index);  

  /* Find attribute index */
  hdf_status = attr_index = SDfindattr(sd_id, attrname);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFINDATTR_FAILED);

  /* Get attribute information */
  hdf_status = SDattrinfo(sd_id, attr_index, attr_name_tmp, &hdf_datatype, &count);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDATTRINFO_FAILED);

  /* Convert to Mtk Data Type */
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  /* Allocate Memory */
  status = MtkDataBufferAllocate(1, count, datatype, &attrbuf_tmp);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  /* Read attribute */
  hdf_status = SDreadattr(sd_id, attr_index, attrbuf_tmp.dataptr);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDREADATTR_FAILED);

  *attrbuf = attrbuf_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:

  MtkDataBufferFree(&attrbuf_tmp);

  return status_code;
}

MTKt_status MtkFieldAttrGetNcid(
  int ncid,               /**< [IN] netCDF File ID */
  const char *fieldname,    /**< [IN] Field name */
  const char *attrname,    /**< [IN] Attribute name */
  MTKt_DataBuffer *attrbuf /**< [OUT] Attribute value */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  MTKt_DataType datatype;       /* Mtk data type */
  MTKt_DataBuffer attrbuf_tmp = MTKT_DATABUFFER_INIT; /* Temp attribute buffer */
  int *gids = NULL;

  if (fieldname == NULL || attrname == NULL || attrbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);


  /* Iterate over groups and fields until fieldname matches.  Only the first match will be tried. */
  MTKt_ncvarid var;
  int found = 0;
  {
    int number_group;
    int nc_status = nc_inq_grps(ncid, &number_group, NULL);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    
    gids = (int *)calloc(number_group, sizeof(int));
    if (gids == NULL) MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

    nc_status = nc_inq_grps(ncid, &number_group, gids);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    for (int i = 0 ; i < number_group ; i++) {
      int gid = gids[i];
      
      status = MtkNCVarId(gid, fieldname, &var);
      if (status == MTK_SUCCESS) {
        found = 1;
        break;
      }
    }
    free(gids); 
    gids = NULL;
  }

  if (found == 0) {
    MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  nc_type nc_datatype;
  {
    int nc_status = nc_inq_atttype(var.gid, var.varid, attrname, &nc_datatype);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  status = MtkDataBufferAllocate(1, 1, datatype, &attrbuf_tmp);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  {
    int nc_status = nc_get_att(var.gid, var.varid, attrname, attrbuf_tmp.dataptr);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *attrbuf = attrbuf_tmp;
  
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (gids != NULL) {
    free(gids);
  }
  MtkDataBufferFree(&attrbuf_tmp);

  return status_code;
}
