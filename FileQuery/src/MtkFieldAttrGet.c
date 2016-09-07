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
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdf_status;
  int32 fid = FAIL;

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  /* Open HDF File */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Get list of field attributes. */
  status = MtkFieldAttrGetFid(fid, fieldname, attrname, attrbuf);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */
  hdf_status = GDclose(fid);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;
 
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fid != FAIL)
    GDclose(fid);
  
  return status_code;
}

/** \brief Version of MtkFieldAttrGet that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFieldAttrGetFid(
  int32 fid,            /**< [IN] HDF-EOS file identifier */
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
