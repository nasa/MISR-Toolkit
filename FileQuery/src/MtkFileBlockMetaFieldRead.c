/*===========================================================================
=                                                                           =
=                         MtkFileBlockMetaFieldRead                          =
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
#include <hdf.h>

/** \brief Read a block metadata field
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the field \c lock_coor_ulc_som_meter.x from the \c PerBlockMetadataCommon block metadata structure in the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileBlockMetaFieldRead("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "PerBlockMetadataCommon", "lock_coor_ulc_som_meter.x", &blockmetabuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by \a blockmetabuf
 *
 *  \note
 *  When reading the \c transform.ref_time field in the \c PerBlockMetadataRad structure the two strings are concatenated, and there is no terminating NULL.
 */

MTKt_status MtkFileBlockMetaFieldRead(
  const char *filename, /**< [IN] Filename */
  const char *blockmetaname, /**< [IN] Block metadata structure name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *blockmetabuf /**< [OUT] Block metadata values */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdfstatus;
  int32 file_id = FAIL;

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  file_id = HDFopen(filename, DFACC_READ, 0);
  if (file_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFOPEN_FAILED);

  /* Read block metadata field. */
  status = MtkFileBlockMetaFieldReadFid(file_id, blockmetaname, fieldname, 
					blockmetabuf);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */  
  hdfstatus = HDFclose(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFCLOSE_FAILED);
  file_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (file_id != FAIL)
    HDFclose(file_id);
  
  return status_code;
}

/** \brief Version of MtkFileBlockMetaFileRead that takes an HDF file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileBlockMetaFieldReadFid(
  int32 file_id,               /**< [IN] HDF file identifier */
  const char *blockmetaname, /**< [IN] Block metadata structure name */
  const char *fieldname, /**< [IN] Field name */
  MTKt_DataBuffer *blockmetabuf /**< [OUT] Block metadata values */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdfstatus;
  int32 hdf_datatype;
  int32 vdata_ref = FAIL;
  int32 vdata_id = FAIL;
  int32 count;
  int32 field_index;
  int32 n_records;
  MTKt_DataBuffer blockmetabuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp attribute buffer */
  MTKt_DataType datatype;       /* Mtk data type */ 
  int vstart_active = 0;

  if (blockmetaname == NULL || fieldname == NULL ||
      blockmetabuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Initialize the vdata interface */
  hdfstatus = Vstart(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSTART_FAILED);
  vstart_active = 1;
  
  /* Find reference number of block metadata structure */
  vdata_ref = VSfind(file_id, blockmetaname);
  if (vdata_ref == 0) /* Failure */
    MTK_ERR_CODE_JUMP(MTK_HDF_VSFIND_FAILED);
  
  vdata_id = VSattach(file_id, vdata_ref, "r");
  if (vdata_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSATTACH_FAILED);
  
  /* Find field index */
  hdfstatus = VSfindex(vdata_id, fieldname, &field_index);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSFINDEX_FAILED);
  
  /* Determine data type of field */
  hdf_datatype = VFfieldtype(vdata_id, field_index);
  if (hdf_datatype == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VFFIELDTYPE_FAILED);
  
  /* Convert from HDF data type to MTK data type */
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  MTK_ERR_COND_JUMP(status);
  
  /* Number of records in a field */
  n_records = VSelts(vdata_id);
  if (n_records == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSELTS_FAILED);
    
  /* Allocate Memory */
  if (datatype == MTKe_char8) /* Data is a string */
  {
  	count = VSsizeof(vdata_id, (char*)fieldname);
  	if (count == FAIL)
  	  MTK_ERR_CODE_JUMP(MTK_HDF_VSSIZEOF_FAILED);
  	
  	status = MtkDataBufferAllocate(n_records, count, datatype, &blockmetabuf_tmp);
    MTK_ERR_COND_JUMP(status);
  }
  else
  {
  	count = VFfieldorder(vdata_id, field_index);
    if (count == FAIL)
  	  MTK_ERR_CODE_JUMP(MTK_HDF_VFFIELDORDER_FAILED);
  	  
    status = MtkDataBufferAllocate(n_records, count, datatype, &blockmetabuf_tmp);
    MTK_ERR_COND_JUMP(status);
  }
  
  /* Select field to read */
  hdfstatus = VSsetfields(vdata_id, fieldname);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSSETFIELDS_FAILED);

  /* Read field, since only one field is being read there is no need to unpack */
  hdfstatus = VSread(vdata_id, blockmetabuf_tmp.dataptr, n_records, FULL_INTERLACE);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSREAD_FAILED);

  hdfstatus = VSdetach(vdata_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSDETACH_FAILED);
  
  /* End access to the vdata interface */     
  hdfstatus = Vend(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VEND_FAILED);
  vstart_active = 0;
  
  *blockmetabuf = blockmetabuf_tmp;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (vdata_id != FAIL)
    VSdetach(vdata_id);
  
  if (vstart_active)
    Vend(file_id);
  
  MtkDataBufferFree(&blockmetabuf_tmp);

  return status_code;
}
