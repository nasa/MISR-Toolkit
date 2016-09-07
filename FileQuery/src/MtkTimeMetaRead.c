/*===========================================================================
=                                                                           =
=                               MtkTimeMetaRead                             =
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
#include <string.h>
#include <hdf.h>
#include <mfhdf.h>

/** \brief Read time metadata from L1B2 Ellipsoid product file
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the time metadata from the file \c MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf
 *
 *  \code
 *  status = MtkTimeMetaRead("MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf", &time_metadata);
 *  \endcode
 *
 *  \note
 *  To compute pixel time use MtkPixelTime().
 */

MTKt_status MtkTimeMetaRead(
  const char *filename, /**< [IN] L1B2 product file */
  MTKt_TimeMetaData *time_metadata /**< [OUT] Time metadata */ )
{
  MTKt_status status;	   /* Return status */
  MTKt_status status_code; /* Return code of this function */
  int32 hdf_status;        /* HDF-EOS return status */
  int32 sd_id = FAIL;      /* HDF SD file identifier. */
  int32 hdf_id = FAIL;
  
  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_id = HDFopen(filename, DFACC_READ, 0);
  if (hdf_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFOPEN_FAILED);

  /* Open HDF File */
  hdf_status = sd_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read time metadata. */
  status = MtkTimeMetaReadFid(hdf_id, sd_id, time_metadata);
  MTK_ERR_COND_JUMP(status);
    
  /* Close HDF File */
  hdf_status = SDend(sd_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sd_id = FAIL;
   
  /* Close HDF file */
  hdf_status = HDFclose(hdf_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFCLOSE_FAILED);
  hdf_id = FAIL;

  return MTK_SUCCESS;
 
ERROR_HANDLE:
  if (hdf_id != FAIL)
    HDFclose(hdf_id);
  
  if (sd_id != FAIL)
    SDend(sd_id);

  return status_code;
}

/** \brief Version of MtkTimeMetaRead that takes an HDF SD file identifier and HDF file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkTimeMetaReadFid(
  int32 hdf_id,         /**< [IN] HDF file identifier */
  int32 sd_id,          /**< [IN] HDF SD file identifier */
  MTKt_TimeMetaData *time_metadata /**< [OUT] Time metadata */ )
{
  MTKt_status status;	   /* Return status */
  MTKt_status status_code; /* Return code of this function */
  int block;
  int i;
  MTKt_DataBuffer number_transform = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer ref_time = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer start_line = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer number_line = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer coeff_line = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer som_ctr_x = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer som_ctr_y = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer camera = MTKT_DATABUFFER_INIT;
  const char *cam_list[9] = {"DF", "CF", "BF", "AF", "AN", "AA", "BA", "CA", "DA"};
  
  if (time_metadata == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
    
  /* Read block metadata */
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"number_transform", &number_transform);
  MTK_ERR_COND_JUMP(status);
  
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.ref_time", &ref_time);
  MTK_ERR_COND_JUMP(status);
      
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.start_line", &start_line);
  MTK_ERR_COND_JUMP(status);
    
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.number_line", &number_line);
  MTK_ERR_COND_JUMP(status);
 
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.coeff_line", &coeff_line);
  MTK_ERR_COND_JUMP(status);
  
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.som_ctr.x", &som_ctr_x);
  MTK_ERR_COND_JUMP(status);
    
  status = MtkFileBlockMetaFieldReadFid(hdf_id, "PerBlockMetadataRad",
					"transform.som_ctr.y", &som_ctr_y);
  MTK_ERR_COND_JUMP(status);
  
  status = MtkFileToPathFid(sd_id, &time_metadata->path);
  MTK_ERR_COND_JUMP(status);
  
  status = MtkFileToBlockRangeFid(sd_id, &time_metadata->start_block,
                               &time_metadata->end_block);  	
  MTK_ERR_COND_JUMP(status);
  
  status = MtkFileAttrGetFid(sd_id, "Camera", &camera);
  MTK_ERR_COND_JUMP(status);

  strcpy(time_metadata->camera, cam_list[camera.data.i32[0][0] - 1]);

  /* Block is 0-based when read from Per Block Metadata convert
     to 1-based when copying to Time Metadata structure */
     
  memcpy(time_metadata->number_transform + 1, number_transform.dataptr,
         sizeof(MTKt_int32) * number_transform.nline);
         
  for (block = 0; block < ref_time.nline; ++block)
  {
  	/* ref_time is stored as two strings concatenated together, split
  	   when copying into Time Metadata structure */
    
    strncpy(time_metadata->ref_time[block + 1][0], ref_time.data.c8[block],
            MTKd_DATETIME_LEN - 1);
	time_metadata->ref_time[block + 1][0][MTKd_DATETIME_LEN - 1] = '\0';

    strncpy(time_metadata->ref_time[block + 1][1], ref_time.data.c8[block] + MTKd_DATETIME_LEN - 1,
            MTKd_DATETIME_LEN - 1);
    time_metadata->ref_time[block + 1][1][MTKd_DATETIME_LEN - 1] = '\0';
  }
  
  memcpy(time_metadata->start_line + 1, start_line.dataptr,
         sizeof(MTKt_int32) * start_line.nline * start_line.nsample);
  
  memcpy(time_metadata->number_line + 1, number_line.dataptr,
         sizeof(MTKt_int32) * number_line.nline * number_line.nsample);
  
  for (block = 0; block < coeff_line.nline; ++block)
    for (i = 0; i < 6; ++i)
    {
      /* reorder coeff_line by grid cell */
  	  time_metadata->coeff_line[block + 1][i][0] = coeff_line.data.d[block][i];
      time_metadata->coeff_line[block + 1][i][1] = coeff_line.data.d[block][i + 6];
    }
  
  memcpy(time_metadata->som_ctr_x + 1, som_ctr_x.dataptr,
         sizeof(MTKt_double) * som_ctr_x.nline * som_ctr_x.nsample);
  
  memcpy(time_metadata->som_ctr_y + 1, som_ctr_y.dataptr,
         sizeof(MTKt_double) * som_ctr_y.nline * som_ctr_y.nsample);
  
  MtkDataBufferFree(&number_transform);
  MtkDataBufferFree(&ref_time);
  MtkDataBufferFree(&start_line);
  MtkDataBufferFree(&number_line);
  MtkDataBufferFree(&coeff_line);
  MtkDataBufferFree(&som_ctr_x);
  MtkDataBufferFree(&som_ctr_y);
  MtkDataBufferFree(&camera);
   
  return MTK_SUCCESS;
 
ERROR_HANDLE:
  MtkDataBufferFree(&number_transform);
  MtkDataBufferFree(&ref_time);
  MtkDataBufferFree(&start_line);
  MtkDataBufferFree(&number_line);
  MtkDataBufferFree(&coeff_line);
  MtkDataBufferFree(&som_ctr_x);
  MtkDataBufferFree(&som_ctr_y);
  MtkDataBufferFree(&camera);
  
  return status_code;
}
