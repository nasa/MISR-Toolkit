/*===========================================================================
=                                                                           =
=                                 MtkCache                                  =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCache.h"
#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------- */
/* MtkCacheInit                                    */
/* ----------------------------------------------- */

/** \brief Initialize Cache
 *
 *  \return Initialized Cache
 */

MTKt_status MtkCacheInitFid( int32 fid,             /**< HDF-EOS file identifier */
			  const char *gridname,  /**< Grid name */
			  const char *fieldname, /**< Field name */
			  MTKt_Cache *cache      /**< Cache */ ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer buf_tmp = MTKT_DATABUFFER_INIT;
				/* Temp block data buffer */
  MTKt_DataType datatype;	/* Mtk datatype */
  int i;			/* Loop index */


  cache->fid = fid;
  cache->ncid = 0;
  cache->gridname = (char *)malloc(strlen(gridname)+1);
  strcpy(cache->gridname, gridname);
  cache->fieldname = (char *)malloc(strlen(fieldname)+1);
  strcpy(cache->fieldname, fieldname);

  cache->block_cnt = 0;
 
  /* Block cache has NBLOCK+1 elements to make it 1-based, thus i <= NBLOCK */

  for (i = 0; i <= NBLOCK; i++) {
    cache->block[i].valid = MTK_FALSE;
    cache->block[i].buf = buf_tmp;
  }

  /* Set fill value, if available otherwise set to zero */

  status = MtkFillValueGetFid(fid, gridname, fieldname, &(cache->fill));
  if (status != MTK_SUCCESS) {
    status = MtkFileGridFieldToDataTypeFid(fid, gridname, fieldname,
					   &datatype);
    if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);
    status = MtkDataBufferAllocate(1, 1, datatype, &(cache->fill));
    if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);
    /* Calloc is used to clear the fill buffer, so zero is the default fill */
  }

  return MTK_SUCCESS;
 ERROR_HANDLE:
  return status_code;
}

MTKt_status MtkCacheInitNcid( int ncid,             /**< file identifier */
			  const char *gridname,  /**< Grid name */
			  const char *fieldname, /**< Field name */
			  MTKt_Cache *cache      /**< Cache */ ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer buf_tmp = MTKT_DATABUFFER_INIT;
				/* Temp block data buffer */
  MTKt_DataType datatype;	/* Mtk datatype */
  int i;			/* Loop index */


  cache->ncid = ncid;
  cache->fid = FAIL;
  cache->gridname = (char *)malloc(strlen(gridname)+1);
  strcpy(cache->gridname, gridname);
  cache->fieldname = (char *)malloc(strlen(fieldname)+1);
  strcpy(cache->fieldname, fieldname);

  cache->block_cnt = 0;
 
  /* Block cache has NBLOCK+1 elements to make it 1-based, thus i <= NBLOCK */

  for (i = 0; i <= NBLOCK; i++) {
    cache->block[i].valid = MTK_FALSE;
    cache->block[i].buf = buf_tmp;
  }

  /* Set fill value, if available otherwise set to zero */

  status = MtkFillValueGetNcid(ncid, gridname, fieldname, &(cache->fill));
  if (status != MTK_SUCCESS) {
    status = MtkFileGridFieldToDataTypeNcid(ncid, gridname, fieldname,
					   &datatype);
    if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);
    status = MtkDataBufferAllocate(1, 1, datatype, &(cache->fill));
    if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);
    /* Calloc is used to clear the fill buffer, so zero is the default fill */
  }

  return MTK_SUCCESS;
 ERROR_HANDLE:
  return status_code;
}

/* ----------------------------------------------- */
/* MtkCachePixelGet                                */
/* ----------------------------------------------- */

/** \brief Get pixel from cache
 *
 *  \return Pixel
 */

MTKt_status MtkCachePixelGet( MTKt_Cache *cache, /**< Cache */
			      int block,         /**< Block Number */
			      float line,        /**< Line */
			      float sample,      /**< Sample */
			      void *pixel        /**< Pixel */ ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int rline;			/* Rounded line */
  int rsample;			/* Rounded sample */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int datasize;			/* Element data size */
  char *dptr;			/* Pointer to data buffer */

  if (block <= 0) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (!cache->block[block].valid) {
    status = MtkCacheLoad(cache, block);
    if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);
  }

  nline = cache->block[block].buf.nline;
  nsample = cache->block[block].buf.nsample;
  datasize = cache->block[block].buf.datasize;
  rline = (int)floor(line+.5);
  rsample = (int)floor(sample+.5);

  if (rline < 0 || sample < 0 ||
      rline >= nline || rsample >= nsample) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  dptr = cache->block[block].buf.dataptr;
  memcpy(pixel, (void *)&(dptr[rline*nsample*datasize + rsample*datasize]),
	 datasize);

  return MTK_SUCCESS;
 ERROR_HANDLE:
  memcpy(pixel, cache->fill.dataptr, cache->fill.datasize); 
  return status_code;
}

/* ----------------------------------------------- */
/* MtkCacheLoad                                    */
/* ----------------------------------------------- */

/** \brief Load Cache
 *
 *
 */

MTKt_status MtkCacheLoad( MTKt_Cache *cache, /**< Cache */
			  int block          /**< Block Number */ ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int iblock;			/* Block to release */

  /* Free a block if cache is full */

  if (cache->block_cnt >= BLOCK_CACHE_MAX) {
    iblock = block - BLOCK_CACHE_MAX;
    MtkDataBufferFree(&(cache->block[iblock].buf));
    cache->block[iblock].valid = MTK_FALSE;
    cache->block_cnt--;
  }

  /* Read a block */

  if (cache->ncid > 0) {
    status = MtkReadBlockNcid(cache->ncid, cache->gridname,
                              cache->fieldname, block,
                              &(cache->block[block].buf));
  } else {
    status = MtkReadBlockFid(cache->fid, cache->gridname,
                             cache->fieldname, block,
                             &(cache->block[block].buf));
  }
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  cache->block[block].valid = MTK_TRUE;
  cache->block_cnt++;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  return status_code;
}

/* ----------------------------------------------- */
/* MtkCacheFree                                    */
/* ----------------------------------------------- */

/** \brief Free Cache
 *
 * 
 */
void MtkCacheFree( MTKt_Cache *cache /**< Cache */) {

  int i;			/* Loop index */

  /* Block cache has NBLOCK+1 elements to make it 1-based, thus i <= NBLOCK */

  for (i = 0; i <= NBLOCK; i++) {
    if (cache->block[i].valid) {
      MtkDataBufferFree(&(cache->block[i].buf));
      cache->block[i].valid = MTK_FALSE;
      cache->block_cnt--;
    }
  }
  MtkDataBufferFree(&(cache->fill));
  free((void *)cache->gridname);
  free((void *)cache->fieldname);
}
