/*===========================================================================
=                                                                           =
=                         MtkDataBufferAllocate3D                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include <stdlib.h>

/** \brief Allocate 3-dimensional Data Buffer (a buffer stack)
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we allocate a 3-dimensional data buffer with a size of 3 blocks by 5 lines by 10 samples of type \c MTKe_int16
 *
 *  \code
 *  status = MtkDataBufferAllocate3D(3, 5, 10, MTKe_int16, &databuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree3D() to free the memory used by \c databuf
 */

MTKt_status MtkDataBufferAllocate3D(
  int nblock,  /**< [IN] Number of blocks */
  int nline,   /**< [IN] Number of lines */
  int nsample, /**< [IN] Number of samples */
  MTKt_DataType datatype, /**< [IN] Data Type */
  MTKt_DataBuffer3D *databuf /**< [OUT] Data Buffer */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
				/* Data buffer structure */
  int datasize[] = MTKd_DataSize; /* Data size by data type */
  int i;			/* Index */
  int j;			/* Index */
  
  if (databuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nline < 0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (nsample < 0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (nblock < 0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  dbuf.nblock = nblock;
  dbuf.nline = nline;
  dbuf.nsample = nsample;
  dbuf.datasize = datasize[datatype];
  dbuf.datatype = datatype;

  /* Create 2D Illiffe vector of sizeof(void *) to point to 3D buffer */

  /* Allocate 1D Illiffe vector */
  dbuf.vdata = (void ***)calloc(dbuf.nblock, sizeof(void **));
  if (dbuf.vdata == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  /* Allocate 2D Illiffe vector */
  dbuf.vdata[0] = (void **)calloc(dbuf.nblock * dbuf.nline, sizeof(void *));
  if (dbuf.vdata[0] == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  /* Connect Illiffe vector */
  for (i = 1; i < nblock; i++) {
    dbuf.vdata[i] = (void *)((unsigned char *)(dbuf.vdata[i-1]) +
			     dbuf.nline * sizeof(void *));
  }

  /* Create 3D buffer and point the 2D Illiffe vector to it */

  /* Allocate 3D buffer */
  dbuf.vdata[0][0] = (void *)calloc(dbuf.nblock * dbuf.nline * dbuf.nsample,
				    dbuf.datasize);
  if (dbuf.vdata[0][0] == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  /* Connect 1D Illiffe vector */
  for (i = 1; i < nblock; i++) {
    dbuf.vdata[i][0] = (void *)((unsigned char *)(dbuf.vdata[i-1][0]) +
				dbuf.nline * dbuf.nsample * dbuf.datasize);
  }

  /* Connect 2D Illiffe vector */
  for (i = 0; i < nblock; i++) {
    for (j = 1; j < nline; j++) {
      dbuf.vdata[i][j] = (void *)((unsigned char *)(dbuf.vdata[i][j-1]) +
				  dbuf.nsample * dbuf.datasize);
    }
  }

  /* Hook the union to the void data pointer */
  dbuf.data.v = dbuf.vdata;
  /* Hook the data pointer to the data */
  dbuf.dataptr = dbuf.vdata[0][0];

  *databuf = dbuf;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
