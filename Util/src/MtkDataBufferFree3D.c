/*===========================================================================
=                                                                           =
=                           MtkDataBufferFree3D                             =
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
#include "MisrError.h"
#include <stdlib.h>

/** \brief Free 3-dimensional data buffer (a buffer stack)
 *
 *  \return MTK_SUCCESS
 *
 *  \par Example:
 *  In this example, we have a MTKt_DataBuffer3D called \c databuf that was previously allocated with MtkDataBufferAllocate3D() and we wish to free it.
 *
 *  \code
 *  status = MtkDataBufferFree3D(&databuf);
 *  \endcode
 */

MTKt_status MtkDataBufferFree3D(
  MTKt_DataBuffer3D *databuf /**< [IN,OUT] Data Buffer */ )
{
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
				/* Data buffer structure */

  if (databuf == NULL)
    return MTK_SUCCESS;
  
  /* Free 3D buffer */
  if (databuf->dataptr != NULL)
    free(databuf->dataptr);

  /* Free Illiffe vectors */
  if (databuf->vdata != NULL) {
    if (databuf->vdata[0] != NULL) {
      /* Free 2D Illiffe vector */
      free(databuf->vdata[0]);
    }
    /* Free 1D Illiffe vector */
    free(databuf->vdata);
  }

  *databuf = dbuf;

  return MTK_SUCCESS;
}
