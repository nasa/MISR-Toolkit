/*===========================================================================
=                                                                           =
=                            MtkDataBufferFree                              =
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

/** \brief Free data buffer
 *
 *  \return MTK_SUCCESS
 *
 *  \par Example:
 *  In this example, we have a MTKt_DataBuffer called \c databuf that was previously allocated with MtkDataBufferAllocate() and we wish to free it.
 *
 *  \code
 *  status = MtkDataBufferFree(&databuf);
 *  \endcode
 */

MTKt_status MtkDataBufferFree(
  MTKt_DataBuffer *databuf /**< [IN,OUT] Data Buffer */ )
{
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				/* Data buffer structure */

  if (databuf == NULL)
    return MTK_SUCCESS;
  
  /* Free 2D buffer */
  if (!databuf->imported)
    if (databuf->dataptr != NULL)
      free(databuf->dataptr);

  /* Free 1D Illiffe vector */
  if (databuf->vdata != NULL)
     free(databuf->vdata);

  *databuf = dbuf;

  return MTK_SUCCESS;
}
