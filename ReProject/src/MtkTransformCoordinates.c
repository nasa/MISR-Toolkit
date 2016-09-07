/*===========================================================================
=                                                                           =
=                         MtkTransformCoordinates                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include "MisrError.h"
#include "MisrUtil.h"

/** \brief Transforms latitude/longitude coordinates into line/sample coordinates for a given mapinfo.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we tranform latitude and longitude 2-D grid into line and sample represented by the given mapinfo.
 *
 *  \code
 *  status = MtkTransformCoordinates(mapinfo, latbuf, latbuf, &linebuf, &samplebuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by linebuf and samplebuf
*/

MTKt_status MtkTransformCoordinates(
  MTKt_MapInfo mapinfo,       /**< [IN] Mapinfo structure */
  MTKt_DataBuffer latbuf,     /**< [IN] Latitude data buffer */
  MTKt_DataBuffer lonbuf,     /**< [IN] Longitue data buffer */
  MTKt_DataBuffer *linebuf,   /**< [OUT] Line data buffer */
  MTKt_DataBuffer *samplebuf  /**< [OUT] Sample data buffer */ )
{
  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer line = MTKT_DATABUFFER_INIT;
                                /* Line data buffer structure */
  MTKt_DataBuffer sample = MTKT_DATABUFFER_INIT;
                                /* Sample data buffer structure */

  /* Check latitude and longitude buffer sizes */
  if (latbuf.nline != lonbuf.nline)
    MTK_ERR_COND_JUMP(MTK_DIMENSION_MISMATCH);
  if (latbuf.nsample != lonbuf.nsample)
    MTK_ERR_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Check latitude and longitude buffer datatypes */
  if (latbuf.datatype != MTKe_double)
    MTK_ERR_COND_JUMP(MTK_DATATYPE_MISMATCH);
  if (lonbuf.datatype != MTKe_double)
    MTK_ERR_COND_JUMP(MTK_DATATYPE_MISMATCH);

   /* --------------------------------------------------- */
  /* Allocate buffers of the latitude and longitude data */
  /* --------------------------------------------------- */

  status = MtkDataBufferAllocate(latbuf.nline, latbuf.nsample,
				 MTKe_float, &line);
  MTK_ERR_COND_JUMP(status);
  status = MtkDataBufferAllocate(latbuf.nline, latbuf.nsample,
				 MTKe_float, &sample);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------- */
  /* Transform lat/lon to line/sample for give mapinfo */
  /* ------------------------------------------------- */

  status = MtkLatLonToLSAry(mapinfo, latbuf.nline * latbuf.nsample,
			    latbuf.dataptr, lonbuf.dataptr,
			    line.dataptr, sample.dataptr);
  if (status == MTK_NULLPTR || status == MTK_BAD_ARGUMENT)
    MTK_ERR_CODE_JUMP(status);

  *linebuf = line;
  *samplebuf = sample;

  return MTK_SUCCESS;
ERROR_HANDLE:
  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample );
  return status_code;
}
