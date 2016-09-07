/*===========================================================================
=                                                                           =
=                               MtkPixelTime                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrFileQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"

#define TIME_BETWEEN_LINE 0.0408 /* Seconds */

/** \brief Given SOM Coordinates compute pixel time
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  Time Format: YYYY-MM-DDThh:mm:ss.ssssssZ (ISO 8601)
 *
 *  \par Example:
 *  In this example, we get the pixel time for SOM X 8176300.0 and SOM Y 574200.0.
 *
 *  \code
 *  status = MtkPixelTime(time_metadata, 8176300.0, 574200.0, pixel_time);
 *  \endcode
 *
 *  \note
 *  Refer to MtkTimeMetaRead() to retrieve time metadata.
 */

MTKt_status MtkPixelTime(
  MTKt_TimeMetaData time_metadata, /**< [IN] Time metadata from L1B2 product file */
  double som_x, /**< [IN] SOM X */
  double som_y, /**< [IN] SOM Y */
  char pixel_time[MTKd_DATETIME_LEN] /**< [OUT] Pixel time */ )
{
  MTKt_status status;	   /* Return status */
  MTKt_status status_code; /* Return code of this function */
  int block;
  float line_275;
  float sample_275;
  int trfm = -1;
  int transform;
  double delta_line;
  double delta_samp;
  double C1, C2, C3, C4, C5, C6;
  double delta_imgl;
  double offset;
  double tai_ref_time;
  
  if (pixel_time == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  status = MtkSomXYToBls(time_metadata.path,MAXRESOLUTION,som_x,som_y,&block,&line_275,&sample_275);  	
  MTK_ERR_COND_JUMP(status);

  /* Make line relative to start of orbit */
  line_275 += (float)(512.0 * (block - 1));
  
  if (block < time_metadata.start_block || block > time_metadata.end_block ||
      time_metadata.number_transform[block] <= 0)
    MTK_ERR_CODE_JUMP(MTK_FAILURE);

  /* Determine which transform to use */
  for (transform = 0; transform < NGRIDCELL; ++transform)
  {
    if (time_metadata.start_line[block][transform] == 0 &&
        time_metadata.number_line[block][transform] == 0)   
      continue;
  
    if (line_275 > time_metadata.start_line[block][transform] &&
        line_275 < time_metadata.start_line[block][transform] +
                   time_metadata.number_line[block][transform])
    {
        trfm = transform;
        break;
    }
  }

  if (trfm < 0 || trfm > 1)
    MTK_ERR_CODE_JUMP(MTK_FAILURE);

  delta_line = line_275 - time_metadata.som_ctr_x[block][trfm];
  delta_samp = sample_275 - time_metadata.som_ctr_y[block][trfm];

  C1 = time_metadata.coeff_line[block][0][trfm];
  C2 = time_metadata.coeff_line[block][1][trfm];
  C3 = time_metadata.coeff_line[block][2][trfm];
  C4 = time_metadata.coeff_line[block][3][trfm];
  C5 = time_metadata.coeff_line[block][4][trfm];
  C6 = time_metadata.coeff_line[block][5][trfm];

  delta_imgl = C1 + C2 * delta_line + C3 * delta_samp + C4 * delta_samp * delta_samp +
               C5 * delta_line * delta_samp + C6 * delta_samp * delta_samp * delta_samp;

  offset = delta_imgl * TIME_BETWEEN_LINE;
  
  status = MtkUtcToTai(time_metadata.ref_time[block][trfm], &tai_ref_time);
  MTK_ERR_COND_JUMP(status);
  
  status = MtkTaiToUtc(tai_ref_time + offset, pixel_time);
  MTK_ERR_COND_JUMP(status);
    
  return MTK_SUCCESS;
 
ERROR_HANDLE:
  return status_code;
}
