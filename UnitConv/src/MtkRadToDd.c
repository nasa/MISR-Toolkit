/*===========================================================================
=                                                                           =
=                                MtkRadToDd                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUnitConv.h"
#include "MisrError.h"
/* M_PI is not defined in math.h in Linux unless __USE_BSD is defined */
/* and you can define it at the gcc command-line if -ansi is set */
#ifndef __USE_BSD
# define __USE_BSD
#endif
#include <math.h>

/** \brief Convert radians to decimal degrees 
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 2.270373886 radians to decimal degrees.
 *
 *  \code
 *  status = MtkRadToDd(2.270373886, &dd);
 *  \endcode
 */

MTKt_status MtkRadToDd(
  double rad, /**< [IN] Radians */
  double *dd  /**< [OUT] Decimal degrees */ )
{
  MTKt_status status_code;          /* Return status of this function */

  if (dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *dd = rad / (M_PI / 180.0);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
