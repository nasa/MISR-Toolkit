/*===========================================================================
=                                                                           =
=                          MtkLinearRegressionCalc                          =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrRegression.h"
#include "MisrUtil.h"
#include <stdlib.h>
#include <math.h>

/** \brief Use linear regression to fit a set of observations (x,y) to the model:  y(x) = a + b * x.    The values of 'x' are assumed to be known exactly.  The values of 'y' may have an associated uncertainty, 'y_sigma'.  Measurements with larger uncertainty are given less weight.  Uncertainty must be greater than 0.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *
 *  \code
 *  double x[4] = {1,5,7,8};
 *  double y[4] = {12,14,19,21};
 *  double y_sigma[4] = {0.1,0.2,0.2,0.3};
 *  double a, b;
 *  status = MtkLinearRegressionCalc(4, x, y, y_sigma, &a, &b)
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkLinearRegressionCalc(
  int Size,   /**< [IN] Size of X and Y arrays */
  const double *X, /**< [IN] X array */
  const double *Y, /**< [IN] Y array */
  const double *Y_Sigma, /**< [IN] Uncertainty in Y */
  double *A, /**< [OUT] A */
  double *B, /**< [OUT] B */
  double *Correlation /**< [OUT] Correlation */
)
{
  MTKt_status status_code;      /* Return status of this function */
  double s, sx, sxx, sxy, sy, syy, delta;
  double pop_sd_x, pop_sd_y;
  int i; 		

  /* -------------------------------------------------------------- */
  /* Argument check: Size < 1                                       */
  /* -------------------------------------------------------------- */

  if (Size < 1) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: X = NULL                                       */
  /* -------------------------------------------------------------- */

  if (X == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Y = NULL                                       */
  /* -------------------------------------------------------------- */

  if (Y == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Y_Sigma = NULL                                 */
  /* -------------------------------------------------------------- */

  if (Y_Sigma == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: A = NULL                                       */
  /* -------------------------------------------------------------- */

  if (A == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: B = NULL                                       */
  /* -------------------------------------------------------------- */

  if (B == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Correlation = NULL                             */
  /* -------------------------------------------------------------- */

  if (Correlation == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* -------------------------------------------------------------- */
  /* Calculate linear regression                                    */
  /* Argument check: Y_Sigma[x] <= 0.0                              */
  /* -------------------------------------------------------------- */

  s = 0;
  sx = 0;
  sy = 0;
  sxx = 0;
  sxy = 0;
  syy = 0;
  for (i = 0 ; i < Size ; i++) {
    double t;

    if (Y_Sigma[i] <= 0.0) {
      MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
    }

    t = 1.0 / (Y_Sigma[i] * Y_Sigma[i]);
    s += 1.0 * t;
    sx += X[i] * t;
    sy += Y[i] * t;
    sxx += (X[i] * X[i]) * t;
    sxy += (X[i] * Y[i]) * t;
    syy += (Y[i] * Y[i]) * t;
  }

  delta = s * sxx - sx * sx;

  if (delta == 0.0) {
    MTK_ERR_CODE_JUMP(MTK_DIV_BY_ZERO);
  }

  pop_sd_x = sqrt(s*sxx-sx*sx);
  pop_sd_y = sqrt(s*syy-sy*sy);

  if (pop_sd_x == 0.0 || pop_sd_y == 0.0) {
    MTK_ERR_CODE_JUMP(MTK_DIV_BY_ZERO);
  }

  *A = (sxx * sy - sx * sxy) / delta;
  *B = (s * sxy - sx * sy) / delta;

  *Correlation = ( ((s * sxy) - (sx * sy))  / 
		   ( pop_sd_x * pop_sd_y )
		   );

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
