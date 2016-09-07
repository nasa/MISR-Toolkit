/*===========================================================================
=                                                                           =
=                        MtkRegressionCoeffAllocate_test                    =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrRegression.h"
#include <stdio.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  MTKt_RegressionCoeff rbuf = MTKT_REGRESSION_COEFF_INIT;
				/* Data buffer structure */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRegressionCoeffAllocate");

  /* Normal test call */
  nline = 5;
  nsample = 10;

  status = MtkRegressionCoeffAllocate(nline, nsample, &rbuf);
  if (status != MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,"(1)");
    error = MTK_TRUE;
  }

  if (rbuf.valid_mask.dataptr == NULL ||
      rbuf.valid_mask.nline != nline ||
      rbuf.valid_mask.nsample != nsample ||
      rbuf.valid_mask.datatype != MTKe_uint8 ||
      rbuf.slope.dataptr == NULL ||
      rbuf.slope.nline != nline ||
      rbuf.slope.nsample != nsample ||
      rbuf.slope.datatype != MTKe_float ||
      rbuf.intercept.dataptr == NULL ||
      rbuf.intercept.nline != nline ||
      rbuf.intercept.nsample != nsample ||
      rbuf.intercept.datatype != MTKe_float ||
      rbuf.correlation.dataptr == NULL ||
      rbuf.correlation.nline != nline ||
      rbuf.correlation.nsample != nsample ||
      rbuf.correlation.datatype != MTKe_float) {
    MTK_PRINT_STATUS(cn,"(2)");
    error = MTK_TRUE;
  }

  /* Free memory */
  MtkRegressionCoeffFree(&rbuf);

  /* Argument check: regressbuf == NULL */
  status = MtkRegressionCoeffAllocate(nline, nsample, NULL);
  if (status != MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,"(3)");
    error = MTK_TRUE;
  }

  /* Argument check: nline < 1 */
  nline = 0;
  status = MtkRegressionCoeffAllocate(nline, nsample, &rbuf);
  if (status != MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,"(4)");
    error = MTK_TRUE;
  }
  nline = 1;

  /* Argument check: nsample < 1 */
  nsample = 0;
  status = MtkRegressionCoeffAllocate(nline, nsample, &rbuf);
  if (status != MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,"(5)");
    error = MTK_TRUE;
  }
  nsample = 1;

  if (error) {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  } else {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  }
}
