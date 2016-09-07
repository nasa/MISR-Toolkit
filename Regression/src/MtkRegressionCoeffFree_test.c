/*===========================================================================
=                                                                           =
=                          MtkRegressionCoeffFree_test                      =
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
#include <stdlib.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  MTKt_RegressionCoeff rbuf = MTKT_REGRESSION_COEFF_INIT;
				/* Data buffer structure */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRegressionCoeffFree");

  /* Normal test call */
  nline = 5;
  nsample = 10;

  status = MtkRegressionCoeffAllocate(nline, nsample, &rbuf);
  if (status != MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,"(1)");
      error = MTK_TRUE;
  }

  /* Free memory */
  status = MtkRegressionCoeffFree(&rbuf);
  if (status != MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,"(2)");
      error = MTK_TRUE;
  }

  /* Attempt to free memory already freed */
  status = MtkRegressionCoeffFree(&rbuf);
  if (status != MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,"(3)");
      error = MTK_TRUE;
  }

  /* Argument check: regresbuf == NULL */
  status = MtkRegressionCoeffFree(NULL);
  if (status != MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,"(4)");
      error = MTK_TRUE;
  }

  if (error) {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  } else {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  }
}
