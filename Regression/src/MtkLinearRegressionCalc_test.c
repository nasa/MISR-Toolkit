/*===========================================================================
=                                                                           =
=                        MtkLinearRegressionCalc_test                       =
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
#include <float.h>

#define SCm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 100 * fabs(x))

#define SIZE 5

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  double x[SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double y[SIZE] = {3.0, 5.0, 7.0, 14.0, 11.0};
  double y_sigma[SIZE] = {1.0, 1.0, 1.0, 1.0, 1.0};
  double a_expect = 0.5;
  double b_expect = 2.5;
  double a;
  double b;
  double correlation;
  double correlation_expect = 0.88388347648318432714;
  int size = SIZE;
  int cn = 0;

  MTK_PRINT_STATUS(cn,"Testing MtkLinearRegressionCalc");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, y, y_sigma, &a, &b, &correlation);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkLinearRegressionCalc(1)\n");
    error = MTK_TRUE;
  }

  if (SCm_CMP_NE_DBL(a, a_expect) ||
      SCm_CMP_NE_DBL(b, b_expect) ||
      SCm_CMP_NE_DBL(correlation, correlation_expect)) {
    printf("a = %20.20g (expected %20.20g)\n",
	   a, a_expect);
    printf("b = %20.20g (expected %20.20g)\n",
	   b, b_expect);
    printf("correlation = %20.20g (expected %20.20g)\n",
	   correlation, correlation_expect);
    printf("Unexpected result(test1).\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  y_sigma[3] = 0.7;
  a_expect = 0.22239502332815061503;
  b_expect = 2.777604976671848469;
  correlation_expect = 0.86881476063906093454;

  status = MtkLinearRegressionCalc(size, x, y, y_sigma, &a, &b, &correlation);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkLinearRegressionCalc(2)\n");
    error = MTK_TRUE;
  }

  if (SCm_CMP_NE_DBL(a, a_expect) ||
      SCm_CMP_NE_DBL(b, b_expect) ||
      SCm_CMP_NE_DBL(correlation, correlation_expect)) {
    printf("a = %20.20g (expected %20.20g)\n",
	   a, a_expect);
    printf("b = %20.20g (expected %20.20g)\n",
	   b, b_expect);
    printf("correlation = %20.20g (expected %20.20g)\n",
	   correlation, correlation_expect);
    printf("Unexpected result(test2).\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Test 3: Divide by zero                                             */
  /* ------------------------------------------------------------------ */
  
  {
    double constant_x[SIZE] = {1.0, 1.0, 1.0, 1.0, 1.0};

    status = MtkLinearRegressionCalc(size, constant_x, y, y_sigma, &a, &b,
				     &correlation);
    if (status != MTK_DIV_BY_ZERO) {
      printf("Unexpected status(test3)\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Test 4: Divide by zero                                             */
  /* ------------------------------------------------------------------ */
  
  {
    double constant_y[SIZE] = {1.0, 1.0, 1.0, 1.0, 1.0};

    status = MtkLinearRegressionCalc(size, x, constant_y, y_sigma, &a, &b,
				     &correlation);
    if (status != MTK_DIV_BY_ZERO) {
      printf("Unexpected status(test4)\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Size < 1                                           */
  /* ------------------------------------------------------------------ */

  size = 0;
  status = MtkLinearRegressionCalc(size, x, y, y_sigma, &a, &b, &correlation);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  size = 1;

  /* ------------------------------------------------------------------ */
  /* Argument check: X = NULL                                           */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, NULL, y, y_sigma, &a, &b, &correlation);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(2)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Y = NULL                                           */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, NULL, y_sigma, &a, &b, &correlation);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(3)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Y_Sigma = NULL                                     */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, y, NULL, &a, &b, &correlation);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(4)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: A = NULL                                           */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, y, y_sigma, NULL, &b, &correlation);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(5)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: B = NULL                                           */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, y, y_sigma, &a, NULL, &correlation);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(6)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Correlation = NULL                                 */
  /* ------------------------------------------------------------------ */

  status = MtkLinearRegressionCalc(size, x, y, y_sigma, &a, &b, NULL);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(7)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Y_Sigma[x] <= 0.0                                  */
  /* ------------------------------------------------------------------ */

  { 
    double y_sigma_bad[SIZE] = {1.0, 1.0, 1.0, -0.01, 1.0};

    size = SIZE;
    status = MtkLinearRegressionCalc(size, x, y, y_sigma_bad, &a, &b, &correlation);
    if (status != MTK_OUTBOUNDS) {
      printf("Unexpected status(8)\n");
      error = MTK_TRUE;
    }
  }

  { 
    double y_sigma_bad[SIZE] = {1.0, 1.0, 1.0, 0.0, 1.0};

    size = SIZE;
    status = MtkLinearRegressionCalc(size, x, y, y_sigma_bad, &a, &b, &correlation);
    if (status != MTK_OUTBOUNDS) {
      printf("Unexpected status(9)\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Report test result.                                                */
  /* ------------------------------------------------------------------ */
      
  if (error) {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  } else {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  }
}
