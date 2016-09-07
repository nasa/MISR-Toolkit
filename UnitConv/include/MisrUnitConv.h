/*===========================================================================
=                                                                           =
=                              MisrUnitConv                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRUNITCONV_H
#define MISRUNITCONV_H

#include <MisrError.h>


MTKt_status MtkDmsToDd( double dms,
			double *dd );

MTKt_status MtkDdToDms( double dd,
			double *dms );

MTKt_status MtkDdToRad( double dd,
			double *rad );

MTKt_status MtkRadToDd( double rad,
			double *dd );

MTKt_status MtkDmsToRad( double dms,
			 double *rad );

MTKt_status MtkRadToDms( double rad,
			 double *dms );

MTKt_status MtkDmsToDegMinSec( double dms,
			       int *deg,
			       int *min,
			       double *sec );

MTKt_status MtkDegMinSecToDms( int deg,
			       int min,
			       double sec,
			       double *dms );

MTKt_status MtkDdToDegMinSec( double dd,
			      int *deg,
			      int *min,
			      double *sec );

MTKt_status MtkDegMinSecToDd( int deg,
			      int min,
			      double sec,
			      double *dd );

MTKt_status MtkRadToDegMinSec( double rad,
			       int *deg,
			       int *min,
			       double *sec );

MTKt_status MtkDegMinSecToRad( int deg,
			       int min,
			       double sec,
			       double *rad );

#endif /* MISRUNITCONV_H */
