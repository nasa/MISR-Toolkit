/*===========================================================================
=                                                                           =
=                                MtkVersion                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */

int main(int argc, char *argv[])
{
  printf("MISR Toolkit Version %s\n", MtkVersion());

  return 0;
}
