/*===========================================================================
=                                                                           =
=                               MtkVersion                                  =
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

/** \brief MisrToolkit Version
 *
 *  \return version
 *
 *  \par Example:
 *  In this example, we query MisrToolkit version.
 *
 *  \code
 *  printf("MisrToolkit Version = %s\n", MtkVersion());
 *  \endcode
 */

char *MtkVersion(void)
{
  return MTK_VERSION;
}
