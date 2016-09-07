/*===========================================================================
=                                                                           =
=                            MtkCoreMetaDataFree                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdlib.h>

/** \brief Free core metadata
 *
 *  \return MTK_SUCCESS
 *
 *  \par Example:
 *  In this example, we have a MtkCoreMetaData structure that was previously allocated and we wish to free it.
 *
 *  \code
 *  status = MtkCoreMetaDataFree(&metadata);
 *  \endcode
 */

MTKt_status MtkCoreMetaDataFree(
  MtkCoreMetaData *metadata /**< [IN,OUT] Core metadata */ )
{
  int i;

  if (metadata == NULL || metadata->data.s == NULL)
    return MTK_SUCCESS;

  switch (metadata->datatype)
  {
    case MTKMETA_CHAR :
      for (i = 0; i < metadata->num_values; ++i)
        free(metadata->data.s[i]);
      free(metadata->data.s);
      break;
    case MTKMETA_INT : free(metadata->data.i);
      break;

    case MTKMETA_DOUBLE : free(metadata->data.d);
      break;
  }

  metadata->data.s = NULL;
  metadata->dataptr = NULL;
  metadata->num_values = 0;

  return MTK_SUCCESS;
}
