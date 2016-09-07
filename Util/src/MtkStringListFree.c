/*===========================================================================
=                                                                           =
=                            MtkStringListFree                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include "MisrError.h"
#include <stdlib.h>

/** \brief Free string list
 *
 *  \return MTK_SUCCESS
 *
 *  \par Example:
 *  In this example, we have a string list and a string size such as \c fieldlist and \c fieldcnt that was previously allocated with MtkFileGridToFieldList() and we wish to free it.
 *
 *  \code
 *  status = MtkStringListFree(fieldcnt, &fieldlist);
 *  \endcode
 */

MTKt_status MtkStringListFree(
  int strcnt,      /**< [IN] String Count */
  char **strlist[] /**< [IN,OUT] String List */ )
{
  int i;

  if (strlist == NULL ||
      *strlist == NULL)
    return MTK_SUCCESS;

  for (i = 0; i < strcnt; ++i)
    free((*strlist)[i]);
  free(*strlist);
  *strlist = NULL;

  return MTK_SUCCESS;
}
