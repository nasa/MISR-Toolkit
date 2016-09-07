/*===========================================================================
=                                                                           =
=                             MtkMakeFilename                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/** \brief Given a base directory, product, camera, path, orbit, version
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  The \c camera parameter can be set to NULL if product is not by camera.
 *  The \c orbit parameter can be set to 0 if product is not by orbit.
 *  The \c basedir parameter can be set to "." for current or no directory specified.
 *
 *  \par Example:
 *  In this example, we create the filename \c misr_products/MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkMakeFilename("misr_products", "GRP_TERRAIN_GM", "DF", 161, 12115, "F03_0021", &filename);
 *  \endcode
 *
 *  \par
 *  In this example, we create the filename \c misr_products/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf
 *
 *  \code
 *  status = MtkMakeFilename("misr_products", "GP_GMP", NULL, 37, 14845, "F02_0009", &filename);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using free() to free the memory used by \c filename  
 */

MTKt_status MtkMakeFilename(
  const char *basedir, /**< [IN] Base Directory */
  const char *product, /**< [IN] Product */
  const char *camera,  /**< [IN] Camera */
  int path,            /**< [IN] Path */
  int orbit,           /**< [IN] Orbit */
  const char *version, /**< [IN] Version */
  char **filename      /**< [OUT] Filename */ )
{
  int status_code;
  int len;
  char *temp;
  int i;
  char *slash = "/";

  if (basedir == NULL || product == NULL || version == NULL ||
      filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  len = (int)strlen(basedir);

  if (len == 0 || strncmp(&(basedir[len-1]), slash, 1) == 0)
    slash = "";

  temp = (char*)malloc((len + 256) * sizeof(char));
  if (temp == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  if (camera == NULL || strlen(camera) == 0)
    if (orbit > 0) {
      sprintf(temp,"%s%sMISR_AM1_%s_P%03d_O%06d_%s.hdf",basedir,
	      slash,product,path,orbit,version);
    } else {
      sprintf(temp,"%s%sMISR_AM1_%s_P%03d_%s.hdf",basedir,
	      slash,product,path,version);
    }
  else
    sprintf(temp,"%s%sMISR_AM1_%s_P%03d_O%06d_%s_%s.hdf",basedir,
            slash,product,path,orbit,camera,version);

  for (i = (int)strlen(basedir); (unsigned)i < strlen(temp) - 4; ++i)
    temp[i] = toupper(temp[i]);

  *filename = temp;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  return status_code;
}
