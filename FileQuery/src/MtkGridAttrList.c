/*===========================================================================
=                                                                           =
=                             MtkGridAttrList                               =
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
#include <mfhdf.h>
#include <HdfEosDef.h>
#include <stdlib.h>
#include <string.h>

/** \brief Get a list of grid attributes
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get a list of grid attributes in the \c RedBand grid from the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkGridAttrList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "RedBand", &num_attrs, &attrlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a attrlist
 */

MTKt_status MtkGridAttrList(
  const char *filename,    /**< [IN] File name */
  const char *gridname,    /**< [IN] Grid name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  int32 hdfstatus;         /* HDF-EOS return status */
  int32 fid = FAIL;	   /* HDF-EOS File id */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Get list of grid attributes. */
  status = MtkGridAttrListFid(fid, gridname, num_attrs, attrlist);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */
  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fid != FAIL)
    GDclose(fid);

  return status_code;
}

/** \brief Version of MtkGridAttrList that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkGridAttrListFid(
  int32 fid,               /**< [IN] HDF-EOS file identifier */
  const char *gridname,    /**< [IN] Grid name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  int32 hdfstatus;         /* HDF-EOS return status */
  int32 gid = FAIL;	   /* HDF-EOS Grid id */
  int32 nattrs = 0;        /* Number of attributes */
  int32 count;
  char **attrlist_tmp = NULL;
  char *list = NULL;
  char *temp = NULL;
  int i;

  if (gridname == NULL || 
      num_attrs == NULL || attrlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Attach to Grid */
  gid = GDattach(fid, (char*)gridname);
  if (gid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  /* Get buffer size */  
  hdfstatus = GDinqattrs(gid,NULL,&count);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQATTRS_FAILED);

  list = (char*)malloc((count + 1) * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Get grid attributes */
  hdfstatus = nattrs = GDinqattrs(gid,list,&count);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQATTRS_FAILED);

  /* Detach from grid */
  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  attrlist_tmp = (char**)calloc(nattrs,sizeof(char*));
  if (attrlist_tmp == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    
  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    attrlist_tmp[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
    if (attrlist_tmp[i] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
    strcpy(attrlist_tmp[i],temp);
    temp = strtok(NULL,",");
    ++i;
  }

  free(list);

  *attrlist = attrlist_tmp;
  *num_attrs = nattrs;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (attrlist_tmp != NULL)
    MtkStringListFree(nattrs, &attrlist_tmp);

  if (num_attrs != NULL)
    *num_attrs = -1;

  free(list);
  GDdetach(gid);

  return status_code;
}
