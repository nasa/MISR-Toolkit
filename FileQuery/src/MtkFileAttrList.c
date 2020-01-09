/*===========================================================================
=                                                                           =
=                             MtkFileAttrList                               =
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
#include <stdlib.h>
#include <string.h>

/** \brief Get a list of file attributes
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get a list of file attributes from the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileAttrList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &num_attrs, &attrlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a attrlist
 */

MTKt_status MtkFileAttrList(
  const char *filename,    /**< [IN] File name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileAttrListNC(filename, num_attrs, attrlist); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileAttrListHDF(filename, num_attrs, attrlist); // try HDF
}

MTKt_status MtkFileAttrListNC(
  const char *filename,    /**< [IN] File name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file */
  int ncid = 0;
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read grid attribute */
  status = MtkFileAttrListNcid(ncid, num_attrs, attrlist);
  MTK_ERR_COND_JUMP(status);

  /* Close file */
  {
    int nc_status = nc_close(ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_CLOSE_FAILED);
  }
  ncid = 0;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (ncid != 0) nc_close(ncid);
  return status_code;
}

MTKt_status MtkFileAttrListHDF(
  const char *filename,    /**< [IN] File name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdf_status;
  int32 sd_id = FAIL;

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_status = sd_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read attrribute list. */
  status = MtkFileAttrListFid(sd_id, num_attrs, attrlist);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF File */
  hdf_status = SDend(sd_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sd_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sd_id != FAIL)
    SDend(sd_id);
  return status_code;
}

/** \brief Version of MtkFileAttrList that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileAttrListFid(
  int32 sd_id,             /**< [IN] HDF SD file identifier */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  intn hdf_status;
  int32 num_datasets;
  int32 num_global_attrs = 0;
  char attr_name[H4_MAX_NC_NAME];
  int32 attr_index = 0;
  int32 hdf_datatype;
  int32 count;
  char **attrlist_tmp = NULL;

  if (num_attrs == NULL || attrlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Get number of gloabl attributes */
  hdf_status = SDfileinfo(sd_id, &num_datasets, &num_global_attrs);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDFILEINFO_FAILED);

  /* Temp attribute list */
  attrlist_tmp = (char**)calloc(num_global_attrs,sizeof(char**));
  if (attrlist_tmp == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  for (attr_index = 0; attr_index < num_global_attrs; ++attr_index)
  {
    /* Get attribute information */
    hdf_status = SDattrinfo(sd_id, attr_index, attr_name, &hdf_datatype, &count);
    if (hdf_status == FAIL)
      MTK_ERR_CODE_JUMP(MTK_HDF_SDATTRINFO_FAILED);

    attrlist_tmp[attr_index] = (char*)malloc((strlen(attr_name) + 1) * sizeof(char));
    if (attrlist_tmp[attr_index] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

    strcpy(attrlist_tmp[attr_index],attr_name);
  }

  *attrlist = attrlist_tmp;
  *num_attrs = num_global_attrs;

  return MTK_SUCCESS;

ERROR_HANDLE:

  MtkStringListFree(num_global_attrs, &attrlist_tmp);

  return status_code;
}

MTKt_status MtkFileAttrListNcid(
  int ncid,               /**< [IN] netCDF File ID */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  int32 num_global_attrs = 0;
  int32 attr_index = 0;
  char **attrlist_tmp = NULL;

  if (num_attrs == NULL || attrlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  {
    int nc_status = nc_inq_varnatts(ncid, NC_GLOBAL, &num_global_attrs);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  /* Temp attribute list */
  attrlist_tmp = (char**)calloc(num_global_attrs,sizeof(char**));
  if (attrlist_tmp == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  for (attr_index = 0; attr_index < num_global_attrs; ++attr_index)
  {
    /* Get attribute information */
    char attr_name[NC_MAX_NAME+1];
    int nc_status = nc_inq_attname(ncid, NC_GLOBAL, attr_index, attr_name);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    attrlist_tmp[attr_index] = (char*)malloc((strlen(attr_name) + 1) * sizeof(char));
    if (attrlist_tmp[attr_index] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

    strcpy(attrlist_tmp[attr_index],attr_name);
  }

  *attrlist = attrlist_tmp;
  *num_attrs = num_global_attrs;

  return MTK_SUCCESS;

ERROR_HANDLE:

  MtkStringListFree(num_global_attrs, &attrlist_tmp);

  return status_code;
}
