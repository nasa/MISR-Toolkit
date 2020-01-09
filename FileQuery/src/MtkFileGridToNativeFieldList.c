/*===========================================================================
=                                                                           =
=                       MtkFileGridToNativeFieldList                        =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <hdf.h>
#include <HdfEosDef.h>
#include <string.h>

/** \brief Read list of native fields from file (excludes derived fields)
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of native fields (excludes derived fields) from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf and the grid \c BlueBand
 *
 *  \code
 *  status = MtkFileGridToNativeFieldList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand", &nfields, &fieldlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a fieldlist
 */

MTKt_status MtkFileGridToNativeFieldList(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileGridToNativeFieldListNC(filename, gridname, nfields, fieldlist); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileGridToNativeFieldListHDF(filename, gridname, nfields, fieldlist); // try HDF
}

MTKt_status MtkFileGridToNativeFieldListNC(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
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
  status = MtkFileGridToNativeFieldListNcid(ncid, gridname, nfields, fieldlist);
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

MTKt_status MtkFileGridToNativeFieldListHDF(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status;		/* Return status of called functions */
  MTKt_status status_code;	/* Return status of this function. */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 Fid = FAIL;             /* HDF-EOS File ID */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);  

  /* Open HDF file for reading */
  hdfstatus = Fid = GDopen((char*)filename,DFACC_READ);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read list of fields. */
  status = MtkFileGridToNativeFieldListFid(Fid, gridname, nfields, fieldlist);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = GDclose(Fid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  Fid = FAIL;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (Fid != FAIL) GDclose(Fid);

  return status_code;
}

/** \brief Version of MtkFileGridToNativeFieldList that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridToNativeFieldListFid(
  int32 Fid,            /**< [IN] HDF-EOS file identifier */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status_code;	/* Return status of this function. */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 Gid = FAIL;             /* HDF-EOS Grid ID */
  int32 num_fields = 0;         /* Number of fields */
  char *list = NULL;            /* List of fields */
  int i;
  char *temp = NULL;
  int32 str_buffer_size = 0;

  /* Check Arguments */
  if (fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *fieldlist = NULL; /* Set output to NULL to prevent freeing unallocated
                        memory in case of error. */

  if (gridname == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nfields == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Attach to grid */
  hdfstatus = Gid = GDattach(Fid,(char*)gridname);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  /* Query length of fields string */
  hdfstatus = GDnentries(Gid, HDFE_NENTDFLD, &str_buffer_size);
  if(hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDNENTRIES_FAILED);

  list = (char*)malloc((str_buffer_size + 1)  * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Get list of fields */
  hdfstatus = num_fields = GDinqfields(Gid,list,NULL,NULL);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQFIELDS_FAILED);

  *nfields = num_fields;
  *fieldlist = (char**)calloc(num_fields,sizeof(char*));
  if (*fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    
  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    (*fieldlist)[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
    if ((*fieldlist)[i] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
    strcpy((*fieldlist)[i],temp);
    temp = strtok(NULL,",");
    ++i;
  }

  free(list);

  hdfstatus = GDdetach(Gid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (fieldlist != NULL)
    MtkStringListFree(num_fields, fieldlist);

  if (nfields != NULL)
    *nfields = -1;

  free(list);
  GDdetach(Gid);

  return status_code;
}

MTKt_status MtkFileGridToNativeFieldListNcid(
  int ncid,               /**< [IN] netCDF File ID */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status_code;	/* Return status of this function. */
  int32 num_fields = 0;         /* Number of fields */
  int *varids = NULL;
  int *group_ids = NULL;

  /* Check Arguments */
  if (fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *fieldlist = NULL; /* Set output to NULL to prevent freeing unallocated
                        memory in case of error. */

  if (gridname == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nfields == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  int group_id;
  int nc_status = nc_inq_ncid(ncid, gridname, &group_id);
  if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

  int ngroups;
  nc_status = nc_inq_grps(group_id, &ngroups, NULL);
  if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

  ngroups++; // Add 1 for base group
  group_ids = (int *)calloc(ngroups, sizeof(int));
  group_ids[0] = group_id; // include base group for the grid

  if (ngroups > 1) {
    nc_status = nc_inq_grps(group_id, NULL, group_ids+1);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  for (int i = 0 ; i < ngroups ; i++) {
    int group_id = group_ids[i];

    char group_name[MAX_NC_NAME] = {0};  // empty string for base group
    if (i > 0) { // if this is not the base group
      nc_status = nc_inq_grpname(group_id, group_name);
      if (nc_status != NC_NOERR) {
        MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
      }
    }

    int nvars;
    nc_status = nc_inq_varids(group_id, &nvars, NULL);
    if (nc_status != NC_NOERR) {
      MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    }

    varids = (int *)calloc(nvars, sizeof(int));
    nc_status = nc_inq_varids(group_id, NULL, varids);
    if (nc_status != NC_NOERR) {
      MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    }

    int start = num_fields;
    num_fields += nvars;
    *nfields = num_fields;
    *fieldlist = (char**)realloc(*fieldlist, num_fields * sizeof(char*));
    if (*fieldlist == NULL) {
      MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    }
    
    for (int i = 0 ; i < nvars ; i++) {
      int ifield = i + start;
      int varid = varids[i];
      char temp[MAX_NC_NAME];

      int nc_status = nc_inq_varname(group_id, varid, temp);
      if (nc_status != NC_NOERR) {
        MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
      }

      (*fieldlist)[ifield] = (char*)calloc(strlen(group_name) + strlen(temp) + 2, sizeof(char));
      if ((*fieldlist)[ifield] == NULL) {
        MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
      }
      strcpy((*fieldlist)[ifield],group_name);
      if (strlen(group_name) > 0) {
        strcat((*fieldlist)[ifield], "/");
      }
      strcat((*fieldlist)[ifield],temp);
    }

    free(varids);
    varids = NULL;
  }

  free(group_ids);
  group_ids = NULL;
  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (fieldlist != NULL)
    MtkStringListFree(num_fields, fieldlist);

  if (nfields != NULL)
    *nfields = -1;

  if (varids != NULL) {
    free(varids);
  }

  if (group_ids != NULL) {
    free(group_ids);
  }

  return status_code;
}
