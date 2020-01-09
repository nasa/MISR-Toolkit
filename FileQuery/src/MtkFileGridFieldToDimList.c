/*===========================================================================
=                                                                           =
=                        MtkFileGridFieldToDimList                          =
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
#include <stdlib.h>

/** \brief Read dimension list of a particular field
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of dimensions for the field <tt> Blue Radiance/RDQI </tt> in the grid \c BlueBand in the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileGridFieldToDimList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand",
 *                "Blue Radiance/RDQI", &dimcnt, &dimlist, &dimsize);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \c dimlist and free() to free the memory used by \c dimsize  
 */

MTKt_status MtkFileGridFieldToDimList(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;       /* Return status */

  status = MtkFileGridFieldToDimListNC(filename, gridname, fieldname, dimcnt, dimlist, dimsize); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileGridFieldToDimListHDF(filename, gridname, fieldname, dimcnt, dimlist, dimsize); // try HDF
}

MTKt_status MtkFileGridFieldToDimListNC(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;
  MTKt_status status_code;
  int ncid = 0;
 
  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file */
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read dimension list. */
  status = MtkFileGridFieldToDimListNcid(ncid, gridname, fieldname, dimcnt, dimlist, dimsize);
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

MTKt_status MtkFileGridFieldToDimListHDF(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;		/* Return status. */
  MTKt_status status_code;      /* Return status of this function. */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 Fid = FAIL;             /* HDF-EOS File ID */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF file for reading */
  hdfstatus = Fid = GDopen((char*)filename,DFACC_READ);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read dimension list. */
  status = MtkFileGridFieldToDimListFid(Fid, gridname, fieldname, 
					dimcnt, dimlist, dimsize);
  MTK_ERR_COND_JUMP(status);
  
  /* Close HDF file. */
  GDclose(Fid);
  
  return MTK_SUCCESS;

ERROR_HANDLE:
  GDclose(Fid);

  return status_code;
}


/** \brief Version of MtkFileGridFieldToDimList that takes an HDF-EOS file ID rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridFieldToDimListFid(
  int32 Fid,            /**< [IN] HDF-EOS File ID */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;		/* Return status. */
  MTKt_status status_code;      /* Return status of this function. */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 Gid = FAIL;             /* HDF-EOS Grid ID */
  int32 rank = 0;
  int32 dims[10];               /* Dimension sizes */
  int32 numbertype;
  char *list = NULL;            /* List of dimensions */
  int i;
  char *temp = NULL;
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  /* Check Arguments */
  if (dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (dimsize == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *dimlist = NULL; /* Set output to NULL to prevent freeing unallocated
                      memory in case of error. */
  *dimsize = NULL;

  if (gridname == NULL || fieldname == NULL ||
      dimcnt == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Attach to grid */
  hdfstatus = Gid = GDattach(Fid,(char*)gridname);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  list = (char*)malloc(5000 * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  /* Get field info */
  hdfstatus = GDfieldinfo(Gid, basefield, &rank, dims, &numbertype, list);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDFIELDINFO_FAILED);

  *dimlist = (char**)calloc(rank,sizeof(char*));
  if (*dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
  *dimsize = (int*)calloc(rank,sizeof(int));
  if (*dimsize == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    
  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    if ((strncmp(temp, "SOMBlockDim", sizeof(*temp))) != 0 &&
	(strncmp(temp, "XDim", sizeof(*temp))) != 0 &&
	(strncmp(temp, "YDim", sizeof(*temp))) != 0) {
      (*dimlist)[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
      if ((*dimlist)[i] == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
      strcpy((*dimlist)[i],temp);
      ++i;
    }
    temp = strtok(NULL,",");
  }

  /* Dimension count with out implied dimensions */
  *dimcnt = i;

  if (*dimcnt > 0) {

    /* Get dimension size info */
    for (i = 0; i < *dimcnt; ++i) {
      (*dimsize)[i] = GDdiminfo(Gid, (*dimlist)[i]);
    }

    /* Resize lists to account for removing dimensions */
    *dimlist = (char **)realloc((void *)*dimlist, *dimcnt * sizeof(char*));
    if (*dimlist == NULL)
      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
    *dimsize = (int *)realloc((void *)*dimsize, *dimcnt * sizeof(int));
    if (*dimsize == NULL)
      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
  } else {
    MtkStringListFree(*dimcnt, dimlist);
    free(*dimsize);
    *dimsize = NULL;
  }

  free(basefield);
  free(extradims);
  free(list);
  GDdetach(Gid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (dimlist != NULL && *dimlist != NULL)
    MtkStringListFree(*dimcnt, dimlist);

  if (dimsize != NULL)
    {
      free(*dimsize);
      *dimsize = NULL;
    }

  if (dimcnt != NULL)
    *dimcnt = -1;

  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);

  free(list);
  GDdetach(Gid);

  return status_code;
}

MTKt_status MtkFileGridFieldToDimListNcid(
  int ncid,            /**< [IN] netCDF file identifier */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;		/* Return status. */
  MTKt_status status_code;      /* Return status of this function. */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  int *dimids = NULL;

  /* Check Arguments */
  if (dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (dimsize == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *dimlist = NULL; /* Set output to NULL to prevent freeing unallocated
                      memory in case of error. */
  *dimsize = NULL;

  if (gridname == NULL || fieldname == NULL ||
      dimcnt == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  int group_id;
  {
    int nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  MTKt_ncvarid var;
  status = MtkNCVarId(group_id, basefield, &var);
  MTK_ERR_COND_JUMP(status);

  int ndims;
  {
    int nc_status = nc_inq_varndims(var.gid, var.varid, &ndims);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  dimids = calloc(ndims, sizeof(int));
  {
    int nc_status = nc_inq_var(var.gid, var.varid, NULL, NULL, NULL, dimids, NULL);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  *dimlist = calloc(ndims, sizeof(char *));
  *dimsize = calloc(ndims, sizeof(int));
  *dimcnt = 0;

  for (size_t i = 0 ; i < ndims ; i++) {
    char name[NC_MAX_NAME+1];  // add 1 for string terminator
    int this_dimid = dimids[i];
    size_t size;
    int nc_status = nc_inq_dim(group_id, this_dimid, name, &size);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    if (0 == strcmp(name, "X_Dim")) continue; // skip X_Dim
    if (0 == strcmp(name, "Y_Dim")) continue; // skip Y_Dim

    char *this_name = calloc(sizeof(name), sizeof(char));
    strcpy(this_name, name);
    (*dimlist)[(*dimcnt)] = this_name;
    (*dimsize)[(*dimcnt)] = size;
    (*dimcnt)++;
  }

  if (*dimcnt == 0) {
    MtkStringListFree(*dimcnt, dimlist);
    free(*dimsize);
    *dimsize = NULL;
  }

  free(dimids);
  free(basefield);
  free(extradims);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (dimids != NULL) free(dimids);
  if (dimlist != NULL && *dimlist != NULL)
    MtkStringListFree(ndims, dimlist);

  if (dimsize != NULL)
    {
      free(*dimsize);
      *dimsize = NULL;
    }

  if (dimcnt != NULL)
    *dimcnt = -1;

  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);

  return status_code;
}
