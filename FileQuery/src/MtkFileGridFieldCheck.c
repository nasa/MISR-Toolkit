/*===========================================================================
=                                                                           =
=                         MtkFileGridFieldCheck                             =
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
#include <stdlib.h>
#include <mfhdf.h>
#include <HdfEosDef.h>
#include <stdio.h>

/** \brief Check if file/grid/field/dimension are valid
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we check the validity of \c MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf the grid \c SubregParamsLnd the field \c LandBRF and the dimensions \c [1][0] where [1] is the Band and [0] is Camera
 *
 *  \code
 *  char *error_mesg[] = MTK_ERR_DESC;
 *  status = MtkFileGridFieldCheck("MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf",
 *                "SubregParamsLnd", "LandBrf[1][0]");
 *  if (status != MTK_SUCCESS) {
 *      printf("%s\n",error_mesg[status]);
 *  }
 *  \endcode
 */

MTKt_status MtkFileGridFieldCheck(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname /**< [IN] Field name */ )
{
  MTKt_status status;       /* Return status */

  status = MtkFileGridFieldCheckNC(filename, gridname, fieldname); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileGridFieldCheckHDF(filename, gridname, fieldname); // try HDF
}

MTKt_status MtkFileGridFieldCheckNC(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname /**< [IN] Field name */ )
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

  /* Check if file/grid/field/dimension are valid */
  status = MtkFileGridFieldCheckNcid(ncid, gridname, fieldname);
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

MTKt_status MtkFileGridFieldCheckHDF(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname /**< [IN] Field name */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 Fid = FAIL;             /* HDF-EOS File ID */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF file for reading */
  hdfstatus = Fid = GDopen((char*)filename,DFACC_READ);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Check if file/grid/field/dimension are valid */
  status = MtkFileGridFieldCheckFid(Fid, gridname, fieldname);
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

/** \brief Version of MtkFileGridFieldCheck that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridFieldCheckFid(
  int32 Fid,            /**< [IN] HDF-EOS file identifier */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname /**< [IN] Field name */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int nfields;			/* Number fields */
  char **fieldlist = NULL;	/* Field list */
  int i;			/* Field index */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */
  char **dimlist = NULL;	/* Dimension name list */
  int *dimsize = NULL;		/* Dimension size list */
  int dimcnt;			/* Number of dimensions */

  if (gridname == NULL || fieldname == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Parse fieldname into basefield number of dimensions and size of each dimension */
  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD);

  /* Check if gridname and fieldname are valid (includes derived fields) */
  status = MtkFileGridToFieldListFid(Fid, gridname, &nfields, &fieldlist);

  switch (status) {
  case MTK_HDFEOS_GDOPEN_FAILED: 
    MTK_ERR_CODE_JUMP(MTK_INVALID_FILE);
    break;

  case MTK_HDFEOS_GDATTACH_FAILED:
    MTK_ERR_CODE_JUMP(MTK_INVALID_GRID);
    break;

  case MTK_SUCCESS:
    for (i = 0; i < nfields; i++) {
      if (strncmp(basefield, fieldlist[i], strlen(fieldlist[i])) == 0)
	break;
    }
    if (i >= nfields) {
      MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD);
    }
    break;

  default:
    MTK_ERR_CODE_JUMP(status);
  }

  /* Check if number of dimensions and bounds of each dimension are valid */
  status = MtkFileGridFieldToDimListFid(Fid, gridname, fieldname,
					&dimcnt, &dimlist, &dimsize);
  MTK_ERR_COND_JUMP(status);

  if (nextradims < dimcnt) {

    MTK_ERR_CODE_JUMP(MTK_MISSING_FIELD_DIMENSION);

  } else if (nextradims > dimcnt) {

    MTK_ERR_CODE_JUMP(MTK_EXTRA_FIELD_DIMENSION);

  }
  for (i = 0; i < nextradims; i++) {
    if ( extradims[i] < 0 || extradims[i] >= dimsize[i] ) {
      MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD_DIMENSION);
    }
  }

  free(basefield);
  free(extradims);
  free(dimsize);
  MtkStringListFree(nfields, &fieldlist);
  MtkStringListFree(dimcnt, &dimlist);
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fieldlist != NULL)MtkStringListFree(nfields, &fieldlist);
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  if (dimlist != NULL) MtkStringListFree(dimcnt, &dimlist);
  if (dimsize != NULL) free(dimsize);
  return status_code;
}

MTKt_status MtkFileGridFieldCheckNcid(
  int ncid,            /**< [IN] netCDF file identifier */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname /**< [IN] Field name */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int nfields;			/* Number fields */
  char **fieldlist = NULL;	/* Field list */
  int i;			/* Field index */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */
  char **dimlist = NULL;	/* Dimension name list */
  int *dimsize = NULL;		/* Dimension size list */
  int dimcnt;			/* Number of dimensions */

  if (gridname == NULL || fieldname == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Parse fieldname into basefield number of dimensions and size of each dimension */
  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD);

  /* Check if gridname and fieldname are valid (includes derived fields) */
  status = MtkFileGridToFieldListNcid(ncid, gridname, &nfields, &fieldlist);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(MTK_INVALID_FILE);

  for (i = 0; i < nfields; i++) {
    if (strncmp(basefield, fieldlist[i], strlen(fieldlist[i])) == 0)
      break;
  }
  if (i >= nfields) {
    MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD);
  }

  /* Check if number of dimensions and bounds of each dimension are valid */
  status = MtkFileGridFieldToDimListNcid(ncid, gridname, fieldname,
                                         &dimcnt, &dimlist, &dimsize);
  MTK_ERR_COND_JUMP(status);

  if (nextradims < dimcnt) {

    MTK_ERR_CODE_JUMP(MTK_MISSING_FIELD_DIMENSION);

  } else if (nextradims > dimcnt) {

    MTK_ERR_CODE_JUMP(MTK_EXTRA_FIELD_DIMENSION);

  }
  for (i = 0; i < nextradims; i++) {
    if ( extradims[i] < 0 || extradims[i] >= dimsize[i] ) {
      MTK_ERR_CODE_JUMP(MTK_INVALID_FIELD_DIMENSION);
    }
  }

  free(basefield);
  free(extradims);
  free(dimsize);
  MtkStringListFree(nfields, &fieldlist);
  MtkStringListFree(dimcnt, &dimlist);
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fieldlist != NULL)MtkStringListFree(nfields, &fieldlist);
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  if (dimlist != NULL) MtkStringListFree(dimcnt, &dimlist);
  if (dimsize != NULL) free(dimsize);
  return status_code;
}
