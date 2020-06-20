/*===========================================================================
=                                                                           =
=                               MtkFileType                                 =
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

/** \brief Determine MISR product file type
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we determine the file type of the file 
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileType("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &filetype);
 *  \endcode
 */

MTKt_status MtkFileType(
  const char *filename,    /**< [IN] File name */
  MTKt_FileType *filetype  /**< [OUT] File type */ )
{
  MTKt_status status;      	/* Return status */

  status = MtkFileTypeNC(filename, filetype); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileTypeHDF(filename, filetype); // try HDF
}

MTKt_status MtkFileTypeNC(
  const char *filename,    /**< [IN] File name */
  MTKt_FileType *filetype  /**< [OUT] File type */ )
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

  /* Determine MISR product file type */
  status = MtkFileTypeNcid(ncid, filetype);
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

MTKt_status MtkFileTypeHDF(
  const char *filename,    /**< [IN] File name */
  MTKt_FileType *filetype  /**< [OUT] File type */ )
{
  MTKt_status status;
  MTKt_status status_code;
  intn hdf_status;
  int32 fid = FAIL;
  
  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fid = GDopen((char *)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Determine MISR product file type */
  status = MtkFileTypeFid(fid, filetype);
  MTK_ERR_COND_JUMP(status);

  hdf_status = GDclose(fid);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkFileType that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileTypeFid(
  int32 Fid,               /**< [IN] HDF-EOS file identifier */
  MTKt_FileType *filetype  /**< [OUT] File type */ )
{
  MTKt_status status;
  MTKt_status status_code;
  intn hdf_status;
  char *lgid;
  char *fn_start;
  int num_grids;
  char **gridlist = NULL;
  int32 gridID;
  int32 projcode;
  int32 zonecode;
  int32 spherecode;
  float64 projparm[13];
  int32 sid;  			/* HDF SD identifier (required by MtkFileLGID) */
  int32 HDFfid; 		/* HDF file identifier (not used) */
  
  if (filetype == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* It's possible for a file to have both conventional and stacked-block
     format in the same file but this is not used for MISR so only the
     first grid in the file is checked. */

  status = MtkFileToGridListFid(Fid,&num_grids,&gridlist);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);
 
  hdf_status = gridID = GDattach(Fid,gridlist[0]);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  hdf_status = GDprojinfo(gridID,&projcode,&zonecode,&spherecode,projparm);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDPROJINFO_FAILED);

  hdf_status = GDdetach(gridID);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  MtkStringListFree(num_grids, &gridlist);

  if (projparm[11] == 0)
  {
    *filetype = MTK_CONVENTIONAL;

    /* Check if MISR_HR product */
    hdf_status = EHidinfo(Fid, &HDFfid, &sid);
    if (hdf_status == FAIL)
      return MTK_SUCCESS;

    /* Get Local Granual ID */
    status = MtkFileLGIDFid(sid,&lgid);
    if (status != MTK_SUCCESS)
      return MTK_SUCCESS;

    fn_start = strstr(lgid, "MISR_HR_");
    if (fn_start == NULL)
      return MTK_SUCCESS;

    fn_start += 8; /* Skip MISR_HR_ */

    /* Determine file type */
    if (strncmp(fn_start,"BRF",3) == 0)
      *filetype = MTK_HR_BRF;
    else if (strncmp(fn_start,"RPV",3) == 0)
      *filetype = MTK_HR_RPV;
    else if (strncmp(fn_start,"TIP",3) == 0)
      *filetype = MTK_HR_TIP;
    else
      *filetype = MTK_UNKNOWN;

    free(lgid);

    return MTK_SUCCESS;
  }

  /* Determine type of stacked-block product */

  /* Get the HDF SD identifier. */
  hdf_status = EHidinfo(Fid, &HDFfid, &sid);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_EHIDINFO_FAILED);

  /* Get Local Granual ID */
  status = MtkFileLGIDFid(sid,&lgid);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(status);

  fn_start = strstr(lgid, "MISR_AM1_");
  if (fn_start == NULL)
    MTK_ERR_CODE_JUMP(MTK_FAILURE);

  fn_start += 9; /* Skip MISR_AM1_ */

  /* Determine file type */
  if (strncmp(fn_start,"AGP",3) == 0)
    *filetype = MTK_AGP;
  else if (strncmp(fn_start,"GP_GMP",6) == 0)
    *filetype = MTK_GP_GMP;
  else if (strncmp(fn_start,"GRP_RCCM",8) == 0)
    *filetype = MTK_GRP_RCCM;
  else if (strncmp(fn_start,"GRP_ELLIPSOID_GM",16) == 0)
    *filetype = MTK_GRP_ELLIPSOID_GM;
  else if (strncmp(fn_start,"GRP_TERRAIN_GM",14) == 0)
    *filetype = MTK_GRP_TERRAIN_GM;
  else if (strncmp(fn_start,"GRP_ELLIPSOID_LM",16) == 0)
    *filetype = MTK_GRP_ELLIPSOID_LM;
  else if (strncmp(fn_start,"GRP_TERRAIN_LM",14) == 0)
    *filetype = MTK_GRP_TERRAIN_LM;
  else if (strncmp(fn_start,"AS_AEROSOL",10) == 0)
    *filetype = MTK_AS_AEROSOL;
  else if (strncmp(fn_start,"AS_LAND",7) == 0)
    *filetype = MTK_AS_LAND;
  else if (strncmp(fn_start,"TC_ALBEDO",9) == 0)
    *filetype = MTK_TC_ALBEDO;
  else if (strncmp(fn_start,"TC_CLASSIFIERS",14) == 0)
    *filetype = MTK_TC_CLASSIFIERS;
  else if (strncmp(fn_start,"TC_STEREO",9) == 0)
    *filetype = MTK_TC_STEREO;
  else if (strncmp(fn_start,"PGRP_ELLIPSOID_GM",17) == 0)
    *filetype = MTK_GRP_ELLIPSOID_GM;
  else if (strncmp(fn_start,"PGRP_TERRAIN_GM",15) == 0)
    *filetype = MTK_GRP_TERRAIN_GM;
  else if (strncmp(fn_start,"PP",2) == 0)
    *filetype = MTK_PP;
  else if (strncmp(fn_start,"TC_CLOUD",8) == 0)
    *filetype = MTK_TC_CLOUD;
  else if (strncmp(fn_start,"CMV_T",5) == 0)
    *filetype = MTK_CMV_NRT;
  else
    *filetype = MTK_UNKNOWN;

  free(lgid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (gridlist != NULL)
    MtkStringListFree(num_grids, &gridlist);

  return status_code;

}

MTKt_status MtkFileTypeNcid(
  int ncid,               /**< [IN] netCDF file identifier */
  MTKt_FileType *filetype  /**< [OUT] File type */ )
{
  MTKt_status status;
  MTKt_status status_code;
  
  if (filetype == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Get Local Granual ID */

  char *lgid;
  status = MtkFileLGIDNcid(ncid,&lgid);
  if (status != MTK_SUCCESS) MTK_ERR_CODE_JUMP(status);

  char *fn_start = strstr(lgid, "MISR_AM1_");
  if (fn_start == NULL)
    MTK_ERR_CODE_JUMP(MTK_FAILURE);

  fn_start += 9; /* Skip MISR_AM1_ */

  /* Determine file type */
  if (strncmp(fn_start,"AS_AEROSOL",10) == 0)
    *filetype = MTK_AS_AEROSOL;
  else if (strncmp(fn_start,"AS_LAND",7) == 0)
    *filetype = MTK_AS_LAND;
  else
    *filetype = MTK_UNKNOWN;

  free(lgid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;

}
