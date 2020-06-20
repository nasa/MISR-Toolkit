/*===========================================================================
=                                                                           =
=                               MtkReadData                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReadData.h"
#include "MisrFileQuery.h"
#include "MisrError.h"
#include <stdlib.h>
#include <string.h>
#include <hdf.h>
#include <HdfEosDef.h>
#include <netcdf.h>

/** \brief Reads any grid/field from any MISR product file and performs
 *         unpacking or unscaling. It also reads any MISR conventional
 *         product file.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read a region centered at lat 35.0, lon -115.0 with a lat extent of 110.0km and lon extent 110.0km.  We read from the field \c AveSceneElev in the grid \c Standard in the file \c MISR_AM1_AGP_P037_F01_24.hdf 
 *
 *  \code
 *   char *error_mesg[] = MTK_ERR_DESC;
 *   MTKt_Region region = MTKT_REGION_INIT;
 *   MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;
 *   MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
 *   status = MtkSetRegionByLatLonExtent(35.0, -115.0, 110.0, 110.0, "km", &region);
 *   status = MtkReadData("MISR_AM1_AGP_P037_F01_24.hdf", "Standard", "AveSceneElev", region, &databuf, &mapinfo);
 *   if (status != MTK_SUCCESS) {
 *       printf("%s\n",error_mesg[status]);
 *   }
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by databuf
 *
 *  \note
 *  The MISR Toolkit read functions always return a 2-D data plane buffer.  Some fields in the MISR data products
 *  are multi-dimensional.  In order to read one of these fields, the slice to read needs to be specified.
 *  A bracket notation on the fieldname is used for this purpose.  For example fieldname = "RetrAppMask[0][5]".
 *
 *  \note
 *  Additional dimensions can be determined by the routine MtkFileGridFieldToDimlist() or by the
 *  MISR Data Product Specification (DPS) Document.  The actually definition of the indices are not described in the
 *  MISR product files and thus not described by the MISR Toolkit.  These will have to be looked up in the
 *  MISR DPS.  All indices are 0-based.
 */

MTKt_status MtkReadData(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */

  status = MtkReadDataNC(filename, gridname, fieldname, region, databuf, mapinfo);  // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkReadDataHDF(filename, gridname, fieldname, region, databuf, mapinfo);  // try HDF
}

MTKt_status MtkReadDataHDF(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_status status;		/* Return status */
  int32 fid = FAIL;		/* HDF-EOS File id */
  intn hdfstatus;		/* HDF return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file. */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read data. */
  status = MtkReadDataFid(fid, gridname, fieldname, region, databuf, mapinfo);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

MTKt_status MtkReadDataNC(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_status status;		/* Return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file. */
  int ncid = 0;
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read data. */
  status = MtkReadDataNcid(ncid, gridname, fieldname, region, databuf, mapinfo);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  {
    int nc_status = nc_close(ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_CLOSE_FAILED);
  }
  ncid = 0;
  /* Close file. */

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (ncid != 0) nc_close(ncid);
  return status_code;
}

/** \brief Version of MtkReadData that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadDataFid(
  int32 fid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_status status;		/* Return status */
  MTKt_FileType filetype;	/* File type */

  /* Determine file type */
  status = MtkFileTypeFid(fid, &filetype);
  MTK_ERR_COND_JUMP(status);

  switch (filetype) {

  case MTK_AGP:
  case MTK_GP_GMP:
  case MTK_GRP_RCCM:
  case MTK_PP:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_GRP_ELLIPSOID_GM:
  case MTK_GRP_ELLIPSOID_LM:
  case MTK_GRP_TERRAIN_GM:
  case MTK_GRP_TERRAIN_LM:
    status = MtkReadL1B2Fid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_AS_AEROSOL:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_AS_LAND:
    status = MtkReadL2LandFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_TC_ALBEDO:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_TC_CLASSIFIERS:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_TC_STEREO:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_TC_CLOUD:
    status = MtkReadL2TCCloudFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_CMV_NRT:
    status = MtkReadRawFid(fid, gridname, fieldname, region,
               databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_HR_BRF:
  case MTK_HR_RPV:
  case MTK_HR_TIP:
  case MTK_CONVENTIONAL:
    status = MtkReadConvFid(fid, gridname, fieldname, region,
			    databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_UNKNOWN:
  default:
    return MTK_FILETYPE_NOT_SUPPORTED;
  }

  return MTK_SUCCESS;

 ERROR_HANDLE:

  return status_code;
}

/** \brief Version of MtkReadData that takes a netCDF file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadDataNcid(
  int ncid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_status status;		/* Return status */
  MTKt_FileType filetype;	/* File type */

  /* Determine file type */
  status = MtkFileTypeNcid(ncid, &filetype);
  MTK_ERR_COND_JUMP(status);

  switch (filetype) {

  case MTK_AS_AEROSOL:
    status = MtkReadRawNcid(ncid, gridname, fieldname, region, databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_AS_LAND:
    status = MtkReadL2LandNcid(ncid, gridname, fieldname, region, databuf, mapinfo);
    MTK_ERR_COND_JUMP(status);
    break;

  case MTK_UNKNOWN:
  default:
    return MTK_FILETYPE_NOT_SUPPORTED;
  }

  return MTK_SUCCESS;

 ERROR_HANDLE:

  return status_code;
}
