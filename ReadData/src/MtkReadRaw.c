/*===========================================================================
=                                                                           =
=                               MtkReadRaw                                  =
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
#include "MisrCoordQuery.h"
#include "MisrFileQuery.h"
#include "MisrError.h"
#include "MisrCache.h"
#include "MisrUtil.h"		/* NOTE: remove when below note is removed */
#include "misrproj.h"
#include <hdf.h>
#include <HdfEosDef.h>
#include <stdlib.h>
#include <string.h>

/** \brief Reads any native grid/field from a MISR product file without
 *         unpacking or unscaling.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Special Note:
 *  Typically this function is not called directly.  Instead use MtkReadData().
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

MTKt_status MtkReadRaw(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */

  status = MtkReadRawNC(filename, gridname, fieldname, region, databuf, mapinfo);  // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkReadRawHDF(filename, gridname, fieldname, region, databuf, mapinfo);  // try HDF
}

MTKt_status MtkReadRawHDF(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  int32 fid = FAIL;		/* HDF-EOS File id */
  intn hdfstatus;		/* HDF return status */

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file. */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read data. */
  status = MtkReadRawFid(fid, gridname, fieldname, region, databuf, mapinfo);
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

MTKt_status MtkReadRawNC(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  int ncid = 0;

  if (filename == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file. */
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read data. */
  status = MtkReadRawNcid(ncid, gridname, fieldname, region, databuf, mapinfo);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
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

/** \brief Version of MtkReadRaw that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
*/

MTKt_status MtkReadRawFid(
  int32 fid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
				/* Map info structure */
  MTKt_DataBuffer pathbuf = MTKT_DATABUFFER_INIT;
				/* Temporary buffer for projection parameters Path_number grid attr. */
  int path;			/* Path */
  int resolution;		/* Resolution */
  MTKt_DataType datatype;	/* Datatype */
  MTKt_Cache blockcache = MTKT_CACHE_INIT;
				/* Block cache */
  intn hdfstatus;		/* HDF return status */
  int32           HDFfid;  	/* HDF file identifier (not used) */
  int32   	    sid;        /* HDF SD identifier */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check for valid file/grid/field/dim  */
  status = MtkFileGridFieldCheckFid(fid, gridname, fieldname);
  MTK_ERR_COND_JUMP(status);

  /* Get the HDF SD identifier. */
  hdfstatus = EHidinfo(fid, &HDFfid, &sid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_EHIDINFO_FAILED);

  /* Determine path of this filename */
  if (strcmp("Projection_Parameters", gridname) == 0) {
    status = MtkGridAttrGetFid(fid, gridname, "Path_number", &pathbuf);
    MTK_ERR_COND_JUMP(status);
    path = pathbuf.data.i32[0][0];
  } else {
    status = MtkFileToPathFid(sid, &path);
    MTK_ERR_COND_JUMP(status);
  }

  /* Determine resolution from filename and gridname */
  status = MtkFileGridToResolutionFid(fid, gridname, &resolution);
  MTK_ERR_COND_JUMP(status);

  /* Determine datatype from filenname, gridname and fieldname */
  status =MtkFileGridFieldToDataTypeFid(fid, gridname, fieldname, &datatype);
  MTK_ERR_COND_JUMP(status);

  /* Snap region to som grid for this path */
  status = MtkSnapToGrid(path, resolution, region, &map);
  MTK_ERR_COND_JUMP(status);

  /* Allocate the data buffer */
  status = MtkDataBufferAllocate(map.nline, map.nsample, datatype, &buf);
  MTK_ERR_COND_JUMP(status);

  /* Inital block cache */
  status = MtkCacheInitFid(fid, gridname, fieldname, &blockcache);
  MTK_ERR_COND_JUMP(status);

  {
    int b;			/* Block */
    float l;			/* Line */
    float s;			/* Sample */
    double x;			/* Som x */
    double y;			/* Som y */
    char *dptr;			/* Data pointer to buffer */
    MTKt_MisrProjParam pp;      /* Projection parameters */
    /* (this could come from map, instead of MtkPathToProjParam call) */

    MtkPathToProjParam(map.path, map.resolution, &pp);
    misr_init(pp.nblock, pp.nline, pp.nsample, pp.reloffset, pp.ulc, pp.lrc);

    dptr = (char *)buf.dataptr;
    for (x = map.som.ulc.x; x <= map.som.lrc.x; x += map.resolution) {
      for (y = map.som.ulc.y; y <= map.som.lrc.y; y += map.resolution) {
	misrfor(x, y, &b, &l, &s);
	MtkCachePixelGet(&blockcache, b, l, s, dptr);
	dptr += buf.datasize;
      }
    }
  }

  MtkCacheFree(&blockcache);

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  MtkCacheFree(&blockcache);
  MtkDataBufferFree(&buf);
  return status_code;
}

/** \brief Version of MtkReadRaw that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
*/

MTKt_status MtkReadRawNcid(
  int ncid,                 /**< [IN] netcdf file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
				/* Map info structure */
  MTKt_DataType datatype;	/* Datatype */
  MTKt_Cache blockcache = MTKT_CACHE_INIT;
				/* Block cache */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check for valid file/grid/field/dim  */
  status = MtkFileGridFieldCheckNcid(ncid, gridname, fieldname);
  MTK_ERR_COND_JUMP(status);

  int path;
  {
    int nc_status = nc_get_att_int(ncid, NC_GLOBAL, "Path_number", &path);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  int resolution;
  status = MtkFileGridToResolutionNcid(ncid, gridname, &resolution);
  MTK_ERR_COND_JUMP(status);

  status = MtkFileGridFieldToDataTypeNcid(ncid, gridname, fieldname, &datatype);
  MTK_ERR_COND_JUMP(status);

  /* Snap region to som grid for this path */
  status = MtkSnapToGrid(path, resolution, region, &map);
  MTK_ERR_COND_JUMP(status);

  /* Allocate the data buffer */
  status = MtkDataBufferAllocate(map.nline, map.nsample, datatype, &buf);
  MTK_ERR_COND_JUMP(status);

  /* Inital block cache */
  status = MtkCacheInitNcid(ncid, gridname, fieldname, &blockcache);
  MTK_ERR_COND_JUMP(status);

  {
    int b;			/* Block */
    float l;			/* Line */
    float s;			/* Sample */
    double x;			/* Som x */
    double y;			/* Som y */
    char *dptr;			/* Data pointer to buffer */
    MTKt_MisrProjParam pp;      /* Projection parameters */
    /* (this could come from map, instead of MtkPathToProjParam call) */

    MtkPathToProjParam(map.path, map.resolution, &pp);
    misr_init(pp.nblock, pp.nline, pp.nsample, pp.reloffset, pp.ulc, pp.lrc);

    dptr = (char *)buf.dataptr;
    for (x = map.som.ulc.x; x <= map.som.lrc.x; x += map.resolution) {
      for (y = map.som.ulc.y; y <= map.som.lrc.y; y += map.resolution) {
        misrfor(x, y, &b, &l, &s);
        status = MtkCachePixelGet(&blockcache, b, l, s, dptr);
        if (status != MTK_SUCCESS && status != MTK_OUTBOUNDS) {
          MTK_ERR_CODE_JUMP(status);
        }
        dptr += buf.datasize;
      }
    }
  }
  MtkCacheFree(&blockcache);

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  MtkCacheFree(&blockcache);
  MtkDataBufferFree(&buf);
  return status_code;
}
