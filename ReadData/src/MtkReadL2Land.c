/*===========================================================================
=                                                                           =
=                              MtkReadL2Land                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

/* For strcasestr prototype in string.h on Linux64 */
#define _GNU_SOURCE

#include "MisrReadData.h"
#include "MisrFileQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <hdf.h>
#include <HdfEosDef.h>

/** \brief Reads, unpacks and unscales any L2 Land grid/field from a
 *         MISR L2 AS Land product file.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Special Note
 *  Typically this function is not called directly.  Instead use MtkReadData().
 */

MTKt_status MtkReadL2Land(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;       /* Return status */

  status = MtkReadL2LandNC(filename, gridname, fieldname, region, databuf, mapinfo); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkReadL2LandHDF(filename, gridname, fieldname, region, databuf, mapinfo); // try HDF
}

MTKt_status MtkReadL2LandNC(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
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

  /* Read data. */
  status = MtkReadL2LandNcid(ncid, gridname, fieldname, region, databuf, mapinfo);
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

MTKt_status MtkReadL2LandHDF(
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
  status = MtkReadL2LandFid(fid, gridname, fieldname, region, databuf, mapinfo);
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

/** \brief Version of MtkReadL2Land that takes and HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadL2LandFid(
  int32 fid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
                                /* Map info structure */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_DataBuffer rawbuf = MTKT_DATABUFFER_INIT;
				/* Raw data buffer structure */
  MTKt_DataBuffer min = MTKT_DATABUFFER_INIT;
				/* Minimum for unscaling */
  MTKt_DataBuffer max = MTKT_DATABUFFER_INIT;
				/* Maximum for unscaling */
  MTKt_DataBuffer offset = MTKT_DATABUFFER_INIT;
				/* Offset for unscaling */
  MTKt_DataBuffer scale = MTKT_DATABUFFER_INIT;
				/* Scale for unscaling */
  MTKt_DataBuffer fill = MTKT_DATABUFFER_INIT;
				/* Fill flag value */
  MTKt_DataBuffer overflow = MTKT_DATABUFFER_INIT;
				/* Overflow flag value */
  MTKt_DataBuffer underflow = MTKT_DATABUFFER_INIT;
				/* Underflow flag value */
  char *field;			/* Pointer to field to read */
  char fieldstr[MAXSTR];	/* Field string */
  char *fieldarr[2];		/* Field array */
  char *sp;			/* Pointer to string */
  char attr[MAXSTR];		/* Attribute name */
  int attrcnt = 0;		/* Attribute count */
  MTKt_boolean flag;		/* Flag field requested */
  MTKt_boolean raw;		/* Raw field requested */
  int l;			/* Line index */
  int s;			/* Sample index */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* ---------------------------------------------------------- */
  /* Parse fieldname to determine unpacking or unscaling method */
  /* ---------------------------------------------------------- */

  /* Make a working copy of fieldname */
  strncpy(fieldstr, fieldname, MAXSTR);
  fieldarr[0] = fieldstr;
  
  /* Separate raw/flag from base field */
  if ((sp = strchr(fieldstr, ' ')) != NULL) {
    *sp = '\0';
    fieldarr[1] = ++sp;
  } else if ((sp = strchr(fieldstr, '\0')) != NULL) {
    fieldarr[1] = sp;
  } else {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* ------------------------------------------------ */
  /* Determine if raw/native or flags is in fieldname */
  /* ------------------------------------------------ */

  if ((sp = strcasestr(fieldname, "raw")) != NULL ||
      (sp = strcasestr(fieldname, "native")) != NULL) {
    flag = MTK_FALSE;
    raw = MTK_TRUE;
    field = fieldarr[1];
  } else if ((sp = strcasestr(fieldname, "flag")) != NULL) {
    flag = MTK_TRUE;
    raw = MTK_FALSE;
    field = fieldarr[1];
  } else {
    flag = MTK_FALSE;
    raw = MTK_FALSE;
    field = fieldarr[0];
  }

  /* -------------------------------------------------------------------- */
  /* Read scale, offset, fill, underflow and overflow value, if available */
  /* -------------------------------------------------------------------- */

  status = MtkParseFieldname(field, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  if (raw == MTK_FALSE) {
    strncpy(attr, "Min ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &min);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Max ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &max);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Scale ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &scale);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Offset ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &offset);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Fill ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &fill);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Underflow ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &underflow);
    if (status == MTK_SUCCESS) attrcnt++;

    strncpy(attr, "Overflow ", MAXSTR);
    strncat(attr, basefield, strlen(basefield));
    status = MtkGridAttrGetFid(fid, gridname, attr, &overflow);
    if (status == MTK_SUCCESS) attrcnt++;
  }

  /* --------------------------------------------------------------- */
  /* Determine which unpacking or unscaling method to do, then do it */
  /* --------------------------------------------------------------- */

  /* -------------------------------------------------------------- */
  /* Read, unpack and unscale LandHDRF and LandBRF fields, only if  */
  /* raw or flag is not set. These two fields are a special case    */
  /* unpacking and unscaling alogrithm.  The raw field is shifted   */
  /* left two bits.                                                 */
  /* -------------------------------------------------------------- */

  if (attrcnt == 7 && flag == MTK_FALSE && raw == MTK_FALSE &&
      ((strncmp(basefield, "LandHDRF", MAXSTR) == 0 &&
	strncmp(basefield, "LandHDRFUnc", MAXSTR) != 0) ||
       strncmp(basefield, "LandBRF", MAXSTR) == 0)) {

    status = MtkReadRawFid(fid, gridname, field, region, &rawbuf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(rawbuf.nline, rawbuf.nsample, MTKe_float, 
                                   &buf);
    MTK_ERR_COND_JUMP(status);

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
        if (rawbuf.data.u16[l][s] == fill.data.u16[0][0]) {
	  buf.data.f[l][s] = (MTKt_float)fill.data.u16[0][0];
	} else if (rawbuf.data.u16[l][s] == underflow.data.u16[0][0]) {
          buf.data.f[l][s] = (MTKt_float)underflow.data.u16[0][0];
	} else if (rawbuf.data.u16[l][s] == overflow.data.u16[0][0]) {
	  buf.data.f[l][s] = (MTKt_float)overflow.data.u16[0][0];
	} else {
          buf.data.f[l][s] = (MTKt_float)((rawbuf.data.u16[l][s] >> 1) * 
                                          scale.data.f[0][0]) +
	                                  offset.data.f[0][0];
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read, unpack and unscale all fields that have scale and offset */
  /* offset values, only if raw or flag is not set.                 */
  /* -------------------------------------------------------------- */

  } else if (attrcnt == 7 && flag == MTK_FALSE && raw == MTK_FALSE) {

    status = MtkReadRawFid(fid, gridname, field, region, &rawbuf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(rawbuf.nline, rawbuf.nsample, MTKe_float, 
                                   &buf);
    MTK_ERR_COND_JUMP(status);

    switch(rawbuf.datatype) {
    case MTKe_uint8:
      for (l = 0; l < buf.nline; l++) {
	for (s = 0; s < buf.nsample; s++) {
	  if (rawbuf.data.u8[l][s] == fill.data.u8[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)fill.data.u8[0][0];
	  } else if (rawbuf.data.u8[l][s] == underflow.data.u8[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)underflow.data.u8[0][0];
	  } else if (rawbuf.data.u8[l][s] == overflow.data.u8[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)overflow.data.u8[0][0];
	  } else {
	    buf.data.f[l][s] = (MTKt_float)(rawbuf.data.u8[l][s] * 
			       scale.data.f[0][0]) + offset.data.f[0][0];
	  }
	}
      }
      break;
    case MTKe_uint16:
      for (l = 0; l < buf.nline; l++) {
	for (s = 0; s < buf.nsample; s++) {
	  if (rawbuf.data.u16[l][s] == fill.data.u16[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)fill.data.u16[0][0];
	  } else if (rawbuf.data.u16[l][s] == underflow.data.u16[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)underflow.data.u16[0][0];
	  } else if (rawbuf.data.u16[l][s] == overflow.data.u16[0][0]) {
	    buf.data.f[l][s] = (MTKt_float)overflow.data.u16[0][0];
	  } else {
	    buf.data.f[l][s] = (MTKt_float)(rawbuf.data.u16[l][s] * 
			       scale.data.f[0][0]) + offset.data.f[0][0];
	  }
	}
      }
      break;
    default:
      MTK_ERR_COND_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
      break;
    }

  /* -------------------------------------------------------------- */
  /* Read and only unpack LAIDelta1 and LAIDelta2, only if raw or   */
  /* flag is not set. The negative sign (flag) needs to be stripped */
  /* to produce the actual real value.		      		    */
  /* -------------------------------------------------------------- */

  } else if (attrcnt == 0 && flag == MTK_FALSE && raw == MTK_FALSE &&
	     (strncmp(basefield, "LAIDelta1", MAXSTR) == 0 ||
	      strncmp(basefield, "LAIDelta2", MAXSTR) == 0)) {

    status = MtkReadRawFid(fid, gridname, field, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill);
    MTK_ERR_COND_JUMP(status)

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
        if (buf.data.f[l][s] != fill.data.f[0][0]) {
          buf.data.f[l][s] = fabsf(buf.data.f[l][s]);
        } else {
          buf.data.f[l][s] = buf.data.f[l][s];
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read flag value for LandHDRF and LandBRF fields, only if flag  */
  /* is set to true.						    */
  /* -------------------------------------------------------------- */

  } else if (attrcnt == 7 && flag == MTK_TRUE && raw == MTK_FALSE &&
	     ((strncmp(basefield, "LandHDRF", MAXSTR) == 0 &&
	       strncmp(basefield, "LandHDRFUnc", MAXSTR) != 0) ||
	      strncmp(basefield, "LandBRF", MAXSTR)  == 0)) {

    status = MtkReadRawFid(fid, gridname, field, region, &rawbuf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(rawbuf.nline, rawbuf.nsample, MTKe_uint16, 
                                   &buf);
    MTK_ERR_COND_JUMP(status);

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
        if (rawbuf.data.u16[l][s] != fill.data.u16[0][0] &&
	    rawbuf.data.u16[l][s] != underflow.data.u16[0][0] &&
	    rawbuf.data.u16[l][s] != overflow.data.u16[0][0]) {
          buf.data.u16[l][s] = rawbuf.data.u16[l][s] & 0x0001;
        } else {
          buf.data.u16[l][s] = rawbuf.data.u16[l][s];
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read flag value for LAIDelta1 and LAIDelta2 fields, only if    */
  /* flag is set to true.					    */
  /* -------------------------------------------------------------- */

  } else if (attrcnt == 0 && flag == MTK_TRUE && raw == MTK_FALSE &&
	     (strncmp(basefield, "LAIDelta1", MAXSTR) == 0 ||
	      strncmp(basefield, "LAIDelta2", MAXSTR) == 0)) {

    status = MtkReadRawFid(fid, gridname, field, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill);
    MTK_ERR_COND_JUMP(status)

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
        if (buf.data.f[l][s] != fill.data.f[0][0]) {
          buf.data.f[l][s] = buf.data.f[l][s] < 0.0f ? -1.0f : 1.0f;
        } else {
          buf.data.f[l][s] = buf.data.f[l][s];
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read all other fields that don't require unpacking/ unscaling. */
  /* -------------------------------------------------------------- */

  } else if (flag == MTK_FALSE) {

    status = MtkReadRawFid(fid, gridname, field, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

  } else {
    MTK_ERR_CODE_JUMP(MTK_NOT_FOUND);
  }

  free(basefield);
  free(extradims);
  MtkDataBufferFree(&rawbuf);
  MtkDataBufferFree(&min);
  MtkDataBufferFree(&max);
  MtkDataBufferFree(&offset);
  MtkDataBufferFree(&scale);
  MtkDataBufferFree(&fill);
  MtkDataBufferFree(&overflow);
  MtkDataBufferFree(&underflow);

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&buf);
  MtkDataBufferFree(&rawbuf);
  MtkDataBufferFree(&min);
  MtkDataBufferFree(&max);
  MtkDataBufferFree(&offset);
  MtkDataBufferFree(&scale);
  MtkDataBufferFree(&fill);
  MtkDataBufferFree(&overflow);
  MtkDataBufferFree(&underflow);
  return status_code;
}

MTKt_status MtkReadL2LandNcid(
  int ncid,                /**< [IN] netCDF file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;     /* Map info structure */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;    /* Data buffer structure */
  MTKt_DataBuffer rawbuf = MTKT_DATABUFFER_INIT;  /* Raw data buffer structure */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* -------------------------------------------------------------------- */
  /* Determine field type                                                 */
  /* -------------------------------------------------------------------- */

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  MTKt_ncvarid var;
  int scaled = 0;
  nc_type nc_datatype;

  {
    int nc_status;
    int group_id;
    nc_status = nc_inq_grp_ncid(ncid, gridname, &group_id);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    status = MtkNCVarId(group_id, basefield, &var);
    MTK_ERR_COND_JUMP(status);

    nc_status = nc_inq_att(var.gid, var.varid, "scale_factor", NULL, NULL);
    if (nc_status == NC_NOERR) scaled = 1;
    nc_status = nc_inq_vartype(var.gid, var.varid, &nc_datatype);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
  }

  /* -------------------------------------------------------------- */
  /* Read and unscale fields of type unsigned short.                */
  /* -------------------------------------------------------------- */

  if ( scaled == 1 && nc_datatype == NC_USHORT ) {
    unsigned short valid_range[2];
    float scale;
    float offset;
    int nc_status;
    nc_status = nc_get_att(var.gid, var.varid, "valid_range", &valid_range);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_get_att(var.gid, var.varid, "scale_factor", &scale);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_get_att(var.gid, var.varid, "add_offset", &offset);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    status = MtkReadRawNcid(ncid, gridname, fieldname, region, &rawbuf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(rawbuf.nline, rawbuf.nsample, MTKe_float, &buf);
    MTK_ERR_COND_JUMP(status);

    float float_fill = -9999.0;

    unsigned short valid_min = valid_range[0];
    unsigned short valid_max = valid_range[1];
    for (int i = 0; i < buf.nline; i++) {
      for (int j = 0; j < buf.nsample; j++) {
        unsigned short raw = rawbuf.data.u16[i][j];
        if (raw < valid_min || raw > valid_max) {
          buf.data.f[i][j] = float_fill;
        } else {
          buf.data.f[i][j] = raw * scale + offset;
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read and unscale fields of type unsigned char                  */
  /* -------------------------------------------------------------- */

  } else if (scaled == 1 && nc_datatype == NC_UBYTE) {
    unsigned char valid_range[2];
    float scale;
    float offset;
    int nc_status;
    nc_status = nc_get_att(var.gid, var.varid, "valid_range", &valid_range);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_get_att(var.gid, var.varid, "scale_factor", &scale);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);
    nc_status = nc_get_att(var.gid, var.varid, "add_offset", &offset);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    status = MtkReadRawNcid(ncid, gridname, fieldname, region, &rawbuf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(rawbuf.nline, rawbuf.nsample, MTKe_float, &buf);
    MTK_ERR_COND_JUMP(status);

    float float_fill = -9999.0;

    unsigned char valid_min = valid_range[0];
    unsigned char valid_max = valid_range[1];
    for (int i = 0; i < buf.nline; i++) {
      for (int j = 0; j < buf.nsample; j++) {
        unsigned char raw = rawbuf.data.u8[i][j];
        if (raw < valid_min || raw > valid_max) {
          buf.data.f[i][j] = float_fill;
        } else {
          buf.data.f[i][j] = raw * scale + offset;
        }
      }
    }

  /* -------------------------------------------------------------- */
  /* Read float fields                                              */
  /* -------------------------------------------------------------- */

  } else {

    status = MtkReadRawNcid(ncid, gridname, fieldname, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

  }

  free(basefield);
  free(extradims);
  MtkDataBufferFree(&rawbuf);

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&buf);
  MtkDataBufferFree(&rawbuf);
  return status_code;
}
