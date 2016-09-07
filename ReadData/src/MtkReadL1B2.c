/*===========================================================================
=                                                                           =
=                               MtkReadL1B2                                 =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <ctype.h>
/* M_PI is not defined in math.h in Linux unless __USE_BSD is defined */
/* and you can define it at the gcc command-line if -ansi is set */
#ifndef __USE_BSD
# define __USE_BSD
#endif
#include <math.h>
#include <hdf.h>
#include <HdfEosDef.h>

/** \brief Reads, unpacks and unscales any L1B2 grid/field from a
 *         MISR L1B2 product file.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Special Note
 *  Typically this function is not called directly.  Instead use MtkReadData().
 */

MTKt_status MtkReadL1B2(
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
  status = MtkReadL1B2Fid(fid, gridname, fieldname, region, databuf, mapinfo);
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

/** \brief Version of MtkReadL1B2 that takes an HDF-EOS file identifier rather than a filename.
 *         MISR L1B2 product file.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadL1B2Fid(
  int32 fid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Return status code for error macros */
  MTKt_FileType filetype;	/* File type */
  MTKt_MapInfo tmpmap = MTKT_MAPINFO_INIT;
                                /* Temp Map info structure */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
                                /* Map info structure */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_DataBuffer radrdqi = MTKT_DATABUFFER_INIT;
				/* Radiance/RDQI Data buffer structure */
  MTKt_DataBuffer conv_factor = MTKT_DATABUFFER_INIT;
				/* Brf conversion factor */
  MTKt_DataBuffer scale_factor = MTKT_DATABUFFER_INIT;
				/* Radiance scale factor */
  MTKt_DataBuffer fill_value = MTKT_DATABUFFER_INIT;
				/* Radiance fill_value */
  MTKt_DataBuffer solar_irradiance = MTKT_DATABUFFER_INIT;
				/* Solar Irradiance */
  MTKt_DataBuffer sun_distance_au = MTKT_DATABUFFER_INIT;
				/* Sun distance in AU */
  double sun_distance_au_squared; /* Sun distance squared (AU) */
  char grid[MAXSTR];		/* Grid to read */
  char field[MAXSTR];		/* Field to read */
  char fieldstr[MAXSTR];	/* Field string */
  char *fieldarr[2];		/* Field array */
  char *sp;			/* Pointer to string */
  int lf;			/* Line index scale factor */
  int sf;			/* Sample index scale factor */
  int l;			/* Line index */
  int s;			/* Sample index */

  if (gridname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  if (fieldname == NULL) MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* ---------------------------------------------------------- */
  /* Parse fieldname to determine unpacking or unscaling method */
  /* ---------------------------------------------------------- */

  /* Make a working copy of fieldname */
  strncpy(fieldstr, fieldname, MAXSTR);
  fieldarr[0] = fieldstr;

  /* Separate band from base field, point fieldarr[1] to null terminator */
  /* of fieldstr if there are now spaces in fieldstr, else just fail */
  if ((sp = strchr(fieldstr, ' ')) != NULL) {
    *sp = '\0';
    fieldarr[1] = ++sp;

    /* Convert to lower case for comparison */
    while (*sp != '\0') { 
      *sp = (char)tolower((int)*sp);
      sp++;
    }
  } else if ((sp = strchr(fieldstr, '\0')) != NULL) {
    fieldarr[1] = sp;
  } else {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* ------------------------------------------------------- */
  /* Determine which unpacking or unscaling method and do it */
  /* ------------------------------------------------------- */

  if (strncmp(fieldarr[1], "radiance", MAXSTR) == 0) {

    /* Radiance */

    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, " ", MAXSTR);
    strncat(field, "Radiance/RDQI", MAXSTR);

    status = MtkReadRawFid(fid, gridname, field, region, &radrdqi, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill_value);
    MTK_ERR_COND_JUMP(status);

    /* If filetype is TERRAIN adjust fill value to account for obscured by
       topography flag. */
    status = MtkFileTypeFid(fid, &filetype);
    MTK_ERR_COND_JUMP(status);
    if (filetype ==  MTK_GRP_TERRAIN_GM || filetype ==  MTK_GRP_TERRAIN_LM)
      fill_value.data.u16[0][0] -= 4;

    status = MtkDataBufferAllocate(radrdqi.nline, radrdqi.nsample, MTKe_float, 
				   &buf);
    MTK_ERR_COND_JUMP(status);

    status = MtkGridAttrGetFid(fid, gridname, "Scale factor", &scale_factor);
    MTK_ERR_COND_JUMP(status);

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
	if (radrdqi.data.u16[l][s] < fill_value.data.u16[0][0]) {
	  buf.data.f[l][s] = (MTKt_float)((radrdqi.data.u16[l][s] >> 2) * 
					  scale_factor.data.d[0][0]);
	} else {
	  buf.data.f[l][s] = 0.0;
	}
      }
    }

    MtkDataBufferFree(&radrdqi);
    MtkDataBufferFree(&scale_factor);
    MtkDataBufferFree(&fill_value);

  } else if (strncmp(fieldarr[1], "scaled radiance", MAXSTR) == 0 ||
	     strncmp(fieldarr[1], "dn", MAXSTR) == 0) {

    /* Scaled Radiance (DN) */

    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, " ", MAXSTR);
    strncat(field, "Radiance/RDQI", MAXSTR);

    status = MtkReadRawFid(fid, gridname, field, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill_value);
    MTK_ERR_COND_JUMP(status);

    /* If filetype is TERRAIN adjust fill value to account for obscured by
       topography flag. */
    status = MtkFileTypeFid(fid, &filetype);
    MTK_ERR_COND_JUMP(status);
    if (filetype ==  MTK_GRP_TERRAIN_GM || filetype ==  MTK_GRP_TERRAIN_LM)
      fill_value.data.u16[0][0] -= 4;

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
	if (buf.data.u16[l][s] < fill_value.data.u16[0][0]) {
	  buf.data.u16[l][s] = buf.data.u16[l][s] >> 2;
	} else {
	  buf.data.u16[l][s] = 0;
	}
      }
    }

    MtkDataBufferFree(&fill_value);

  } else if (strncmp(fieldarr[1], "rdqi", MAXSTR) == 0) {

    /* RDQI */

    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, " ", MAXSTR);
    strncat(field, "Radiance/RDQI", MAXSTR);

    status = MtkReadRawFid(fid, gridname, field, region, &radrdqi, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(radrdqi.nline, radrdqi.nsample, MTKe_uint8, 
				   &buf);
    MTK_ERR_COND_JUMP(status);

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
	buf.data.u8[l][s] = (MTKt_uint8)(radrdqi.data.u16[l][s] & 0x0003);
      }
    }

    MtkDataBufferFree(&radrdqi);

  } else if (strncmp(fieldarr[1], "equivalent reflectance", MAXSTR) == 0) {

    /* Equivalent Reflectance */

    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, " ", MAXSTR);
    strncat(field, "Radiance/RDQI", MAXSTR);

    status = MtkReadRawFid(fid, gridname, field, region, &radrdqi, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill_value);
    MTK_ERR_COND_JUMP(status);

    /* If filetype is TERRAIN adjust fill value to account for obscured by
       topography flag. */
    status = MtkFileTypeFid(fid, &filetype);
    MTK_ERR_COND_JUMP(status);
    if (filetype ==  MTK_GRP_TERRAIN_GM || filetype ==  MTK_GRP_TERRAIN_LM)
      fill_value.data.u16[0][0] -= 4;

    status = MtkDataBufferAllocate(radrdqi.nline, radrdqi.nsample, MTKe_float, 
				   &buf);
    MTK_ERR_COND_JUMP(status);

    status = MtkGridAttrGetFid(fid, gridname, "Scale factor", &scale_factor);
    MTK_ERR_COND_JUMP(status);

    status = MtkGridAttrGetFid(fid, gridname, "std_solar_wgted_height",
			    &solar_irradiance);
    MTK_ERR_COND_JUMP(status);

    status = MtkGridAttrGetFid(fid, gridname, "SunDistanceAU",
			    &sun_distance_au);
    MTK_ERR_COND_JUMP(status);

    sun_distance_au_squared = sun_distance_au.data.d[0][0] * 
      sun_distance_au.data.d[0][0];

    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
	if (radrdqi.data.u16[l][s] < fill_value.data.u16[0][0]) { 
	  buf.data.f[l][s] = (MTKt_float)(M_PI * sun_distance_au_squared *
					  ((radrdqi.data.u16[l][s] >> 2) *
					  scale_factor.data.d[0][0]) /
					  solar_irradiance.data.f[0][0]);
	} else {
	  buf.data.f[l][s] = 0.0;
	}
      }
    }

    MtkDataBufferFree(&radrdqi);
    MtkDataBufferFree(&scale_factor);
    MtkDataBufferFree(&fill_value);
    MtkDataBufferFree(&solar_irradiance);
    MtkDataBufferFree(&sun_distance_au);

  } else if (strncmp(fieldarr[1], "brf", MAXSTR) == 0) {

    /* Brf */

    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, " ", MAXSTR);
    strncat(field, "Radiance/RDQI", MAXSTR);

    status = MtkReadRawFid(fid, gridname, field, region, &radrdqi, &map);
    MTK_ERR_COND_JUMP(status);

    status = MtkFillValueGetFid(fid, gridname, field, &fill_value);
    MTK_ERR_COND_JUMP(status)

    /* If filetype is TERRAIN adjust fill value to account for obscured by
       topography flag. */
    status = MtkFileTypeFid(fid, &filetype);
    MTK_ERR_COND_JUMP(status);
    if (filetype ==  MTK_GRP_TERRAIN_GM || filetype ==  MTK_GRP_TERRAIN_LM)
      fill_value.data.u16[0][0] -= 4;

    grid[0] = '\0';
    strncat(grid, "BRF Conversion Factors", MAXSTR);
    field[0] = '\0';
    strncat(field, fieldarr[0], MAXSTR);
    strncat(field, "ConversionFactor", MAXSTR);

    status = MtkReadRawFid(fid, grid, field, region, &conv_factor, &tmpmap);
    MTK_ERR_COND_JUMP(status);

    status = MtkDataBufferAllocate(radrdqi.nline, radrdqi.nsample, MTKe_float, 
				   &buf);
    MTK_ERR_COND_JUMP(status);

    status = MtkGridAttrGetFid(fid, gridname, "Scale factor", &scale_factor);
    MTK_ERR_COND_JUMP(status)

    lf = radrdqi.nline / conv_factor.nline;
    sf = radrdqi.nsample / conv_factor.nsample;
    for (l = 0; l < buf.nline; l++) {
      for (s = 0; s < buf.nsample; s++) {
	if (radrdqi.data.u16[l][s] < fill_value.data.u16[0][0] &&
	    conv_factor.data.f[l/lf][s/sf] > 0.0) { 
	  buf.data.f[l][s] = (MTKt_float)((radrdqi.data.u16[l][s] >> 2) *
					  scale_factor.data.d[0][0] *
					  conv_factor.data.f[l/lf][s/sf]);
	} else {
	  buf.data.f[l][s] = 0.0;
	}
      }
    }

    MtkDataBufferFree(&radrdqi);
    MtkDataBufferFree(&conv_factor);
    MtkDataBufferFree(&scale_factor);
    MtkDataBufferFree(&fill_value);

  } else {

    /* Native Radiance/RDQI or Geometric Parameters */

    status = MtkReadRawFid(fid, gridname, fieldname, region, &buf, &map);
    MTK_ERR_COND_JUMP(status);

  }

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  MtkDataBufferFree(&radrdqi);
  MtkDataBufferFree(&conv_factor);
  MtkDataBufferFree(&scale_factor);
  MtkDataBufferFree(&fill_value);
  MtkDataBufferFree(&buf);
  MtkDataBufferFree(&solar_irradiance);
  MtkDataBufferFree(&sun_distance_au);
  return status_code;
}
