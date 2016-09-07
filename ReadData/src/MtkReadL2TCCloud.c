/*===========================================================================
=                                                                           =
=                              MtkReadL2TCCloud                             =
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

/** \brief Reads, unpacks and unscales TC_CLOUD grid/fields from a
 *         MISR L2 TC_CLOUD product file.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Special Note
 *  Typically this function is not called directly.  Instead use MtkReadData().
 */

MTKt_status MtkReadL2TCCloud(
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
  status = MtkReadL2TCCloudFid(fid, gridname, fieldname, region, databuf, mapinfo);
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

/** \brief Version of MtkReadL2TCCloud that takes and HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadL2TCCloudFid(
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
 // int num_attrs;                /* Number of attributes */
//  char **attrlist;              /* Attribute list */
/*  status = MtkFieldAttrListFid(fid,"CloudMotionCrossTrack",&num_attrs,&attrlist);
  status = MtkFieldAttrGetFid(fid, "CloudMotionCrossTrack", "scale_factor", &scale);   */

  if (raw == MTK_FALSE) {
    strcpy(attr, "valid_min");    
    status = MtkFieldAttrGetFid(fid, basefield, attr, &min);
    if (status == MTK_SUCCESS) attrcnt++;

    strcpy(attr, "valid_max");
    status = MtkFieldAttrGetFid(fid, basefield, attr, &max);
    if (status == MTK_SUCCESS) attrcnt++;

    strcpy(attr, "scale_factor");
    status = MtkFieldAttrGetFid(fid, basefield, attr, &scale);
    if (status == MTK_SUCCESS) attrcnt++;

    strcpy(attr, "add_offset");    
    status = MtkFieldAttrGetFid(fid, basefield, attr, &offset);
    if (status == MTK_SUCCESS) attrcnt++;

    strcpy(attr, "_FillValue");
    status = MtkFieldAttrGetFid(fid, basefield, attr, &fill);
    if (status == MTK_SUCCESS) attrcnt++;    
  }

  /* --------------------------------------------------------------- */
  /* Determine which unpacking or unscaling method to do, then do it */
  /* --------------------------------------------------------------- */

  
  if (attrcnt == 5 && flag == MTK_FALSE && raw == MTK_FALSE) {

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
	        } else {
	          buf.data.f[l][s] = (MTKt_float)(rawbuf.data.u16[l][s] * 
			       scale.data.f[0][0]) + offset.data.f[0][0];
	        }
	      }
      }
      break;

    case MTKe_int16:
      for (l = 0; l < buf.nline; l++) {
	      for (s = 0; s < buf.nsample; s++) {
	        if (rawbuf.data.i16[l][s] == fill.data.i16[0][0]) {
	          buf.data.f[l][s] = (MTKt_float)fill.data.i16[0][0];
	        } else {
	          buf.data.f[l][s] = (MTKt_float)(rawbuf.data.i16[l][s] * 
			       scale.data.f[0][0]) + offset.data.f[0][0];
	        }
	      }
      }
      if ((sp = strcasestr(basefield, "ERImageContrast_WithoutWindCorrection")) != NULL ) {
        for (l = 0; l < buf.nline; l++) {
  	      for (s = 0; s < buf.nsample; s++) {
  	        if (rawbuf.data.i16[l][s] == fill.data.i16[0][0]) {
  	          buf.data.f[l][s] = (MTKt_float)fill.data.i16[0][0];
  	        } else {
  	          buf.data.f[l][s] = (MTKt_float)pow(10.0,(buf.data.f[l][s]));
  	        }
  	      }
        }
      }
      break;      
 
    default:
      MTK_ERR_COND_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
      break;
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
  return status_code;
}
