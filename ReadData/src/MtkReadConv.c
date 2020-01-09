/*===========================================================================
=                                                                           =
=                               MtkReadConv                                 =
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
#include "MisrMapQuery.h"
#include "MisrFileQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <mfhdf.h>
#include <HdfEosDef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/** \brief Reads any grid/field from a MISR conventional product file.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Special Note
 *  Typically this function is not called directly.  Instead use MtkReadData().
 */

MTKt_status MtkReadConv(
  const char *filename,     /**< [IN] File name */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int32 fid = FAIL;		/* HDF-EOS File id */
  intn hdfstatus;		/* HDF return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file. */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read data. */
  status = MtkReadConvFid(fid, gridname, fieldname, region, databuf, mapinfo);
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

/** \brief Version of MtkReadConv that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkReadConvFid(
  int32 fid,                /**< [IN] HDF-EOS file identifier */
  const char *gridname,     /**< [IN] Grid name */
  const char *fieldname,    /**< [IN] Field name */
  MTKt_Region region,       /**< [IN] Region */
  MTKt_DataBuffer *databuf, /**< [OUT] Data buffer */
  MTKt_MapInfo *mapinfo     /**< [OUT] Mapinfo */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer buf = MTKT_DATABUFFER_INIT;
                                /* Data buffer structure */
  MTKt_DataBuffer filebuf_tmp = MTKT_DATABUFFER_INIT;
                                /* Temp data buffer structure for file */
  MTKt_DataBuffer fill = MTKT_DATABUFFER_INIT;
                                /* Fill buffer structure */
  MTKt_MapInfo map = MTKT_MAPINFO_INIT;
				/* Map info structure */
  MTKt_MapInfo mapfile = MTKT_MAPINFO_INIT;
				/* Map info structure for the file */
  int path;			/* Path */
  int resolution;		/* Resolution */ 
  MTKt_SomCoord ulc_som = MTKT_SOMCOORD_INIT;
				/* ULC SOM coordinates */
  MTKt_SomCoord lrc_som = MTKT_SOMCOORD_INIT;
				/* LRC SOM coordinates */
  double x;			/* Som X */
  int ulc_line;			/* ULC Line */
  int ulc_sample;		/* ULC Sample */
  int lrc_line;			/* LRC Line */
  int lrc_sample;		/* LRC Sample */
  float line;			/* Line */
  float sample;			/* Sample */
  int nline;			/* Number of line */
  int nsample;			/* Number of sample */
  int l;			/* Line index */
  int s;			/* Sample index */
  MTKt_DataType datatype;	/* Datatype */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 gid = FAIL;		/* HDF-EOS Grid id */
  int32 start[10];		/* HDF-EOS start dimension */
  int32 start_copy[10];		/* HDF-EOS start dimension copy*/
  int32 edge[10];		/* HDF-EOS edge dimension */
  int32 edge_copy[10];		/* HDF-EOS edge dimension copy */
  int32 hdf_datatype;		/* HDF-EOS data type */
  int32 rank;			/* HDF-EOS rank */
  int32 dims[10];		/* HDF-EOS dimensions */
  int32 dims_copy[10];		/* HDF-EOS dimensions copy */
  char dimlist[80];		/* HDF-EOS dimension name list */
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */
  int dims_reversed = 0;    /* Whether the dims are "reversed"*/
  int i;			/* Loop index */
  int32 xdimsize;		/* X dimension size of file */
  int32 ydimsize;		/* Y dimension size of file */
  float64 ulc[2];		/* Upper left corner in meters */
  float64 lrc[2];		/* Lower right corner in meters */
  char *dataptr;		/* Data pointer */
  int32           HDFfid;  	/* HDF file identifier (not used) */
  int32   	    sid;        /* HDF SD identifier */

  if (gridname == NULL) return MTK_NULLPTR;
  if (fieldname == NULL) return MTK_NULLPTR;

  /* Get the HDF SD identifier. */
  hdfstatus = EHidinfo(fid, &HDFfid, &sid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_EHIDINFO_FAILED);

  /* Determine path of this file */
  status = MtkFileToPathFid(sid, &path);
  MTK_ERR_COND_JUMP(status);

  /* Determine resolution from file and gridname */
  status = MtkFileGridToResolutionFid(fid, gridname, &resolution);
  MTK_ERR_COND_JUMP(status);

  /* Snap region to som grid for this path */
  status = MtkSnapToGrid(path, resolution, region, &map);
  MTK_ERR_COND_JUMP(status);

  /* Attach to grid */
  gid = GDattach(fid, (char*)gridname);
  if (gid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  /* Get grid info */
  hdfstatus = GDgridinfo(gid, &xdimsize, &ydimsize, ulc, lrc);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDGRIDINFO_FAILED);

  /* Parse fieldname for extra dimensions */
  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  /* Determine rank, dimensions and datatype of the field */
  hdfstatus = GDfieldinfo(gid, basefield, &rank, dims, &hdf_datatype, dimlist);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDFIELDINFO_FAILED);

  /* If dimlist doesn't start with an X for XDim, assume dims are from MISR HR */
  /* Band,Camera,XDim,YDim should be XDim,YDim,Band,Camera */
  if (dimlist[0] != 'X') {
	  /* XDim */
	  dims_copy[0] = dims[rank - 2];
	  /* YDim */
	  dims_copy[1] = dims[rank - 1];
	  /* Remaining, shifted by two */
	  for (i = 2; i < rank; i++) {
		  dims_copy[i] = dims[i - 2];
	  }
	  for (i = 0; i < rank; i++) {
		  dims[i] = dims_copy[i];
	  }
	  dims_reversed = 1;
  }

  /* Check range against extra dimensions */
  if (rank != nextradims + 2) MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  /* Convert hdf datatype to mtk datatype */
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  MTK_ERR_COND_JUMP(status);

  /* Create a map structure for the contents of the file, then we can */
  /* determine the coordinates of the intersecting map area we need to load */
  /* Note - ulc y and lrc y are flipped. */
  mapfile = map;
  mapfile.nline = dims[0];
  mapfile.nsample = dims[1];
  mapfile.som.ulc.x = ulc[0];
  mapfile.som.ulc.y = lrc[1];
  mapfile.som.lrc.x = lrc[0];
  mapfile.som.lrc.y = ulc[1];
  mapfile.som.ctr.x = mapfile.som.ulc.x + 
    ((mapfile.som.lrc.x - mapfile.som.ulc.x) / 2.0);
  mapfile.som.ctr.y = mapfile.som.ulc.y + 
    ((mapfile.som.lrc.y - mapfile.som.ulc.y) / 2.0);

  /* Determine ulc line and sample of intersecting area. */
  /* If out of bounds then clip to size of file coordinates. */
  /* The status of this routine can be ignored because we catch */
  /* out of bounds when line or sample is -1.0. */
  /* Also, subtract half a pixel to center coordinate. */
  MtkSomXYToLS(mapfile, map.som.ulc.x, map.som.ulc.y, &line, &sample);

  line -= 0.5;
  sample -= 0.5;

  if (line < 0.0) ulc_line = 0;
  else ulc_line = (int)line;

  if (sample < 0.0) ulc_sample = 0;
  else ulc_sample = (int)sample;

  /* Determine som x/y coordinates of ulc intersecting coordinate */
  status = MtkLSToSomXY(mapfile, (float)(ulc_line + 0.5), (float)(ulc_sample + 0.5), &ulc_som.x, &ulc_som.y);
  MTK_ERR_COND_JUMP(status);

  /* Determine lrc line and sample of intersecting area */
  /* If out of bounds then clip to size of file coordinates. */
  /* The status of this routine can be ignored because we catch */
  /* out of bounds when line or sample is -1.0. */
  /* Also, subtract half a pixel to center coordinate. */
  MtkSomXYToLS(mapfile, map.som.lrc.x, map.som.lrc.y, &line, &sample);

  line -= 0.5;
  sample -= 0.5;

  if (line < 0.0) lrc_line = dims[0] - 1;
  else lrc_line = (int)line;

  if (sample < 0.0) lrc_sample = dims[1] - 1;
  else lrc_sample = (int)sample;

  /* Determine som x/y coordinates of lrc intersecting coordinate */
  status = MtkLSToSomXY(mapfile, (float)(lrc_line + 0.5), (float)(lrc_sample + 0.5), &lrc_som.x, &lrc_som.y);
  MTK_ERR_COND_JUMP(status);

  /* Determine number of lines and samples of the intersecting area to load */
  nline = lrc_line - ulc_line + 1;
  nsample = lrc_sample - ulc_sample + 1;

  start[0] = ulc_line;
  start[1] = ulc_sample;

  edge[0] = nline;
  edge[1] = nsample;

  for (i = 2; i < rank; i++) {
    start[i] = extradims[i-2];
    edge[i] = 1;
  }

  /* Dims didn't start with XDim, assuming MISR HR format */
  if (dims_reversed) {
	  /* XDim */
	  start_copy[rank - 2] = start[0];
	  edge_copy[rank - 2] = edge[0];
	  /* YDim */
	  start_copy[rank - 1] = start[1];
	  edge_copy[rank - 1] = edge[1];
	  /* Remaining, shifted by two */
	  for (i = 0; i < rank - 2; i++) {
		  start_copy[i] = start[i + 2];
		  edge_copy[i] = edge[i + 2];
	  }
	  for (i = 0; i < rank; i++) {
		  start[i] = start_copy[i];
		  edge[i] = edge_copy[i];
	  }
  }

  /* Allocate the temp data buffer */
  status = MtkDataBufferAllocate(nline, nsample, datatype, &filebuf_tmp);
  MTK_ERR_COND_JUMP(status);

  /* Read field data */
  hdfstatus = GDreadfield(gid, basefield, start, NULL, edge, 
			  filebuf_tmp.dataptr);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDREADFIELD_FAILED);

  /* Detach from grid */
  hdfstatus = GDdetach(gid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  /* Allocate the data buffer */
  status = MtkDataBufferAllocate(map.nline, map.nsample, datatype, &buf);
  MTK_ERR_COND_JUMP(status);

  /* Get fill value and prefill buffer only if a fill value is defined */
  status = MtkFillValueGetFid(fid, gridname, fieldname, &fill);
  if (status == MTK_SUCCESS) {
    dataptr = (char *)buf.dataptr;
    for (l = 0; l < map.nline; l++) {
      for (s = 0; s < map.nsample; s++) {
	memcpy((void *)dataptr, fill.dataptr, buf.datasize);
	dataptr += buf.datasize;
      }
    }
    MtkDataBufferFree(&fill);
  }

  /* Copy intersecting region to output buffer */
  i = 0;
  for (x = ulc_som.x; x <= lrc_som.x; x += map.resolution) {
    MtkSomXYToLS(map, x, ulc_som.y, &line, &sample);
    memcpy((void *)&(buf.data.u8[(int)line][(int)sample]),
	   (void *)&(filebuf_tmp.data.u8[i++][0]),
	   nsample * buf.datasize);
  }

  free(basefield);
  free(extradims);
  MtkDataBufferFree(&filebuf_tmp);

  *databuf = buf;
  *mapinfo = map;

  return MTK_SUCCESS;
 ERROR_HANDLE:
  if (gid != FAIL) GDdetach(gid);
  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);
  MtkDataBufferFree(&filebuf_tmp);
  MtkDataBufferFree(&buf);
  return status_code;
}
