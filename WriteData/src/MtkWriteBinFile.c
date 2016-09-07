/*===========================================================================
=                                                                           =
=                             MtkWriteBinFile                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrWriteData.h"
#include "MisrError.h"
#include "MisrUtil.h"
#include <string.h>
#include <stdio.h>

/** \brief Write binary file 
 *
 *  \return MTK_SUCCESS if successful.
 *
 */

MTKt_status MtkWriteBinFile(
  const char *filename, /**< [IN] File name */
  MTKt_DataBuffer buf,  /**< [IN] Data buffer*/
  MTKt_MapInfo mapinfo  /**< [IN] Mapinfo */ )
{
  FILE *fp;			/* File pointer */
  char infofname[300];		/* Filename plus extension */
  char rawfname[300];		/* Filename plus extension */
  char *datatype[MTKd_NDATATYPE] = MTKd_DataType;
				/* Datatype description mapping */
  int i;			/* Loop index */
  int endian = 1;		/* Endian test */
  char *endian_ptr = (char *)&endian;

  if (filename == NULL)
    return MTK_NULLPTR;

  /* Write binary file */
  strcpy(rawfname, filename);
  strcat(rawfname, ".raw");
  if ((fp = fopen(rawfname, "wb")) == NULL) {
    MTK_ERR_MSG_JUMP("Error opening binfile");
  }
  fwrite(buf.dataptr, buf.datasize, buf.nline * buf.nsample, fp);
  fclose(fp);


  /* Write info file */
  strcpy(infofname, filename);
  strcat(infofname, ".info");
  if ((fp = fopen(infofname, "wb")) == NULL) {
    MTK_ERR_MSG_JUMP("Error opening infofile");
  }

  fprintf(fp, "filename = %s\n", rawfname);

  fprintf(fp, "image.header_bytes = 0\n");
  fprintf(fp, "image.nline = %d\n", buf.nline);
  fprintf(fp, "image.nsample = %d\n", buf.nsample);
  fprintf(fp, "image.datatype = %s\n", datatype[buf.datatype]);
  fprintf(fp, "image.datasize = %d\n", buf.datasize);
  if (endian_ptr[0] == 1)
    fprintf(fp,"image.byteorder = little_endian\n");
  else
    fprintf(fp,"image.byteorder = big_endian\n");

  fprintf(fp, "mapinfo.path = %d\n", mapinfo.path);
  fprintf(fp, "mapinfo.start_block = %d\n", mapinfo.start_block);
  fprintf(fp, "mapinfo.end_block = %d\n", mapinfo.end_block);
  fprintf(fp, "mapinfo.resolution = %d\n", mapinfo.resolution);
  fprintf(fp, "mapinfo.resfactor = %d\n", mapinfo.resfactor);
  fprintf(fp, "mapinfo.nline = %d\n", mapinfo.nline);
  fprintf(fp, "mapinfo.nsample = %d\n", mapinfo.nsample);
  fprintf(fp, "mapinfo.pixelcenter = %s\n",
	  mapinfo.pixelcenter == 1 ? "true" : "false");
  fprintf(fp, "mapinfo.som.ulc.x = %f\n", mapinfo.som.ulc.x);
  fprintf(fp, "mapinfo.som.ulc.y = %f\n", mapinfo.som.ulc.y);
  fprintf(fp, "mapinfo.som.ctr.x = %f\n", mapinfo.som.ctr.x);
  fprintf(fp, "mapinfo.som.ctr.y = %f\n", mapinfo.som.ctr.y);
  fprintf(fp, "mapinfo.som.lrc.x = %f\n", mapinfo.som.lrc.x);
  fprintf(fp, "mapinfo.som.lrc.y = %f\n", mapinfo.som.lrc.y);

  fprintf(fp, "mapinfo.geo.ulc.lat = %f\n", mapinfo.geo.ulc.lat);
  fprintf(fp, "mapinfo.geo.ulc.lon = %f\n", mapinfo.geo.ulc.lon);
  fprintf(fp, "mapinfo.geo.urc.lat = %f\n", mapinfo.geo.ulc.lat);
  fprintf(fp, "mapinfo.geo.urc.lon = %f\n", mapinfo.geo.ulc.lon);
  fprintf(fp, "mapinfo.geo.ctr.lat = %f\n", mapinfo.geo.ctr.lat);
  fprintf(fp, "mapinfo.geo.ctr.lon = %f\n", mapinfo.geo.ctr.lon);
  fprintf(fp, "mapinfo.geo.lrc.lat = %f\n", mapinfo.geo.lrc.lat);
  fprintf(fp, "mapinfo.geo.lrc.lon = %f\n", mapinfo.geo.lrc.lon);
  fprintf(fp, "mapinfo.geo.llc.lat = %f\n", mapinfo.geo.lrc.lat);
  fprintf(fp, "mapinfo.geo.llc.lon = %f\n", mapinfo.geo.lrc.lon);

  fprintf(fp, "mapinfo.pp.path = %d\n", mapinfo.pp.path);
  fprintf(fp, "mapinfo.pp.projcode = %lld\n", mapinfo.pp.projcode);
  fprintf(fp, "mapinfo.pp.zonecode = %lld\n", mapinfo.pp.zonecode);
  fprintf(fp, "mapinfo.pp.spherecode = %lld\n", mapinfo.pp.spherecode);
  for (i = 0; i < 15 ; i++)
    fprintf(fp, "mapinfo.pp.projparam[%d] = %f\n", i, mapinfo.pp.projparam[i]);
  for (i = 0; i < 2 ; i++)
    fprintf(fp, "mapinfo.pp.ulc[%d] = %f\n", i, mapinfo.pp.ulc[i]);
  for (i = 0; i < 2 ; i++)
    fprintf(fp, "mapinfo.pp.lrc[%d] = %f\n", i, mapinfo.pp.lrc[i]);
  fprintf(fp, "mapinfo.pp.nblock = %d\n", mapinfo.pp.nblock);
  fprintf(fp, "mapinfo.pp.nline = %d\n", mapinfo.pp.nline);
  fprintf(fp, "mapinfo.pp.nsample = %d\n", mapinfo.pp.nsample);
  for (i = 0; i < 179 ; i++)
    fprintf(fp, "mapinfo.pp.reloffset[%d] = %f\n", i, mapinfo.pp.reloffset[i]);
  fprintf(fp, "mapinfo.pp.resolution = %d\n", mapinfo.pp.resolution);

  fclose(fp);

  return MTK_SUCCESS;
ERROR_HANDLE:
  return MTK_FAILURE;
}
