/*===========================================================================
=                                                                           =
=                            MtkWriteBinFile3D                              =
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

/** \brief Write binary file from 3D buffer in stacked block format
 *
 *  \return MTK_SUCCESS if successful.
 *
 */

MTKt_status MtkWriteBinFile3D(
  const char *filename, /**< [IN] File name */
  MTKt_DataBuffer3D buf /**< [IN] Data buffer*/ )
{
  FILE *fp;			/* File pointer */
  char infofname[300];		/* Filename plus extension */
  char rawfname[300];		/* Filename plus extension */
  char *datatype[MTKd_NDATATYPE] = MTKd_DataType;
				/* Datatype description mapping */
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
  fwrite(buf.dataptr, buf.datasize, buf.nblock * buf.nline * buf.nsample, fp);
  fclose(fp);


  /* Write info file */
  strcpy(infofname, filename);
  strcat(infofname, ".info");
  if ((fp = fopen(infofname, "wb")) == NULL) {
    MTK_ERR_MSG_JUMP("Error opening infofile");
  }

  fprintf(fp, "filename = %s\n", rawfname);

  fprintf(fp, "image.header_bytes = 0\n");
  fprintf(fp, "image.nblock = %d\n", buf.nblock);
  fprintf(fp, "image.nline = %d\n", buf.nline);
  fprintf(fp, "image.nsample = %d\n", buf.nsample);
  fprintf(fp, "image.nblock * image.nline = %d\n", buf.nblock * buf.nline);
  fprintf(fp, "image.datatype = %s\n", datatype[buf.datatype]);
  fprintf(fp, "image.datasize = %d\n", buf.datasize);
  if (endian_ptr[0] == 1)
    fprintf(fp,"image.byteorder = little_endian\n");
  else
    fprintf(fp,"image.byteorder = big_endian\n");

  fclose(fp);

  return MTK_SUCCESS;
ERROR_HANDLE:
  return MTK_FAILURE;
}
