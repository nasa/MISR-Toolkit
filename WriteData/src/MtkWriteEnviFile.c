/*===========================================================================
=                                                                           =
=                             MtkWriteEnviFile                              =
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
#include "MisrUnitConv.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/** \brief Write ENVI file 
 *
 *  \return MTK_SUCCESS if successful.
 *
 */

MTKt_status MtkWriteEnviFile(
  const char *envifilename,  /**< [IN] ENVI output file name */
  MTKt_DataBuffer buf,       /**< [IN] Data buffer*/
  MTKt_MapInfo mapinfo,      /**< [IN] Mapinfo */
  const char *misrfilename,  /**< [IN] MISR file name */
  const char *misrgridname,  /**< [IN] MISR grid name */
  const char *misrfieldname  /**< [IN] MISR field name */ )
{
  FILE *fp;			/* File pointer */
  char hdrfname[300];		/* ENVI Filename plus extension */
  char imgfname[300];		/* ENVI Filename plus extension */
  int idl_datatype[MTKd_NDATATYPE] = { 0, 1, 1, 1, 1, 2, 12, 3, 13, 14, 15, 4, 5 };
				/* IDL/MTK Datatype mapping - 
				   1=8 bit byte;
				   2=16-bit signed integer; 
				   3=32-bit signed long integer; 
				   4=32-bit floating point; 
				   5=64-bit double precision floating point; 
				   6=2x32-bit complex, real-imaginary pair of double precision; 
				   9=2x64-bit double precision complex, real-imaginary pair of double precision; 
				   12=16-bit unsigned integer; 
				   13=32-bit unsigned long integer; 
				   14=64-bit signed long integer; and 
				   15=64-bit unsigned long integer. */
  double a, b, e2;		/* Ellipsoid semi-major axis, semi-minor axis, 
				   and eccentricity squared */
  double IncAng, AscLong;	/* Inclination angle and Longitude of ascending node (dd) */
  double x0, y0;		/* False Easting and False Northing (dd) */
  double PSRev, LRat;		/* Satellite revolution and Landsat Ratio (dd) */
  int endian = 1;		/* Endian test */
  char *endian_ptr = (char *)&endian;

  if (envifilename == NULL)
    return MTK_NULLPTR;

  /* Write binary file */

  strcpy(imgfname, envifilename);
  strcat(imgfname, ".img");
  if ((fp = fopen(imgfname, "wb")) == NULL) {
    MTK_ERR_MSG_JUMP("Error opening binfile");
  }
  fwrite(buf.dataptr, buf.datasize, buf.nline * buf.nsample, fp);
  fclose(fp);


  /* Write hdr file */

  strcpy(hdrfname, envifilename);
  strcat(hdrfname, ".hdr");
  if ((fp = fopen(hdrfname, "wb")) == NULL) {
    MTK_ERR_MSG_JUMP("Error opening hdrfile");
  }

  fprintf(fp, "ENVI\n");
  fprintf(fp, "description = {%s}\n", misrfilename);
  fprintf(fp, "samples = %d\n", buf.nsample);
  fprintf(fp, "lines = %d\n", buf.nline);
  fprintf(fp, "bands = 1\n");
  fprintf(fp, "header offset = 0\n");
  fprintf(fp, "file type = ENVI Standard\n");
  fprintf(fp, "data type = %d\n", idl_datatype[buf.datatype]);
  fprintf(fp, "interleave = bsq\n");
  fprintf(fp, "sensor type = MISR\n");
  if (endian_ptr[0] == 1)
    fprintf(fp,"byte order = 0\n");
  else
    fprintf(fp,"byte order = 1\n");

  fprintf(fp, "x start = 0.0\n");
  fprintf(fp, "y start = 0.0\n");

  fprintf(fp, "map info = {");
  fprintf(fp, "Space Oblique Mercator A (MISR Path %d), ", mapinfo.path);
  fprintf(fp, "1.0, ");
  fprintf(fp, "1.0, ");
  fprintf(fp, "%f, ", mapinfo.som.ulc.x);
  fprintf(fp, "%f, ", mapinfo.som.ulc.y);
  fprintf(fp, "%f, ", (float)mapinfo.resolution);
  fprintf(fp, "%f, ", (float)mapinfo.resolution);
  fprintf(fp, "WGS-84, ");
  fprintf(fp, "units=Meters, ");
  fprintf(fp, "rotation=90.0");
  fprintf(fp, "}\n");

  /* Convert eccentricity fo the ellipsoid to b parameter */
  a = mapinfo.pp.projparam[0];
  e2 = mapinfo.pp.projparam[1];
  e2 = fabs(e2);
  b = sqrt(pow(a,2) - e2 * pow(a,2));
  /* Convert from packed DMS to decimal degrees */
  MtkDmsToDd(mapinfo.pp.projparam[3], &IncAng);
  MtkDmsToDd(mapinfo.pp.projparam[4], &AscLong);
  PSRev = mapinfo.pp.projparam[8];
  MtkDmsToDd(mapinfo.pp.projparam[9], &LRat);
  /* No false easting or false northing */
  x0 = 0.0;
  y0 = 0.0;
  /* Apparently Envi uses 37 for generic SOM A - see ENVI's map_proj.txt */

  fprintf(fp, "projection info = {");
  fprintf(fp, "37, ");
  fprintf(fp, "%f, ", a);
  fprintf(fp, "%f, ", b);
  fprintf(fp, "%f, ", IncAng);
  fprintf(fp, "%f, ", AscLong);
  fprintf(fp, "%f, ", x0);
  fprintf(fp, "%f, ", y0);
  fprintf(fp, "%f, ", PSRev);
  fprintf(fp, "%f, ", LRat);
  fprintf(fp, "WGS-84, ");
  fprintf(fp, "Space Oblique Mercator A (MISR Path %d), ", mapinfo.path);
  fprintf(fp, "units=Meters");
  fprintf(fp, "}\n");

  fprintf(fp, "pixel size = {");
  fprintf(fp, "%f, ", (float)mapinfo.resolution);
  fprintf(fp, "%f, ", (float)mapinfo.resolution);
  fprintf(fp, "units=Meters");
  fprintf(fp, "}\n");

  fprintf(fp, "band names = {");
  fprintf(fp, "%s:%s", misrgridname, misrfieldname);
  fprintf(fp, "}\n");

  fclose(fp);

  return MTK_SUCCESS;
ERROR_HANDLE:
  return MTK_FAILURE;
}
