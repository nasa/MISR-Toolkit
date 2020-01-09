/*===========================================================================
=                                                                           =
=                          MtkFileGridToFieldList                           =
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

/** \brief Read list of fields from file
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of fields from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf and the grid \c BlueBand
 *
 *  \code
 *  status = MtkFileGridToFieldList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand", &nfields, &fieldlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a fieldlist
 */

MTKt_status MtkFileGridToFieldList(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status;       /* Return status */

  status = MtkFileGridToFieldListNC(filename, gridname, nfields, fieldlist); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileGridToFieldListHDF(filename, gridname, nfields, fieldlist); // try HDF
}

MTKt_status MtkFileGridToFieldListNC(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
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

  /* Read list of fields. */
  status = MtkFileGridToFieldListNcid(ncid, gridname, nfields, fieldlist);
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

MTKt_status MtkFileGridToFieldListHDF(
  const char *filename, /**< [IN] Filename */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status;		/* Return status of called functions */
  MTKt_status status_code;	/* Return status of this function. */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 Fid = FAIL;             /* HDF-EOS File ID */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);  

  /* Open HDF file for reading */
  hdfstatus = Fid = GDopen((char*)filename,DFACC_READ);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read list of fields. */
  status = MtkFileGridToFieldListFid(Fid, gridname, nfields, fieldlist);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = GDclose(Fid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  Fid = FAIL;

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (Fid != FAIL) GDclose(Fid);

  return status_code;
}

/** \brief Version of MtkFileGridToFieldList that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridToFieldListFid(
  int32 Fid,            /**< [IN] HDF-EOS file identifier */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status;		/* Return status of called functions */
  MTKt_status status_code;	/* Return status of this function. */
  intn hdfstatus;		/* HDF-EOS return status */
  int32 Gid = FAIL;             /* HDF-EOS Grid ID */
  int32 num_fields = 0;         /* Number of fields */
  char *list = NULL;            /* List of fields */
  int i;
  char *temp = NULL;
  MTKt_FileType filetype;
  int32 str_buffer_size = 0;

  /* Check Arguments */
  if (fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *fieldlist = NULL; /* Set output to NULL to prevent freeing unallocated
                        memory in case of error. */

  if (gridname == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nfields == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Attach to grid */
  hdfstatus = Gid = GDattach(Fid,(char*)gridname);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  /* Query length of fields string */
  hdfstatus = GDnentries(Gid, HDFE_NENTDFLD, &str_buffer_size);
  if(hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDNENTRIES_FAILED);

  list = (char*)malloc((str_buffer_size + 1) * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Get list of fields */
  hdfstatus = num_fields = GDinqfields(Gid,list,NULL,NULL);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQFIELDS_FAILED);

  *nfields = num_fields;
  *fieldlist = (char**)calloc(num_fields,sizeof(char*));
  if (*fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    
  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    (*fieldlist)[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
    if ((*fieldlist)[i] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
    strcpy((*fieldlist)[i],temp);
    temp = strtok(NULL,",");
    ++i;
  }

  free(list);

  hdfstatus = GDdetach(Gid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDDETACH_FAILED);

  /* Add all possible unpacked and unscaled fields to raw field list */
  status = MtkFileTypeFid(Fid, &filetype);
  MTK_ERR_COND_JUMP(status);

  switch (filetype) {

  case MTK_GRP_ELLIPSOID_GM:
  case MTK_GRP_TERRAIN_GM:
  case MTK_GRP_ELLIPSOID_LM:
  case MTK_GRP_TERRAIN_LM:
    if (strstr(gridname, "Band") != NULL)
    {
      char **fieldlist_tmp;
      char band[15];
      char *nf[] = { "Radiance", "RDQI", "DN", "Equivalent Reflectance", "Brf" };
      int nf_size = sizeof(nf)/sizeof(*nf);
      char *sp;
      int k, len;

      *nfields += nf_size;
      fieldlist_tmp = (char**)calloc(*nfields,sizeof(char*));
      if (fieldlist_tmp == NULL)
	MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

      /* Copy the native fields */
      for (i = 0; i < num_fields; ++i) {
	fieldlist_tmp[i] = (*fieldlist)[i];
      }

      /* Isolate the Band */
      strncpy(band, gridname, sizeof(band));
      if ((sp = strstr(band, "Band")) == NULL)
	MTK_ERR_CODE_JUMP(MTK_NULLPTR);
      *sp = '\0';

      /* Fill in unpacked/unscaled fields */
      for(i = 0; i < nf_size; i++) {
	k = i + num_fields;
	len = (int)(strlen(band) + strlen(nf[i]) + 2);
	fieldlist_tmp[k] = (char *)malloc(len *sizeof(char));
	if(fieldlist_tmp[k] == NULL)
	  MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
	strcpy(fieldlist_tmp[k], band);
	strcat(fieldlist_tmp[k], " ");
	strcat(fieldlist_tmp[k], nf[i]);
      }

      free(*fieldlist);
      *fieldlist = fieldlist_tmp;
    }
    break;
  case MTK_AS_LAND:
    {
      char **fieldlist_tmp;
      int j = 0;		/* field count */
      MTKt_DataBuffer sf = MTKT_DATABUFFER_INIT;
      char attrname[MAXSTR];
      int len;

      fieldlist_tmp = (char**)calloc(num_fields * 2,sizeof(char*));
      if (fieldlist_tmp == NULL)
	MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

      for (i = 0; i < num_fields; ++i) {
	fieldlist_tmp[j++] = (*fieldlist)[i];
	strncpy(attrname, "Scale ", MAXSTR);
	strncat(attrname, (*fieldlist)[i], MAXSTR);
	status = MtkGridAttrGetFid(Fid, gridname, attrname, &sf);
	if (status == MTK_SUCCESS) {
	  len = (int)(strlen("Raw ") + strlen((*fieldlist)[i]) + 1);
	  fieldlist_tmp[j] = (char *)malloc(len * sizeof(char));
	  if (fieldlist_tmp[j] == NULL)
	    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
	  strncpy(fieldlist_tmp[j], "Raw ", len);
      	  strncat(fieldlist_tmp[j++], (*fieldlist)[i], len);
	  MtkDataBufferFree(&sf);
	}
	if (((strncmp((*fieldlist)[i], "LandHDRF", MAXSTR) == 0) &&
	     (strncmp((*fieldlist)[i], "LandHDRFUnc", MAXSTR) != 0)) ||
	    (strncmp((*fieldlist)[i], "LandBRF", MAXSTR)  == 0) ||
	    (strncmp((*fieldlist)[i], "LAIDelta1", MAXSTR) ==0) ||
	    (strncmp((*fieldlist)[i], "LAIDelta2", MAXSTR) ==0)) {
	  len = (int)(strlen("Flag ") + strlen((*fieldlist)[i]) + 1);
	  fieldlist_tmp[j] = (char *)malloc(len * sizeof(char));
	  if (fieldlist_tmp[j] == NULL)
	    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
	  strncpy(fieldlist_tmp[j], "Flag ", len);
      	  strncat(fieldlist_tmp[j++], (*fieldlist)[i], len);
	}
      }
      *nfields = j;

      free(*fieldlist);
      *fieldlist = (char **)realloc((void *)fieldlist_tmp, 
				    *nfields * sizeof(char *));
      if (*fieldlist == NULL)
	MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
    }
    break;    
  case MTK_TC_CLOUD:
  {
    char **fieldlist_tmp;
    int j = 0;		/* field count */
    int len;
        

    fieldlist_tmp = (char**)calloc(num_fields * 2,sizeof(char*));
    if (fieldlist_tmp == NULL)
      MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

    for (i = 0; i < num_fields; ++i) {
      MTKt_DataBuffer scale = MTKT_DATABUFFER_INIT;
      fieldlist_tmp[j++] = (*fieldlist)[i];
      MtkFieldAttrGetFid(Fid, (*fieldlist)[i], "scale_factor", &scale);
      if (scale.nline) {         
    	  len = (int)(strlen("Raw ") + strlen((*fieldlist)[i]) + 1);
    	  fieldlist_tmp[j] = (char *)malloc(len * sizeof(char));
    	  if (fieldlist_tmp[j] == NULL)
    	    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
    	  strncpy(fieldlist_tmp[j], "Raw ", len);
          	  strncat(fieldlist_tmp[j++], (*fieldlist)[i], len);
      }
      MtkDataBufferFree(&scale);
    }
    *nfields = j;
    free(*fieldlist);
    *fieldlist = (char **)realloc((void *)fieldlist_tmp, 
			    *nfields * sizeof(char *));
    if (*fieldlist == NULL)
      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
    break;
  }  
  default:
    break;
  }

  return MTK_SUCCESS;

 ERROR_HANDLE:
  if (fieldlist != NULL)
    MtkStringListFree(num_fields, fieldlist);

  if (nfields != NULL)
    *nfields = -1;

  free(list);
  GDdetach(Gid);

  return status_code;
}

MTKt_status MtkFileGridToFieldListNcid(
  int ncid,            /**< [IN] netCDF file identifier */
  const char *gridname, /**< [IN] Gridname */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  return MtkFileGridToNativeFieldListNcid(ncid, gridname, nfields, fieldlist); // No derived fields
}
