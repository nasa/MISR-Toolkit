/*===========================================================================
=                                                                           =
=                           MtkFileCoreMetaDataGet                          =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "odldef.h"
#include "odlinter.h"
#include <errno.h>
#include <stdlib.h>

/** \brief Get core metadata parameter
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the value for \c LOCALGRANULEID from the core metadata in the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  MtkCoreMetaData metadata = MTK_CORE_METADATA_INIT;
 *  status = MtkFileCoreMetaDataGet("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "LOCALGRANULEID", &metadata);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkCoreMetaDataFree() to free the memory used by \c metadata
 */

MTKt_status MtkFileCoreMetaDataGet(
  const char *filename, /**< [IN] File name*/
  const char *param, /**< [IN] Parameter */
  MtkCoreMetaData *metadata  /**< [OUT] Core metadata */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileCoreMetaDataGetNC(filename, param, metadata); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileCoreMetaDataGetHDF(filename, param, metadata); // try HDF
}

MTKt_status MtkFileCoreMetaDataGetNC(
  const char *filename, /**< [IN] File name*/
  const char *param, /**< [IN] Parameter */
  MtkCoreMetaData *metadata  /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */

  if (filename == NULL) return MTK_NULLPTR;

  /* Open file */
  int ncid = 0;
  {
    int nc_status = nc_open(filename, NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_OPEN_FAILED);
  }

  /* Read grid attribute */
  status = MtkFileCoreMetaDataGetNcid(ncid, param, metadata);
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

MTKt_status MtkFileCoreMetaDataGetHDF(
  const char *filename, /**< [IN] File name*/
  const char *param, /**< [IN] Parameter */
  MtkCoreMetaData *metadata  /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;
  int32 hdf_status;        /* HDF-EOS return status */
  int32 sd_id = FAIL;      /* HDF SD file identifier. */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  hdf_status = sd_id = SDstart(filename, DFACC_READ);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDSTART_FAILED);

  /* Read coremetadata. */
  status = MtkFileCoreMetaDataGetFid(sd_id, param, metadata);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF File */
  hdf_status = SDend(sd_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_SDEND_FAILED);
  sd_id = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (sd_id != FAIL)
    SDend(sd_id);

  return status_code;
}

/** \brief Version of MtkFileCoreMetaDataGet that takes an HDF SD identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileCoreMetaDataGetFid(
  int32 sd_id,       /**< [IN] HDF SD file identifier */
  const char *param, /**< [IN] Parameter */
  MtkCoreMetaData *metadata  /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;
  char *coremeta = NULL;
  FILE *temp = NULL;
#ifdef _WIN32
  char *temp_file_name;
#endif
  MtkCoreMetaData md = MTK_CORE_METADATA_INIT;

  AGGREGATE aggNode = NULL;
  GROUP grpNode = NULL;
  OBJECT objNode = NULL;
  PARAMETER parmNode = NULL;
    VALUE valueNode = NULL;
  char **strPtr1 = NULL;
  int *intPtr = NULL;
  double *dblPtr = NULL;
  int odlRetVal = 0;  /* ODl function return value */

  if (param == NULL || metadata == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* clear the errno for the ODL routines */
  if (errno == ERANGE)
  {
      errno = 0;
  }
    
  /* attributes are contained in a separate file */
  aggNode = NewAggregate(aggNode, KA_GROUP, "locAggr", "");
  if (aggNode == NULL)
  {
    printf("Unable to create odl aggregate locAggr\n");
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  /* Read core metadata from HDF file */
  status = MtkFileCoreMetaDataRawFid(sd_id, &coremeta);
  MTK_ERR_COND_JUMP(status);

  /* ODL parser requires input from a FILE */
  #ifdef _WIN32
     /* On Win32 tmpfile() creates a file in the root directory, */
     /* the user may not have write access.*/
     temp_file_name = _tempnam(NULL, NULL);
     temp = fopen(temp_file_name, "w+");
  #else
     temp = tmpfile();
  #endif
  
  if (temp == NULL)
     MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fprintf(temp,"%s",coremeta);
  rewind(temp);

  free(coremeta);
  coremeta = NULL;

  odlRetVal = ReadLabel(temp, aggNode);
  if (odlRetVal != 1)
  {
    printf("Unable to convert to an odl structure\n");
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  fclose(temp);
  temp = NULL;

  /* extract the values and dump in the buffer provided */
  grpNode = FindGroup(aggNode, "INVENTORYMETADATA");
  if (grpNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  objNode = FindObject(grpNode, (char *)param, NULL);
  if (objNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  parmNode = FindParameter(objNode, "VALUE"); 
  if (parmNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  valueNode = FirstValue(parmNode);
 
  /* copy the value string(s) into user provided string */
  if (valueNode->item.type == TV_STRING  || valueNode->item.type == TV_SYMBOL)
  {
    md.data.s = (char**)calloc(valueNode->parameter->value_count,sizeof(char*));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.s == NULL)
       MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
            
    md.dataptr = md.data.s;
    md.datatype = MTKMETA_CHAR;
    strPtr1 = md.data.s;
    while (valueNode != NULL)
    {
      *strPtr1 = (char*)malloc((strlen(valueNode->item.value.string) + 1) * sizeof(char));
      if (*strPtr1 == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

      strcpy(*strPtr1, valueNode->item.value.string);
      strPtr1++;
      valueNode = NextValue(valueNode);
    }
  }   
  else if (valueNode->item.type == TV_INTEGER)
  {
    md.data.i = (int*)malloc(valueNode->parameter->value_count * sizeof(int));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.i == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
            
    md.dataptr = md.data.i;
    md.datatype = MTKMETA_INT;
    intPtr = md.data.i;
    while (valueNode != NULL)
    {
      *intPtr = (int)valueNode->item.value.integer.number; 
      intPtr++;
      valueNode = NextValue(valueNode);
    }
  }
  else 
  {    
    md.data.d = (double*)malloc(valueNode->parameter->value_count * sizeof(double));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.d == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

    md.dataptr = md.data.d;
    md.datatype = MTKMETA_DOUBLE;
    dblPtr = md.data.d;
    while (valueNode != NULL)
    {
      *dblPtr = (double)valueNode->item.value.real.number; 
      dblPtr++;
      valueNode = NextValue(valueNode);
    }
  }
  
  /* Destroy tree */
  RemoveAggregate(aggNode);

  /* Copy data to output */
  metadata->data.s = md.data.s;
  metadata->num_values = md.num_values;
  metadata->datatype = md.datatype;
  metadata->dataptr = md.dataptr;

  return MTK_SUCCESS;

ERROR_HANDLE:
  RemoveAggregate(aggNode);

  if (temp != NULL)
    fclose(temp);

  if (coremeta != NULL)
    free(coremeta);

  MtkCoreMetaDataFree(&md);

  return status_code;
}

MTKt_status MtkFileCoreMetaDataGetNcid(
  int ncid,               /**< [IN] netCDF File ID */
  const char *param, /**< [IN] Parameter */
  MtkCoreMetaData *metadata  /**< [OUT] Core metadata */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;
  char *coremeta = NULL;
  FILE *temp = NULL;
#ifdef _WIN32
  char *temp_file_name;
#endif
  MtkCoreMetaData md = MTK_CORE_METADATA_INIT;

  AGGREGATE aggNode = NULL;
  GROUP grpNode = NULL;
  OBJECT objNode = NULL;
  PARAMETER parmNode = NULL;
    VALUE valueNode = NULL;
  char **strPtr1 = NULL;
  int *intPtr = NULL;
  double *dblPtr = NULL;
  int odlRetVal = 0;  /* ODl function return value */

  if (param == NULL || metadata == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* clear the errno for the ODL routines */
  if (errno == ERANGE)
  {
      errno = 0;
  }
    
  /* attributes are contained in a separate file */
  aggNode = NewAggregate(aggNode, KA_GROUP, "locAggr", "");
  if (aggNode == NULL)
  {
    printf("Unable to create odl aggregate locAggr\n");
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  /* Read core metadata from HDF file */
  status = MtkFileCoreMetaDataRawNcid(ncid, &coremeta);
  MTK_ERR_COND_JUMP(status);

  /* ODL parser requires input from a FILE */
  #ifdef _WIN32
     /* On Win32 tmpfile() creates a file in the root directory, */
     /* the user may not have write access.*/
     temp_file_name = _tempnam(NULL, NULL);
     temp = fopen(temp_file_name, "w+");
  #else
     temp = tmpfile();
  #endif
  
  if (temp == NULL)
     MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  fprintf(temp,"%s",coremeta);
  rewind(temp);

  free(coremeta);
  coremeta = NULL;

  odlRetVal = ReadLabel(temp, aggNode);
  if (odlRetVal != 1)
  {
    printf("Unable to convert to an odl structure\n");
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  fclose(temp);
  temp = NULL;

  /* extract the values and dump in the buffer provided */
  grpNode = FindGroup(aggNode, "INVENTORYMETADATA");
  if (grpNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  objNode = FindObject(grpNode, (char *)param, NULL);
  if (objNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  parmNode = FindParameter(objNode, "VALUE"); 
  if (parmNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }

  valueNode = FirstValue(parmNode);
 
  /* copy the value string(s) into user provided string */
  if (valueNode->item.type == TV_STRING  || valueNode->item.type == TV_SYMBOL)
  {
    md.data.s = (char**)calloc(valueNode->parameter->value_count,sizeof(char*));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.s == NULL)
       MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
            
    md.dataptr = md.data.s;
    md.datatype = MTKMETA_CHAR;
    strPtr1 = md.data.s;
    while (valueNode != NULL)
    {
      *strPtr1 = (char*)malloc((strlen(valueNode->item.value.string) + 1) * sizeof(char));
      if (*strPtr1 == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

      strcpy(*strPtr1, valueNode->item.value.string);
      strPtr1++;
      valueNode = NextValue(valueNode);
    }
  }   
  else if (valueNode->item.type == TV_INTEGER)
  {
    md.data.i = (int*)malloc(valueNode->parameter->value_count * sizeof(int));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.i == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
            
    md.dataptr = md.data.i;
    md.datatype = MTKMETA_INT;
    intPtr = md.data.i;
    while (valueNode != NULL)
    {
      *intPtr = (int)valueNode->item.value.integer.number; 
      intPtr++;
      valueNode = NextValue(valueNode);
    }
  }
  else 
  {    
    md.data.d = (double*)malloc(valueNode->parameter->value_count * sizeof(double));
    md.num_values = valueNode->parameter->value_count;
    if (md.data.d == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

    md.dataptr = md.data.d;
    md.datatype = MTKMETA_DOUBLE;
    dblPtr = md.data.d;
    while (valueNode != NULL)
    {
      *dblPtr = (double)valueNode->item.value.real.number; 
      dblPtr++;
      valueNode = NextValue(valueNode);
    }
  }
  
  /* Destroy tree */
  RemoveAggregate(aggNode);

  /* Copy data to output */
  metadata->data.s = md.data.s;
  metadata->num_values = md.num_values;
  metadata->datatype = md.datatype;
  metadata->dataptr = md.dataptr;

  return MTK_SUCCESS;

ERROR_HANDLE:
  RemoveAggregate(aggNode);

  if (temp != NULL)
    fclose(temp);

  if (coremeta != NULL)
    free(coremeta);

  MtkCoreMetaDataFree(&md);

  return status_code;
}
