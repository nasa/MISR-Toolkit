/*===========================================================================
=                                                                           =
=                         MtkFileCoreMetaDataQuery                          =
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

/** \brief Query file for core metadata
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we query the core metadata in the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileCoreMetaDataQuery("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &nparam, &paramlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \c paramlist
 */

MTKt_status MtkFileCoreMetaDataQuery(
  const char *filename, /**< [IN] File name */
  int *nparam, /**< [OUT] Number of parameters */
  char ***paramlist /**< [OUT] Parameter list */ )
{
  MTKt_status status;      /* Return status */

  status = MtkFileCoreMetaDataQueryNC(filename, nparam, paramlist); // try netCDF
  if (status != MTK_NETCDF_OPEN_FAILED) return status;

  return MtkFileCoreMetaDataQueryHDF(filename, nparam, paramlist); // try HDF
}

MTKt_status MtkFileCoreMetaDataQueryNC(
  const char *filename, /**< [IN] File name */
  int *nparam, /**< [OUT] Number of parameters */
  char ***paramlist /**< [OUT] Parameter list */ )
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
  status = MtkFileCoreMetaDataQueryNcid(ncid, nparam, paramlist);
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

MTKt_status MtkFileCoreMetaDataQueryHDF(
  const char *filename, /**< [IN] File name */
  int *nparam, /**< [OUT] Number of parameters */
  char ***paramlist /**< [OUT] Parameter list */ )
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
  status = MtkFileCoreMetaDataQueryFid(sd_id, nparam, paramlist);
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

/** \brief Version of MtkFileCoreMetaDataQuery that takes an HDF SD file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileCoreMetaDataQueryFid(
  int32 sd_id, /**< [IN] HDF SD file identifier */
  int *nparam, /**< [OUT] Number of parameters */
  char ***paramlist /**< [OUT] Parameter list */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;
  char *coremeta = NULL;
  FILE *temp = NULL;
#ifdef _WIN32
  char *temp_file_name;
#endif
  char **temp_list = NULL;
  int temp_count = 0;
  int temp_list_max = 50;
  int i;

  AGGREGATE aggNode = NULL;
  GROUP grpNode = NULL;
  int odlRetVal = 0;  /* ODl function return value */
  AGGREGATE base_node;
  AGGREGATE  node;                    /* Pointer to current node           */

  if (nparam == NULL || paramlist == NULL)
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

  /* extract the values and dump in the buffer provided */
  grpNode = FindGroup(aggNode, "INVENTORYMETADATA");
  if (grpNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }
    
  /* Start searching with the base node and stop searching when we
     have visited all of the progeny of the base node  */

  temp_list = (char**)calloc(temp_list_max, sizeof(char*));
  if (temp_list == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  node = base_node = aggNode;

  while (node != NULL)
  {
    if (node->kind == KA_OBJECT)
    {
      if (temp_list_max == temp_count)
      {
        temp_list_max += 50;
	temp_list = (char**)realloc(temp_list,temp_list_max);
	if (temp_list == NULL)
	  MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
      }

      temp_list[temp_count] = (char*)malloc((strlen(node->name) + 1) *
                                             sizeof(char));
      if (temp_list[temp_count] == NULL)
        MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

      strcpy(temp_list[temp_count],node->name);
      ++temp_count;
    }
   
    node = NextSubObject (base_node, node);
  }

  RemoveAggregate(aggNode);

  /* Copy list to output arguments */
  *nparam = temp_count;

  *paramlist = (char**)malloc(temp_count * sizeof(char*));
  if (*paramlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (i = 0; i < temp_count; ++i)
    (*paramlist)[i] = temp_list[i];

  free(temp_list);

  return MTK_SUCCESS;

ERROR_HANDLE:
  RemoveAggregate(aggNode);

  if (temp != NULL)
    fclose(temp);

  if (coremeta != NULL)
    free(coremeta);

  if (temp_list != NULL)
  {
    for (i = 0; i < temp_count; ++i)
      free(temp_list[i]);

    free(temp_list);
  }

  return status_code;
}

MTKt_status MtkFileCoreMetaDataQueryNcid(
  int ncid,               /**< [IN] netCDF File ID */
  int *nparam, /**< [OUT] Number of parameters */
  char ***paramlist /**< [OUT] Parameter list */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;
  char *coremeta = NULL;
  FILE *temp = NULL;
#ifdef _WIN32
  char *temp_file_name;
#endif
  char **temp_list = NULL;
  int temp_count = 0;
  int temp_list_max = 50;
  int i;

  AGGREGATE aggNode = NULL;
  GROUP grpNode = NULL;
  int odlRetVal = 0;  /* ODl function return value */
  AGGREGATE base_node;
  AGGREGATE  node;                    /* Pointer to current node           */

  if (nparam == NULL || paramlist == NULL)
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

  /* extract the values and dump in the buffer provided */
  grpNode = FindGroup(aggNode, "INVENTORYMETADATA");
  if (grpNode == NULL)
  {
    MTK_ERR_CODE_JUMP(MTK_FAILURE);
  }
    
  /* Start searching with the base node and stop searching when we
     have visited all of the progeny of the base node  */

  temp_list = (char**)calloc(temp_list_max, sizeof(char*));
  if (temp_list == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  node = base_node = aggNode;

  while (node != NULL)
  {
    if (node->kind == KA_OBJECT)
    {
      if (temp_list_max == temp_count)
      {
        temp_list_max += 50;
	temp_list = (char**)realloc(temp_list,temp_list_max);
	if (temp_list == NULL)
	  MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
      }

      temp_list[temp_count] = (char*)malloc((strlen(node->name) + 1) *
                                             sizeof(char));
      if (temp_list[temp_count] == NULL)
        MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

      strcpy(temp_list[temp_count],node->name);
      ++temp_count;
    }
   
    node = NextSubObject (base_node, node);
  }

  RemoveAggregate(aggNode);

  /* Copy list to output arguments */
  *nparam = temp_count;

  *paramlist = (char**)malloc(temp_count * sizeof(char*));
  if (*paramlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (i = 0; i < temp_count; ++i)
    (*paramlist)[i] = temp_list[i];

  free(temp_list);

  return MTK_SUCCESS;

ERROR_HANDLE:
  RemoveAggregate(aggNode);

  if (temp != NULL)
    fclose(temp);

  if (coremeta != NULL)
    free(coremeta);

  if (temp_list != NULL)
  {
    for (i = 0; i < temp_count; ++i)
      free(temp_list[i]);

    free(temp_list);
  }

  return status_code;
}
