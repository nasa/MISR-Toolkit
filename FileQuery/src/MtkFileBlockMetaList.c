/*===========================================================================
=                                                                           =
=                          MtkFileBlockMetaList                             =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <hdf.h>
#include <string.h>
#include <stdlib.h>


/** \brief List block metadata structures
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get a list of the block metadata structures in the file \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileBlockMetaList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &nblockmeta, &blockmetalist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a blockmetalist
 */

MTKt_status MtkFileBlockMetaList(
  const char *filename, /**< [IN] File name */
  int *nblockmeta, /**< [OUT] Number of Block Metadata */
  char ***blockmetalist /**< [OUT] Block Metadata List */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  intn hdf_status;
  int32 file_id = FAIL;
  
  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  /* Open HDF File */
  file_id = HDFopen(filename, DFACC_READ, 0);
  if (file_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFOPEN_FAILED);

  /* Read list of block metadata structures. */
  status = MtkFileBlockMetaListFid(file_id, nblockmeta, blockmetalist);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */
  hdf_status = HDFclose(file_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFCLOSE_FAILED);
  file_id = FAIL;
    
  return MTK_SUCCESS;
  
ERROR_HANDLE:
  if (file_id != FAIL)
    HDFclose(file_id);
  
  return status_code;
}

/** \brief Version of MtkFileBlockMetaList that takes an HDF file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileBlockMetaListFid(
  int32 file_id,   /**< [IN] HDF file identifier */
  int *nblockmeta, /**< [OUT] Number of Block Metadata */
  char ***blockmetalist /**< [OUT] Block Metadata List */ )
{
  MTKt_status status_code; /* Return status of this function */
  /*MTKt_status status;*/      /* Return status */
  intn hdf_status;
  int32 vdata_ref = FAIL;
  int32 vdata_id = FAIL;
  char vdata_name[VSNAMELENMAX + 1];
  char **temp_list = NULL;
  int temp_count = 0;
  int temp_list_max = 5;
  int i;
  int vstart_active = 0;
  
  if (nblockmeta == NULL || blockmetalist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  /* Initialize the vdata interface */
  hdf_status = Vstart(file_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSTART_FAILED);
  vstart_active = 1;
  
  temp_list = (char**)calloc(temp_list_max, sizeof(char*));
  if (temp_list == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
        
  /* Search through file for vdatas */
  vdata_ref = VSgetid(file_id, vdata_ref);
  while (vdata_ref != FAIL)
  {
    vdata_id = VSattach(file_id, vdata_ref, "r");
    if (vdata_id == FAIL)
      MTK_ERR_CODE_JUMP(MTK_HDF_VSATTACH_FAILED);
    
    hdf_status = VSgetname(vdata_id, vdata_name);
    if (hdf_status == FAIL)
      MTK_ERR_CODE_JUMP(MTK_HDF_VSGETNAME_FAILED);
    
    if (strstr(vdata_name,"Metadata") != NULL && strstr(vdata_name,"Block") != NULL)
    {
      if (temp_list_max == temp_count)
      {
        temp_list_max += 5;
	    temp_list = (char**)realloc(temp_list,temp_list_max);
	    if (temp_list == NULL)
	      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
      }

      temp_list[temp_count] = (char*)malloc((strlen(vdata_name) + 1) *
                                             sizeof(char));
      if (temp_list[temp_count] == NULL)
        MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

      strcpy(temp_list[temp_count],vdata_name);
      ++temp_count;
    }
  
    hdf_status = VSdetach(vdata_id);
    if (hdf_status == FAIL)
      MTK_ERR_CODE_JUMP(MTK_HDF_VSDETACH_FAILED);
    vdata_id = FAIL;
  
    vdata_ref = VSgetid(file_id, vdata_ref);
  }
  
  /* End access to the vdata interface */   
  hdf_status = Vend(file_id);
  if (hdf_status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VEND_FAILED);
  vstart_active = 0;
  
  /* Copy list to output arguments */
  *nblockmeta = temp_count;

  *blockmetalist = (char**)malloc(temp_count * sizeof(char*));
  if (*blockmetalist == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  for (i = 0; i < temp_count; ++i)
    (*blockmetalist)[i] = temp_list[i];

  free(temp_list);

  return MTK_SUCCESS;
  
ERROR_HANDLE:
  if (vdata_id != FAIL)
    VSdetach(vdata_id);

  if (vstart_active)
    Vend(file_id);

  MtkStringListFree(temp_count, &temp_list);
  
  return status_code;
}
