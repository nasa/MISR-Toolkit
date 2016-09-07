/*===========================================================================
=                                                                           =
=                         MtkFileBlockMetaFieldList                         =
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

/** \brief Read list of fields in a block metadata structure
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of fields in the \c PerBlockMetadataCommon structure in the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileBlockMetaFieldList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "PerBlockMetadataCommon", &nfields, &fieldlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a fieldlist
 */

MTKt_status MtkFileBlockMetaFieldList(
  const char *filename, /**< [IN] Filename */
  const char *blockmetaname, /**< [IN] Block metadata structure name */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status_code;	/* Return status of this function. */
  MTKt_status status;      /* Return status */
  intn hdfstatus;		    /* HDF-EOS return status */
  int32 file_id = FAIL;

  /* Check Arguments */
  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  /* Open HDF File */
  file_id = HDFopen(filename, DFACC_READ, 0);
  if (file_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_OPEN_FAILED);

  /* Read list of fields in a block metadata structure */
  status = MtkFileBlockMetaFieldListFid(file_id, blockmetaname, 
					nfields, fieldlist);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */
  hdfstatus = HDFclose(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_HDFCLOSE_FAILED);
  file_id = FAIL;
  
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (file_id != FAIL)
    HDFclose(file_id);

  return status_code;
}

/** \brief Version of MtkFileBlockMetaFieldList that takes an HDF file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileBlockMetaFieldListFid(
  int32 file_id,             /**< [IN] HDF file identifier */
  const char *blockmetaname, /**< [IN] Block metadata structure name */
  int *nfields, /**< [OUT] Number of Fields */
  char **fieldlist[] /**< [OUT] List of Fields */ )
{
  MTKt_status status_code;	/* Return status of this function. */
  intn hdfstatus;		    /* HDF-EOS return status */
  int32 vdata_ref = FAIL;
  int32 vdata_id = FAIL;
  int32 num_fields = 0;     /* Number of fields */
  char *list = NULL;        /* List of fields */
  int i;
  char *temp = NULL;
  int vstart_active = 0;

  /* Check Arguments */
  if (blockmetaname == NULL ||
      nfields == NULL || fieldlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  
  *fieldlist = NULL;

  /* Initialize the vdata interface */
  hdfstatus = Vstart(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSFIND_FAILED);
  vstart_active = 1;
  
  /* Find reference number of block metadata structure */
  vdata_ref = VSfind(file_id, blockmetaname);
  if (vdata_ref == 0) /* Failure */
    MTK_ERR_CODE_JUMP(MTK_HDF_VSFIND_FAILED);
  
  vdata_id = VSattach(file_id, vdata_ref, "r");
  if (vdata_id == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSATTACH_FAILED);
  
  /* Determine number of fields in block metadata structure */
  num_fields = VFnfields(vdata_id);
  if (num_fields == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VFNFIELDS_FAILED);
  
  list = (char*)malloc((VSNAMELENMAX + 1) * num_fields * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Get list of fields */
  hdfstatus = VSgetfields(vdata_id, list);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSGETFIELDS_FAILED);
  
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
  list = NULL;
  temp = NULL;
  
  hdfstatus = VSdetach(vdata_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VSDETACH_FAILED);

  /* End access to the vdata interface */ 
  hdfstatus = Vend(file_id);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDF_VEND_FAILED);
  vstart_active = 0;
    
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (vdata_id != FAIL)
    VSdetach(vdata_id);

  if (vstart_active)
    Vend(file_id);

  free(list);

  if (fieldlist != NULL && *fieldlist != NULL)
    MtkStringListFree(num_fields,&*fieldlist);

  return status_code;
}
