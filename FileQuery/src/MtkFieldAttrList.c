/*===========================================================================
=                                                                           =
=                             MtkFieldAttrList                              =
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
#include "MisrError.h"
#include <mfhdf.h>
#include <HdfEosDef.h>
#include <stdlib.h>
#include <string.h>

/** \brief Get a list of field attributes
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get a list of field attributes in the \c CloudTopHeight field within the Stereo_1.1_km grid from the file \c MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf
 *
 *  \code
 *  status = MtkFieldAttrList("MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf", "Stereo_1.1_km", "CloudTopHeight", &num_attrs, &attrlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \a attrlist
 */

MTKt_status MtkFieldAttrList(
  const char *filename,    /**< [IN] File name */
  const char *fieldname,    /**< [IN] Field name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code; /* Return status of this function */
  MTKt_status status;      /* Return status */
  int32 hdfstatus;         /* HDF-EOS return status */
  int32 fid = FAIL;	   /* HDF-EOS File id */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF File */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Get list of field attributes. */
  status = MtkFieldAttrListFid(fid, fieldname, num_attrs, attrlist);
  MTK_ERR_COND_JUMP(status);

  /* Close HDF file */
  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fid != FAIL)
    GDclose(fid);

  return status_code;
}

/** \brief Version of MtkFieldAttrList that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFieldAttrListFid(
  int32 fid,               /**< [IN] HDF-EOS file identifier */
  const char *fieldname,    /**< [IN] Field name */
  int *num_attrs,          /**< [OUT] Number of attributes */
  char **attrlist[]        /**< [OUT] List of Attributes */ )
{
  MTKt_status status_code;     /* Return status of this function */
  int32 n_attrs = 0;           /* Number of attributes */
  int32 count;                 /* Number of values in attribute */
  int32 HDFfid = FAIL;         /* HDF File ID */
  int32 sdInterfaceID = FAIL;  /* SD Interface ID (file level) */
  int32 sd_id = FAIL;          /* SDS ID (field level) */
  int32 sds_index = 0;         /* SDS index/offset for field */  
  int32 rank;                  /* Number of dimensions for field */
  int32 data_type;             /* Field data type */
  char name[MAX_NC_NAME];      /* SDS field name */
  intn hdf_status;             /* HDF return status */
  char attr_name[MAX_NC_NAME]; /* Name of attribute */
  int32 attr_index = 0;        /* Index of attribute */
  int32 hdf_datatype;          /* Attribute data type */
  char **attrlist_tmp = NULL;  
  int32 dim_sizes[MAX_VAR_DIMS];
 
  if (fieldname == NULL || 
      num_attrs == NULL || attrlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Transform HDF-EOS fid to plain HDF fid and SDS fid for file */  
  EHidinfo(fid, &HDFfid, &sdInterfaceID);  
  
  sds_index = SDnametoindex(sdInterfaceID, fieldname);
  sd_id = SDselect(sdInterfaceID,sds_index);  
  SDgetinfo(sd_id, name, &rank, dim_sizes, &data_type, &n_attrs);
  
  attrlist_tmp = (char**)calloc(n_attrs,sizeof(char**));
  if (attrlist_tmp == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED); 
  
  for (attr_index = 0; attr_index < n_attrs; ++attr_index)
  {
    /* Get attribute information */
    hdf_status = SDattrinfo(sd_id, attr_index, attr_name, &hdf_datatype, &count);
    if (hdf_status == FAIL)
      MTK_ERR_CODE_JUMP(MTK_HDF_SDATTRINFO_FAILED);

    attrlist_tmp[attr_index] = (char*)malloc((strlen(attr_name) + 1) * sizeof(char));
    if (attrlist_tmp[attr_index] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

    strcpy(attrlist_tmp[attr_index],attr_name);
  }
  *attrlist = attrlist_tmp;
  *num_attrs = n_attrs;
  return MTK_SUCCESS;
  
  
ERROR_HANDLE:
  if (attrlist_tmp != NULL)
    MtkStringListFree(n_attrs, &attrlist_tmp);

  if (num_attrs != NULL)
    *num_attrs = -1;

  return status_code;
}
