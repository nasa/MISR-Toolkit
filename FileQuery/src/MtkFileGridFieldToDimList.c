/*===========================================================================
=                                                                           =
=                        MtkFileGridFieldToDimList                          =
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
#include <stdlib.h>

/** \brief Read dimension list of a particular field
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of dimensions for the field <tt> Blue Radiance/RDQI </tt> in the grid \c BlueBand in the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileGridFieldToDimList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", "BlueBand",
 *                "Blue Radiance/RDQI", &dimcnt, &dimlist, &dimsize);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \c dimlist and free() to free the memory used by \c dimsize  
 */

MTKt_status MtkFileGridFieldToDimList(
  const char *filename, /**< [IN] File name */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;		/* Return status. */
  MTKt_status status_code;      /* Return status of this function. */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 Fid = FAIL;             /* HDF-EOS File ID */

  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open HDF file for reading */
  hdfstatus = Fid = GDopen((char*)filename,DFACC_READ);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read dimension list. */
  status = MtkFileGridFieldToDimListFid(Fid, gridname, fieldname, 
					dimcnt, dimlist, dimsize);
  MTK_ERR_COND_JUMP(status);
  
  /* Close HDF file. */
  GDclose(Fid);
  
  return MTK_SUCCESS;

ERROR_HANDLE:
  GDclose(Fid);

  return status_code;
}

/** \brief Version of MtkFileGridFieldToDimList that takes an HDF-EOS file ID rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileGridFieldToDimListFid(
  int32 Fid,            /**< [IN] HDF-EOS File ID */
  const char *gridname, /**< [IN] Grid name */
  const char *fieldname,/**< [IN] Field name */
  int *dimcnt,          /**< [OUT] Dimension count */
  char **dimlist[],     /**< [OUT] Dimension list */
  int **dimsize         /**< [OUT] Dimension size */ )
{
  MTKt_status status;		/* Return status. */
  MTKt_status status_code;      /* Return status of this function. */
  intn hdfstatus;               /* HDF-EOS return status */
  int32 Gid = FAIL;             /* HDF-EOS Grid ID */
  int32 rank = 0;
  int32 dims[10];               /* Dimension sizes */
  int32 numbertype;
  char *list = NULL;            /* List of dimensions */
  int i;
  char *temp = NULL;
  char *basefield = NULL;	/* Base fieldname */
  int nextradims;               /* Number of extra dimensions */
  int *extradims = NULL;	/* Extra dimension list */

  /* Check Arguments */
  if (dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (dimsize == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *dimlist = NULL; /* Set output to NULL to prevent freeing unallocated
                      memory in case of error. */
  *dimsize = NULL;

  if (gridname == NULL || fieldname == NULL ||
      dimcnt == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Attach to grid */
  hdfstatus = Gid = GDattach(Fid,(char*)gridname);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDATTACH_FAILED);

  list = (char*)malloc(5000 * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  status = MtkParseFieldname(fieldname, &basefield, &nextradims, &extradims);
  MTK_ERR_COND_JUMP(status);

  /* Get field info */
  hdfstatus = GDfieldinfo(Gid, basefield, &rank, dims, &numbertype, list);
  if (hdfstatus == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDFIELDINFO_FAILED);

  *dimlist = (char**)calloc(rank,sizeof(char*));
  if (*dimlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
  *dimsize = (int*)calloc(rank,sizeof(int));
  if (*dimsize == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
    
  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    if ((strncmp(temp, "SOMBlockDim", sizeof(*temp))) != 0 &&
	(strncmp(temp, "XDim", sizeof(*temp))) != 0 &&
	(strncmp(temp, "YDim", sizeof(*temp))) != 0) {
      (*dimlist)[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
      if ((*dimlist)[i] == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
      strcpy((*dimlist)[i],temp);
      ++i;
    }
    temp = strtok(NULL,",");
  }

  /* Dimension count with out implied dimensions */
  *dimcnt = i;

  if (*dimcnt > 0) {

    /* Get dimension size info */
    for (i = 0; i < *dimcnt; ++i) {
      (*dimsize)[i] = GDdiminfo(Gid, (*dimlist)[i]);
    }

    /* Resize lists to account for removing dimensions */
    *dimlist = (char **)realloc((void *)*dimlist, *dimcnt * sizeof(char*));
    if (*dimlist == NULL)
      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
    *dimsize = (int *)realloc((void *)*dimsize, *dimcnt * sizeof(int));
    if (*dimsize == NULL)
      MTK_ERR_CODE_JUMP(MTK_REALLOC_FAILED);
  } else {
    MtkStringListFree(*dimcnt, dimlist);
    free(*dimsize);
    *dimsize = NULL;
  }

  free(basefield);
  free(extradims);
  free(list);
  GDdetach(Gid);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (dimlist != NULL && *dimlist != NULL)
    MtkStringListFree(*dimcnt, dimlist);

  if (dimsize != NULL)
    {
      free(*dimsize);
      *dimsize = NULL;
    }

  if (dimcnt != NULL)
    *dimcnt = -1;

  if (basefield != NULL) free(basefield);
  if (extradims != NULL) free(extradims);

  free(list);
  GDdetach(Gid);

  return status_code;
}
