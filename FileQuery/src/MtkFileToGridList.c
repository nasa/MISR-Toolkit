/*===========================================================================
=                                                                           =
=                            MtkFileToGridList                              =
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
#include <stdio.h>

/** \brief Read list of grids from a file
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we read the list of grids from the file
 *  \c MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf
 *
 *  \code
 *  status = MtkFileToGridList("MISR_AM1_GRP_TERRAIN_GM_P161_O012115_DF_F03_0021.hdf", &ngrids, &gridlist);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkStringListFree() to free the memory used by \c gridlist  
 */

MTKt_status MtkFileToGridList(
  const char *filename, /**< [IN]  File name */
  int *ngrids,          /**< [OUT] Number of grids */
  char **gridlist[]     /**< [OUT] Grid list */ )
{
  MTKt_status status_code;      /* Return status of this function. */
  MTKt_status status;      	/* Return status */
  intn hdfstatus;		/* HDF return status */
  int32 fid = FAIL;		/* HDF-EOS file identifier */

  /* Check Arguments */
  if (filename == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Open file. */
  fid = GDopen((char *)filename, DFACC_READ);
  if (fid == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDOPEN_FAILED);

  /* Read the list of grids. */
  status = MtkFileToGridListFid(fid, ngrids, gridlist);
  MTK_ERR_COND_JUMP(status);

  /* Close file. */
  hdfstatus = GDclose(fid);
  if (hdfstatus == FAIL) MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDCLOSE_FAILED);
  fid = FAIL;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (fid != FAIL) GDclose(fid);
  return status_code;
}

/** \brief Version of MtkFileToGridList that takes an HDF-EOS file identifier rather than a filename.
 *
 *  \return MTK_SUCCESS if successful.
 */

MTKt_status MtkFileToGridListFid(
  int32 fid,            /**< [IN]  HDF-EOS file identifier */
  int *ngrids,          /**< [OUT] Number of grids */
  char **gridlist[]     /**< [OUT] Grid list */ )
{
  MTKt_status status_code;      /* Return status of this function. */
  intn status;			/* HDF return status */
  int32 strbufsize;             /* String length of grid list */
  int32 num_grids = 0;          /* Number of grids */
  char *list = NULL;            /* List of grids */
  char *temp = NULL;
  int i;

  /* Check Arguments */
  if (gridlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  *gridlist = NULL; /* Set output to NULL to prevent freeing unallocated
                       memory in case of error. */

  if (ngrids == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Get the size of the buffer */
  status = GDinqgridfid(fid, NULL, &strbufsize);
  if (status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQGRID_FAILED);

  /* GDinqgrid does not account for null termination, */
  /* so we add 1 to strbufsize */
  list = (char*)malloc((strbufsize + 1) * sizeof(char));
  if (list == NULL)
    MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);

  /* Get list of grids */
  status = num_grids = GDinqgridfid(fid, list, &strbufsize);
  if (status == FAIL)
    MTK_ERR_CODE_JUMP(MTK_HDFEOS_GDINQGRID_FAILED);

  *ngrids = num_grids;
  *gridlist = (char**)calloc(num_grids, sizeof(char*));
  if (*gridlist == NULL)
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);

  temp = strtok(list,",");
  i = 0;
  while (temp != NULL)
  {
    (*gridlist)[i] = (char*)malloc((strlen(temp) + 1) * sizeof(char));
    if ((*gridlist)[i] == NULL)
      MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
    strcpy((*gridlist)[i],temp);
    temp = strtok(NULL,",");
    ++i;
  }

  free(list);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (gridlist != NULL)
    MtkStringListFree(num_grids, gridlist);

  if (ngrids != NULL)
    *ngrids = -1;

  free(list);

  return status_code;
}

/** \brief Replacement for HDF-EOS GDinqgrid that takes an HDF-EOS file id rather than a filename.
 *
 *  \note
 *  Code is copied from the HDF-EOS source file EHapi.c, function EHinquire.
 *
 *  \return MTK_SUCCESS if successful.
 */

int32
GDinqgridfid(int32 fid, char *objectlist, int32 * strbufsize)
{
    int32           vgRef;	/* Vgroup reference number */
    int32           vGrpID;	/* Vgroup ID */
    int32           nobj = 0;	/* Number of HDFEOS objects in file */
    int32           slen;	/* String length */

    char            name[FIELDNAMELENMAX];	/* Object name */
    char            class[FIELDNAMELENMAX];	/* Object class */
    int32           HDFfid;  	/* HDF file identifier */
    int32   	    sid;        /* HDF SD identifier (not used) */
    intn            hdf_status_code; 


    /* Get HDF file identifier. */
    hdf_status_code = EHidinfo(fid, &HDFfid, &sid);
    if (hdf_status_code == FAIL)
      return FAIL;

    /* Start Vgroup Interface */
    /* ---------------------- */
    Vstart(HDFfid);


    /* If string buffer size is requested then zero out counter */
    /* -------------------------------------------------------- */
    if (strbufsize != NULL)
    {
	*strbufsize = 0;
    }
    /* Search for objects from begining of HDF file */
    /* -------------------------------------------- */
    vgRef = -1;

    /* Loop through all objects */
    /* ------------------------ */
    while (1)
    {
	/* Get Vgroup reference number */
	/* --------------------------- */
	vgRef = Vgetid(HDFfid, vgRef);

	/* If no more then exist search loop */
	/* --------------------------------- */
	if (vgRef == -1)
	{
	    break;
	}
	/* Get Vgroup ID, name, and class */
	/* ------------------------------ */
	vGrpID = Vattach(HDFfid, vgRef, "r");
	Vgetname(vGrpID, name);
	Vgetclass(vGrpID, class);


	/* If object of desired type (GRID) ... */
	/* -------------------------------------------------- */
	if (strcmp(class, "GRID") == 0)
	{

	    /* Increment counter */
	    /* ----------------- */
	    nobj++;


	    /* If object list requested add name to list */
	    /* ----------------------------------------- */
	    if (objectlist != NULL)
	    {
		if (nobj == 1)
		{
		    strcpy(objectlist, name);
		} else
		{
		    strcat(objectlist, ",");
		    strcat(objectlist, name);
		}
	    }
	    /* Compute string length of object entry */
	    /* ------------------------------------- */
	    slen = (nobj == 1) ? strlen(name) : strlen(name) + 1;


	    /* If string buffer size is requested then increment buffer size */
	    /* ------------------------------------------------------------- */
	    if (strbufsize != NULL)
	    {
		*strbufsize += slen;
	    }
	}
	/* Detach Vgroup */
	/* ------------- */
	Vdetach(vGrpID);
    }

    /* "Close" Vgroup interface and HDFEOS file */
    /* ---------------------------------------- */
    Vend(HDFfid);

    return (nobj);
}

