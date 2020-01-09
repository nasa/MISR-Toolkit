/*===========================================================================
=                                                                           =
=                          MtkNCVarId                                       =
=                                                                           =
=============================================================================

            Copyright 2017, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#define _GNU_SOURCE

#include "MisrUtil.h"
#include <netcdf.h>  // Definition of nc_inq_varid
#include <string.h>  // Definition of strcspn
#include <stdlib.h>  // Definition of calloc
#include <stdio.h>

MTKt_status MtkNCVarId(int Ncid, const char *Name, MTKt_ncvarid *Var)
{
  MTKt_status status_code;      /* Return status code for error macros */
  const char *p = Name;
  size_t len;
  Var->gid = Ncid;

  while (strlen(p) != (len = strcspn(p, "/"))) {

    char *group_name = calloc(len+1, sizeof(char));
    strncpy(group_name, p, len);

    int nc_status = nc_inq_grp_ncid(Var->gid, group_name, &(Var->gid));
    free(group_name);
    if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

    p += len + 1;
  }

  int nc_status = nc_inq_varid(Var->gid, p, &(Var->varid));
  if (nc_status != NC_NOERR) MTK_ERR_CODE_JUMP(MTK_NETCDF_READ_FAILED);

  return MTK_SUCCESS;
  ERROR_HANDLE:
  return status_code;
}
