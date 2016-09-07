/*===========================================================================
=                                                                           =
=                        MtkFileGridFieldToDimList                          =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */
#include <stdlib.h>

int main(int argc, char *argv[])
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int dimcnt = 0;               /* Dimension count */
  char **dimlist = NULL;        /* Dimension list */
  int *dimsize = NULL;          /* Dimension size */
  int i;

  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s <MISR Product File> <Grid Name> <Field Name>\n",argv[0]);
    exit(1);
  }

  status = MtkFileGridFieldToDimList(argv[1],argv[2],argv[3],&dimcnt,&dimlist,&dimsize);
  MTK_ERR_COND_JUMP(status);

  for (i = 0; i < dimcnt; ++i)
    printf("%-20s %3d\n",dimlist[i],dimsize[i]);

  MtkStringListFree(dimcnt, &dimlist);
  free(dimsize);

  return 0;

ERROR_HANDLE:
  if (status_code == MTK_HDFEOS_GDOPEN_FAILED)
    fprintf(stderr, "Error opening file: %s\n", argv[1]);

  if (status_code == MTK_HDFEOS_GDATTACH_FAILED)
    fprintf(stderr, "Failed to find grid: %s\n", argv[2]);

  if (status_code == MTK_HDFEOS_GDFIELDINFO_FAILED)
    fprintf(stderr, "Failed to find field: %s\n", argv[3]);

  return status_code;
}
