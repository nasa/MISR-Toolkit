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

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */
#include <stdlib.h>

int main(int argc, char *argv[])
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int nfields;                   /* Number of fields */
  char **fieldlist;              /* Field list */
  int i;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <MISR Product File> <Grid Name>\n",argv[0]);
    exit(1);
  }

  status = MtkFileGridToFieldList(argv[1],argv[2],&nfields,&fieldlist);
  MTK_ERR_COND_JUMP(status);

  for (i = 0; i < nfields; ++i)
    printf("%s\n",fieldlist[i]);

  MtkStringListFree(nfields, &fieldlist);

  return 0;

ERROR_HANDLE:
  if (status_code == MTK_HDFEOS_GDOPEN_FAILED)
    fprintf(stderr, "Error opening file: %s\n", argv[1]);

  if (status_code == MTK_HDFEOS_GDATTACH_FAILED)
    fprintf(stderr, "Failed to find grid: %s\n", argv[2]);

  return status_code;
}
