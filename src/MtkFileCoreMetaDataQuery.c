/*===========================================================================
=                                                                           =
=                         MtkFileCoreMetaDataQuery                          =
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
  int nparam;                   /* Number of parameters */
  char **paramlist;             /* Parameter list */
  int i;

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <MISR Product File>\n",argv[0]);
    exit(1);
  }

  status = MtkFileCoreMetaDataQuery(argv[1],&nparam,&paramlist);
  MTK_ERR_COND_JUMP(status);

  for (i = 0; i < nparam; ++i)
    printf("%s\n",paramlist[i]);

  MtkStringListFree(nparam, &paramlist);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Error opening file: %s\n", argv[1]);
  return status_code;
}
