/*===========================================================================
=                                                                           =
=                           MtkFileToBlockRange                             =
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

int main(int argc, char *argv[])
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int start_block;              /* Start Block */
  int end_block;                /* End Block */

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <MISR Product File>\n",argv[0]);
    exit(1);
  }

  status = MtkFileToBlockRange(argv[1],&start_block,&end_block);
  MTK_ERR_COND_JUMP(status);

  printf("Start Block: %d  End Block: %d\n",start_block,end_block);

  return 0;

ERROR_HANDLE:
  fprintf(stderr, "Error opening file: %s\n", argv[1]);
  return status_code;
}
