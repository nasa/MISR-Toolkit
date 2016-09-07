/*===========================================================================
=                                                                           =
=                          MtkFileCoreMetaDataGet                           =
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
  MtkCoreMetaData metadata;     /* Metadata */
  int i;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <MISR Product File> <Parameter Name>\n",argv[0]);
    exit(1);
  }

  status = MtkFileCoreMetaDataGet(argv[1],argv[2],&metadata);
  MTK_ERR_COND_JUMP(status);

  switch (metadata.datatype)
  {
    case MTKMETA_CHAR :
      for (i = 0; i < metadata.num_values; ++i)
	printf("%s\n",metadata.data.s[i]);
      break;
    case MTKMETA_INT :
      for (i = 0; i < metadata.num_values; ++i)
	printf("%d\n",metadata.data.i[i]);
      break;
    case MTKMETA_DOUBLE :
      for (i = 0; i < metadata.num_values; ++i)
	printf("%f\n",metadata.data.d[i]);
      break;
  }

  MtkCoreMetaDataFree(&metadata);

  return 0;

ERROR_HANDLE:
  if (status_code == MTK_HDF_SDSTART_FAILED)
    fprintf(stderr, "Error opening file: %s\n", argv[1]);

  if (status_code == MTK_FAILURE)
    fprintf(stderr, "Failed to find parameter: %s\n", argv[2]);

  return status_code;
}
