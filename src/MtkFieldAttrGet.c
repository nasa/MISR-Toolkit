/*===========================================================================
=                                                                           =
=                              MtkFieldAttrGet                               =
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
  MTKt_DataBuffer attrbuf = MTKT_DATABUFFER_INIT; /* Attribute value */
  int i;

  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s <MISR Product File> <Field Name> <Attribute Name>\n",argv[0]);
    exit(1);
  }

  status = MtkFieldAttrGet(argv[1],argv[2],argv[3],&attrbuf);
  MTK_ERR_COND_JUMP(status);

  switch (attrbuf.datatype)
  {
    case MTKe_void :
      break;
    case  MTKe_char8 :
    case  MTKe_uchar8 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%c",attrbuf.data.uc8[0][i]);
      printf("\n");
      break;
    case  MTKe_int8 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%d\n",attrbuf.data.i8[0][i]);
      break;
    case  MTKe_uint8 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%u\n",attrbuf.data.u8[0][i]);
      break;
    case  MTKe_int16 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%d\n",attrbuf.data.i16[0][i]);
      break;
    case MTKe_uint16 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%u\n",attrbuf.data.u16[0][i]);
      break;
    case  MTKe_int32 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%d\n",attrbuf.data.i32[0][i]);
      break;
    case  MTKe_uint32 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%u\n",attrbuf.data.u32[0][i]);
      break;
    case  MTKe_int64 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%lld\n",(long long int) attrbuf.data.i64[0][i]);
      break;
    case  MTKe_uint64 :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%llu\n",(long long unsigned int) attrbuf.data.u64[0][i]);
      break;
    case MTKe_float :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%f\n",attrbuf.data.f[0][i]);
      break;
    case  MTKe_double :
      for (i = 0; i < attrbuf.nsample; ++i)
	printf("%f\n",attrbuf.data.d[0][i]);
      break;
  }

  MtkDataBufferFree(&attrbuf);

  return 0;

ERROR_HANDLE:
  if (status_code == MTK_HDF_SDSTART_FAILED)
    fprintf(stderr, "Error opening file: %s\n", argv[1]);

  if (status_code == MTK_HDF_SDFINDATTR_FAILED)
    fprintf(stderr, "Failed to find attribute: %s\n", argv[2]);

  return status_code;
}
