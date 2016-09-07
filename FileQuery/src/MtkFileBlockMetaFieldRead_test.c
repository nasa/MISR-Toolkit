/*===========================================================================
=                                                                           =
=                      MtkFileBlockMetaFieldRead_test                        =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  int i;
  int cn = 0;			/* Column number */
  char filename[200];		/* HDF-EOS filename */
  char blockmetaname[200];	/* HDF-EOS gridname */
  char fieldname[200];
  MTKt_DataBuffer blockmetabuf;
  double field_expected[] = {11684750.0, 11825550.0, 11966350.0,
	                         12107150.0, 12247950.0 };
  double field_expected1[] = {9.1568693E+03, 9.8937055E-01, 4.0689267E-03,
                              1.0377393E-05, -3.0672129E-06, -3.3978681E-11,
                              9.4101564E+03,  9.8941301E-01, 3.2922175E-03,
                              1.0390057E-05, -2.9363045E-06, -2.5240831E-11 };
  int field_expected2[] = {25983, 26111, 26239, 26367, 26495 };
  char *field_expected3[] = {"2005-06-04T18:09:16.908637Z",
	                         "2005-06-04T18:09:37.609788Z",
	                         "2005-06-04T18:09:58.265817Z",
	                         "2005-06-04T18:10:18.919861Z",
	                         "2005-06-04T18:10:39.629666Z" };
  double field_expected4[] = {149717371749.589905, 149717361677.946320,
  	                          149717351737.717163, 149717341665.642548,
  	                          149717331721.284851 };

  MTK_PRINT_STATUS(cn,"Testing MtkFileBlockMetaFieldRead");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(blockmetaname, "PerBlockMetadataCommon");
  strcpy(fieldname, "Block_coor_ulc_som_meter.x");

  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_SUCCESS)
  {
    if (blockmetabuf.datatype != MTKe_double || blockmetabuf.nline != 140 ||
        blockmetabuf.nsample != 1)
      data_ok = MTK_FALSE;

    for (i = 0; i < 5; ++i)
      if (fabs(blockmetabuf.data.d[i + 30][0]) - fabs(field_expected[i]) > 0.00001)
      {
        data_ok = MTK_FALSE;
        break;
      }
    
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(blockmetaname, "PerBlockMetadataRad");
  strcpy(fieldname, "transform.ref_time");

  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);    
  if (status == MTK_SUCCESS)
  {
  	if (blockmetabuf.datatype != MTKe_char8 || blockmetabuf.nline != 140 ||
        blockmetabuf.nsample != 54)
      data_ok = MTK_FALSE;

    if (strncmp(blockmetabuf.data.c8[54],
                "2005-06-04T17:58:13.127920Z2005-06-04T17:58:13.127920Z",54) != 0)
      data_ok = MTK_FALSE;
    
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(blockmetaname, "PerBlockMetadataRad");
  strcpy(fieldname, "transform.coeff_line");

  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_SUCCESS)
  {
    if (blockmetabuf.datatype != MTKe_double || blockmetabuf.nline != 140 ||
        blockmetabuf.nsample != 12)
      data_ok = MTK_FALSE;

    for (i = 0; i < 12; ++i)
      if (fabs(blockmetabuf.data.d[14][i]) - fabs(field_expected1[i]) > 0.0001)
      {
        data_ok = MTK_FALSE;
        break;
      }
    
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P177_F01_24.hdf");
  strcpy(blockmetaname, "PerBlockMetadataAGP");
  strcpy(fieldname, "ULC_som_pixel.x");
  
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_SUCCESS)
  {
    if (blockmetabuf.datatype != MTKe_int32 || blockmetabuf.nline != 180 ||
        blockmetabuf.nsample != 1)
      data_ok = MTK_FALSE;

    for (i = 0; i < 5; ++i)
      if (blockmetabuf.data.i32[i + 150][0] != field_expected2[i])
      {
        data_ok = MTK_FALSE;
        break;
      }
    
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf");
  strcpy(blockmetaname, "PerBlockMetadataTime");
  strcpy(fieldname, "BlockCenterTime");
    
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_SUCCESS)
  {
    if (blockmetabuf.datatype != MTKe_char8 || blockmetabuf.nline != 140 ||
        blockmetabuf.nsample != 28)
      data_ok = MTK_FALSE;
    
    for (i = 0; i < 5; ++i)
      if (strcmp(blockmetabuf.data.c8[i + 30],field_expected3[i]) != 0)
      {
        data_ok = MTK_FALSE;
        break;
      }
         
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  strcpy(blockmetaname, "PerBlockMetadataGeoParm");
  strcpy(fieldname, "SunDistance");
  
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_SUCCESS)
  {
    if (blockmetabuf.datatype != MTKe_double || blockmetabuf.nline != 163 ||
        blockmetabuf.nsample != 1)
      data_ok = MTK_FALSE;

    for (i = 0; i < 5; ++i)
      if (fabs(blockmetabuf.data.d[i + 30][0]) - fabs(field_expected4[i]) > 0.00001)
      {
        data_ok = MTK_FALSE;
        break;
      }
    
    MtkDataBufferFree(&blockmetabuf);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
  
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_HDF_HDFOPEN_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
    /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  strcpy(blockmetaname, "abcd");
  
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_HDF_VSFIND_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Failure test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GP_GMP_P037_O014845_F02_0009.hdf");
  strcpy(blockmetaname, "PerBlockMetadataGeoParm");
  strcpy(fieldname, "abcd");
  
  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_HDF_VSFINDEX_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileBlockMetaFieldRead(NULL,blockmetaname,fieldname,&blockmetabuf);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldRead(filename,NULL,fieldname,&blockmetabuf);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,NULL,&blockmetabuf);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileBlockMetaFieldRead(filename,blockmetaname,fieldname,NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
