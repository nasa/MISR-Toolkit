/*===========================================================================
=                                                                           =
=                           MtkTimeMetaRead_test                            =
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
  int j;
  int cn = 0;			/* Column number */
  char filename[200];		/* HDF-EOS filename */
  MTKt_TimeMetaData time_metadata = MTKT_TIME_METADATA_INIT;
  MTKt_double coeff_line[6][NGRIDCELL] = {{5.2314916E+04, 5.2567776E+04},
  	                                      {9.8764910E-01, 9.8778724E-01},
  	                                      {-6.4932617E-02, -6.4070067E-02},
  	                                      {1.0238983E-05, 1.0254441E-05},
  	                                      {3.3763384E-06, 3.4716382E-06},
  	                                      {5.3088579E-11, 5.5760239E-11}};

  MTK_PRINT_STATUS(cn,"Testing MtkTimeMetaRead");

  /* Normal test call */
  data_ok = MTK_TRUE;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  
  status = MtkTimeMetaRead(filename,&time_metadata);
  if (status == MTK_SUCCESS)
  {
  	if (time_metadata.path != 37 ||
  	    time_metadata.start_block != 1 ||
  	    time_metadata.end_block != 140 ||
  	    strcmp(time_metadata.camera,"AA") != 0 ||
  	    time_metadata.number_transform[100] != 2 ||
  	    strcmp(time_metadata.ref_time[100][0],"2005-06-04T17:58:13.127920Z") != 0 ||
  	    strcmp(time_metadata.ref_time[100][1],"2005-06-04T17:58:13.127920Z") != 0 ||
  	    time_metadata.start_line[100][0] != 50688 ||
        time_metadata.start_line[100][1] != 50944 ||
        time_metadata.number_line[100][0] != 256 ||
        time_metadata.number_line[100][1] != 256)
       data_ok = MTK_FALSE;
       
     for (i = 0; i < 6; ++i)
       for (j = 0; j < 2; ++j)
         if (fabs(time_metadata.coeff_line[100][i][j]) - fabs(coeff_line[i][j]) > 0.0000001)
           data_ok = MTK_FALSE;
        
     if (fabs(time_metadata.som_ctr_x[100][0]) - fabs(5.0816000E+04) > 0.00001 ||
        fabs(time_metadata.som_ctr_x[100][1]) - fabs(5.1072000E+04) > 0.00001 ||
        fabs(time_metadata.som_ctr_y[100][0]) - fabs(1.0240000E+03) > 0.00001 ||
        fabs(time_metadata.som_ctr_y[100][1]) - fabs(1.0240000E+03) > 0.00001)
       data_ok = MTK_FALSE;
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

  /* File has no time metadata */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf");

  status = MtkTimeMetaRead(filename,&time_metadata);
  if (status == MTK_HDF_VSFIND_FAILED)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Argument Checks */
  status = MtkTimeMetaRead(NULL,&time_metadata);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeMetaRead(filename,NULL);
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
