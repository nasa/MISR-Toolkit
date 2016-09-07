/*===========================================================================
=                                                                           =
=                           MtkGCTPCreateLatLon_test                        =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrMapQuery.h"
#include <cproj.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 100 * fabs(x))

#define SIZE_X 4
#define SIZE_Y 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_GCTPProjInfo proj_info = MTKT_GCTPPROJINFO_INIT;
  MTKt_GenericMapInfo map_info = MTKT_GENERICMAPINFO_INIT;
  MTKt_DataBuffer latitude = MTKT_DATABUFFER_INIT;
				/* Latitude values */
  MTKt_DataBuffer longitude = MTKT_DATABUFFER_INIT;
				/* Longitude values */
  int iline; 	   
  int isample;
  double min_x = 214140.509824;
  double min_y = 3451656.620416;
  double resolution_x = 100.0;
  double resolution_y = 200.0;
  int number_pixel_x = SIZE_X;
  int number_pixel_y = SIZE_Y;
  int origin_code = MTKe_ORIGIN_UL;  /* Line = Y; Sample = X */
  int pix_reg_code = MTKe_PIX_REG_CENTER;
  int proj_code = 1;  /* 1 = UTM */
  int zone_code = 13;
  int sphere_code = 12; 	/* 12 = WGS84 */
  double proj_param[15] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  MTK_PRINT_STATUS(cn,"Testing MtkGCTPCreateLatLon");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1 - UTM zone 13                                        */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfo(1)\n");
    error = MTK_TRUE;
  }

  status = MtkGCTPProjInfo(proj_code,
			   sphere_code,
			   zone_code,
			   proj_param,
			   &proj_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPProjInfo(1)\n");
    error = MTK_TRUE;
  }

  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPCreateLatLon(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double latitude_expect[SIZE_Y][SIZE_X] = {
      {31.173974536548413994, 31.173998965694096341, 31.174023386325664831, 31.174047798443087487 },
      {31.172172419767566254, 31.172196847187521485, 31.172221266093963266, 31.17224567648685607 },
      {31.170370302388374029, 31.170394728082662539, 31.17041914526404156, 31.170443553932475567 },
      {31.168568184410862187, 31.168592608379551478, 31.16861702383593169, 31.168641430779963741 },
      {31.16676606583506981, 31.166790488078227384, 31.16681490180966918, 31.166839307029366779 },
      {31.164963946661011107, 31.164988367178693807, 31.165012779185261138, 31.165037182680688232 }
    };
    double longitude_expect[SIZE_Y][SIZE_X] = {
      {-107.9986523629855526, -107.99760440171924358, -107.99655643917664349, -107.99550847535817866 },
      {-107.99859553182326977, -107.99754759034561857, -107.99649964759177578, -107.99545170356218193 },
      {-107.99853870576762915, -107.99749078407685943, -107.99644286111001179, -107.99539493686749836 },
      {-107.99848188481821865, -107.99743398291258245, -107.99638607973092519, -107.99533817527373003 },
      {-107.99842506897464034, -107.99737718685234711, -107.9963293034541465, -107.99528141878046483 },
      {-107.99836825823651054, -107.99732039589576971, -107.99627253227923518, -107.99522466738733328 }
    };

    int nline_expect = SIZE_Y;
    int nsample_expect = SIZE_X;

    if (latitude.nline != nline_expect ||
	latitude.nsample != nsample_expect ||
	longitude.nline != nline_expect ||
	longitude.nsample != nsample_expect) {
      fprintf(stderr,"latitude.nline = %d (expected %d)\n",
	      latitude.nline, nline_expect);
      fprintf(stderr,"latitude.nsample = %d (expected %d)\n",
	      latitude.nsample, nsample_expect);
      fprintf(stderr,"longitude.nline = %d (expected %d)\n",
	      longitude.nline, nline_expect);
      fprintf(stderr,"longitude.nsample = %d (expected %d)\n",
	      longitude.nsample, nsample_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
    for (iline = 0 ; iline < latitude.nline ; iline++) {
      for (isample = 0 ; isample < latitude.nsample ; isample++) {
	if (MTKm_CMP_NE_DBL(latitude.data.d[iline][isample], 
			    latitude_expect[iline][isample]) ||
	    MTKm_CMP_NE_DBL(longitude.data.d[iline][isample], 
			    longitude_expect[iline][isample])) {
	  fprintf(stderr,"latitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, latitude.data.d[iline][isample], 
		  latitude_expect[iline][isample]);
	  fprintf(stderr,"longitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, longitude.data.d[iline][isample], 
		  longitude_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }
    if (error) {
      fprintf(stderr,"latitude:\n");
      for (iline = 0 ; iline < latitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < latitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",latitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }

      fprintf(stderr,"longitude:\n");
      for (iline = 0 ; iline < longitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < longitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",longitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2 - UTM zone 13 , origin UR                            */
  /* ------------------------------------------------------------------ */

  origin_code = MTKe_ORIGIN_UR;  /* Line = X; Sample = Y */
  
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfo(1)\n");
    error = MTK_TRUE;
  }

  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPCreateLatLon(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double latitude_expect[SIZE_X][SIZE_Y] = {
      {31.174047798443087487, 31.17224567648685607, 31.170443553932475567, 31.168641430779963741, 31.166839307029366779, 31.165037182680688232 },
      {31.174023386325664831, 31.172221266093963266, 31.17041914526404156, 31.16861702383593169, 31.16681490180966918, 31.165012779185261138 },
      {31.173998965694096341, 31.172196847187521485, 31.170394728082662539, 31.168592608379551478, 31.166790488078227384, 31.164988367178693807 },
      {31.173974536548413994, 31.172172419767566254, 31.170370302388374029, 31.168568184410862187, 31.16676606583506981, 31.164963946661011107 }
    };
    double longitude_expect[SIZE_X][SIZE_Y] = {
      {-107.99550847535817866, -107.99545170356218193, -107.99539493686749836, -107.99533817527373003, -107.99528141878046483, -107.99522466738733328 },
      {-107.99655643917664349, -107.99649964759177578, -107.99644286111001179, -107.99638607973092519, -107.9963293034541465, -107.99627253227923518 },
      {-107.99760440171924358, -107.99754759034561857, -107.99749078407685943, -107.99743398291258245, -107.99737718685234711, -107.99732039589576971 },
      {-107.9986523629855526, -107.99859553182326977, -107.99853870576762915, -107.99848188481821865, -107.99842506897464034, -107.99836825823651054 }
    };

    int nline_expect = SIZE_X;
    int nsample_expect = SIZE_Y;

    if (latitude.nline != nline_expect ||
	latitude.nsample != nsample_expect ||
	longitude.nline != nline_expect ||
	longitude.nsample != nsample_expect) {
      fprintf(stderr,"latitude.nline = %d (expected %d)\n",
	      latitude.nline, nline_expect);
      fprintf(stderr,"latitude.nsample = %d (expected %d)\n",
	      latitude.nsample, nsample_expect);
      fprintf(stderr,"longitude.nline = %d (expected %d)\n",
	      longitude.nline, nline_expect);
      fprintf(stderr,"longitude.nsample = %d (expected %d)\n",
	      longitude.nsample, nsample_expect);
      fprintf(stderr,"Unexpected result(test2).\n");
      error = MTK_TRUE;
    }
    for (iline = 0 ; iline < latitude.nline ; iline++) {
      for (isample = 0 ; isample < latitude.nsample ; isample++) {
	if (MTKm_CMP_NE_DBL(latitude.data.d[iline][isample], 
			    latitude_expect[iline][isample]) ||
	    MTKm_CMP_NE_DBL(longitude.data.d[iline][isample], 
			    longitude_expect[iline][isample])) {
	  fprintf(stderr,"latitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, latitude.data.d[iline][isample], 
		  latitude_expect[iline][isample]);
	  fprintf(stderr,"longitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, longitude.data.d[iline][isample], 
		  longitude_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test2).\n");
	  error = MTK_TRUE;
	}
      }
    }
    if (error) {
      fprintf(stderr,"latitude:\n");
      for (iline = 0 ; iline < latitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < latitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",latitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }

      fprintf(stderr,"longitude:\n");
      for (iline = 0 ; iline < longitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < longitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",longitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 3 - Albers Conic Equal-area                            */
  /* ------------------------------------------------------------------ */

  min_x = 1930612.449614;
  min_y = 2493633.488881;
  origin_code = MTKe_ORIGIN_UL;  /* Line = Y; Sample = X */

  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfo(1)\n");
    error = MTK_TRUE;
  }

  proj_code = 3; /* Albers */
  zone_code = -1;
  sphere_code = 8; /* GRS80 */
  proj_param[2] = 29030000.0; /* Latitude of 1st standard parallel */
  proj_param[3] = 45030000.0; /* Latitude of 2nd standard parallel */
  proj_param[4] = -96000000.0; /* Longitude of central meridian */ 
  proj_param[5] = 23000000.0; /* Latitude of the projection origin */
  status = MtkGCTPProjInfo(proj_code,
			   sphere_code,
			   zone_code,
			   proj_param,
			   &proj_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPProjInfo(1)\n");
    error = MTK_TRUE;
  }

  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPCreateLatLon(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double latitude_expect[SIZE_Y][SIZE_X] = {
      {43.227635745212630525, 43.227410646902264091, 43.227185537752440325, 43.226960417763422129 },
      {43.225902262061381975, 43.22567717006087662, 43.225452067221226571, 43.225226953542659203 },
      {43.224168780868161832, 43.223943695176998858, 43.223718598646961198, 43.223493491278119905 },
      {43.222435301631158211, 43.222210222248818923, 43.221985132027754162, 43.22176003096816288 },
      {43.220701824348665809, 43.220476751274418348, 43.220251667361793579, 43.220026572610848348 },
      {43.218968349018730635, 43.218743282252148674, 43.218518204647374148, 43.218293116204485216 }
    };
    double longitude_expect[SIZE_Y][SIZE_X] = {
      {-71.853298490560831624, -71.852100938629916982, -71.850903394537553481, -71.849705858284025339 },
      {-71.853920485327776646, -71.852722961542255575, -71.851525445594703001, -71.850327937485388929 },
      {-71.854542448744112448, -71.853344953102777026, -71.852147465298799034, -71.850949985332533743 },
      {-71.855164380812141189, -71.853966913313769282, -71.852769453652186371, -71.851572001827719305 },
      {-71.85578628153420766, -71.854588842177577135, -71.853391410657167171, -71.852193986973276196 },
      {-71.856408150912614019, -71.855210739696531164, -71.854013336316057803, -71.852815940771563419 }
    };

    int nline_expect = SIZE_Y;
    int nsample_expect = SIZE_X;

    if (latitude.nline != nline_expect ||
	latitude.nsample != nsample_expect ||
	longitude.nline != nline_expect ||
	longitude.nsample != nsample_expect) {
      fprintf(stderr,"latitude.nline = %d (expected %d)\n",
	      latitude.nline, nline_expect);
      fprintf(stderr,"latitude.nsample = %d (expected %d)\n",
	      latitude.nsample, nsample_expect);
      fprintf(stderr,"longitude.nline = %d (expected %d)\n",
	      longitude.nline, nline_expect);
      fprintf(stderr,"longitude.nsample = %d (expected %d)\n",
	      longitude.nsample, nsample_expect);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
    for (iline = 0 ; iline < latitude.nline ; iline++) {
      for (isample = 0 ; isample < latitude.nsample ; isample++) {
	if (MTKm_CMP_NE_DBL(latitude.data.d[iline][isample], 
			    latitude_expect[iline][isample]) ||
	    MTKm_CMP_NE_DBL(longitude.data.d[iline][isample], 
			    longitude_expect[iline][isample])) {
	  fprintf(stderr,"latitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, latitude.data.d[iline][isample], 
		  latitude_expect[iline][isample]);
	  fprintf(stderr,"longitude.data.d[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, longitude.data.d[iline][isample], 
		  longitude_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test3).\n");
	  error = MTK_TRUE;
	}
      }
    }
    if (error) {
      fprintf(stderr,"latitude:\n");
      for (iline = 0 ; iline < latitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < latitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",latitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }

      fprintf(stderr,"longitude:\n");
      for (iline = 0 ; iline < longitude.nline ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < longitude.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",longitude.data.d[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /*                 Map_info->size_line < 1                            */
  /*                 Map_info->size_sample < 1                          */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPCreateLatLon(NULL,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  map_info.size_line = 0;
  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info.size_line = SIZE_X;

  map_info.size_sample = 0;
  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info.size_sample = SIZE_Y;

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPCreateLatLon(&map_info,
			       NULL,
			       &latitude,
			       &longitude);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Latitude == NULL                                   */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       NULL,
			       &longitude);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Longitude == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPCreateLatLon(&map_info,
			       &proj_info,
			       &latitude,
			       NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Report test result.                                                */
  /* ------------------------------------------------------------------ */
      
  if (error) {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  } else {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  }
}
