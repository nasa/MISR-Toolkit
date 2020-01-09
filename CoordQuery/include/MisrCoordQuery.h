/*===========================================================================
=                                                                           =
=                              MisrCoordQuery                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRCOORDQUERY_H
#define MISRCOORDQUERY_H

#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrSetRegion.h"
#include "MisrFileQuery.h"

/** \brief Geographic Block Coordinates */
typedef struct {
  int block_number;		/**< Block number */
  MTKt_GeoCoord ulc;            /**< Upper left corner block coordinate */
  MTKt_GeoCoord urc;            /**< Upper right corner block coordinate */
  MTKt_GeoCoord ctr;		/**< Center block coordinate */
  MTKt_GeoCoord lrc;            /**< Lower right corner block coordinate */
  MTKt_GeoCoord llc;            /**< Lower left corner block coordinate */
} MTKt_GeoBlock;

#define MTKT_GEOBLOCK_INIT { -1, MTKT_GEOCOORD_INIT, \
                             MTKT_GEOCOORD_INIT, MTKT_GEOCOORD_INIT, \
                             MTKT_GEOCOORD_INIT, MTKT_GEOCOORD_INIT }

/** \brief Block Corners */
typedef struct {
  int path;			/**< Path */
  int start_block;		/**< Start block */
  int end_block;		/**< End block */
  MTKt_GeoBlock block[NBLOCK+1]; /**< Array of block coordinates index by 1-based block number */
} MTKt_BlockCorners;

#define MTKT_BLOCKCORNERS_INIT { 0, 0, 0, { MTKT_GEOBLOCK_INIT }}


MTKt_status MtkPathToProjParam( int path,
				int resolution_meters,
				MTKt_MisrProjParam *pp );

MTKt_status MtkLatLonToSomXY( int path,
			      double lat_dd,
			      double lon_dd,
			      double *som_x,
			      double *som_y );

MTKt_status MtkLatLonToSomXYAry( int path,
				 int nelement,
				 const double *lat_dd,
				 const double *lon_dd,
				 double *som_x,
				 double *som_y );

MTKt_status MtkSomXYToLatLon( int path,
			      double som_x,
			      double som_y,
			      double *lat_dd,
			      double *lon_dd );

MTKt_status MtkSomXYToLatLonAry( int path,
				 int nelement,
				 const double *som_x,
				 const double *som_y,
				 double *lat_dd,
				 double *lon_dd );

MTKt_status MtkBlsToSomXY( int path,
			   int resolution_meters,
			   int block,
			   float line,
			   float sample,
			   double *som_x,
			   double *som_y );

MTKt_status MtkBlsToSomXYAry( int path,
			      int resolution_meters,
			      int nelement,
			      const int *block,
			      const float *line,
			      const float *sample,
			      double *som_x,
			      double *som_y );

MTKt_status MtkSomXYToBls( int path,
			   int resolution_meters,
			   double som_x,
			   double som_y,
			   int *block,
			   float *line,
			   float *sample );

MTKt_status MtkSomXYToBlsAry( int path,
			      int resolution_meters,
			      int nelement,
			      const double *som_x,
			      const double *som_y,
			      int *block,
			      float *line,
			      float *sample );

MTKt_status MtkLatLonToBls( int path,
			    int resolution_meters,
			    double lat_dd,
			    double lon_dd,
			    int *block,
			    float *line,
			    float *sample );

MTKt_status MtkLatLonToBlsAry( int path,
			       int resolution_meters,
			       int nelement,
			       const double *lat_dd,
			       const double *lon_dd,
			       int *block,
			       float *line,
			       float *sample );

MTKt_status MtkBlsToLatLon( int path,
			    int resolution_meters,
			    int block,
			    float line,
			    float sample,
			    double *lat_dd,
			    double *lon_dd );

MTKt_status MtkBlsToLatLonAry( int path,
			       int resolution_meters,
			       int nelement,
			       const int *block,
			       const float *line,
			       const float *sample,
			       double *lat_dd,
			       double *lon_dd );

MTKt_status MtkPathBlockRangeToBlockCorners( int path,
					     int start_block,
					     int end_block,
					     MTKt_BlockCorners *block_corners );
					     
MTKt_status MtkPixelTime( MTKt_TimeMetaData time_metadata,
                          double som_x,
                          double som_y,
                          char pixel_time[MTKd_DATETIME_LEN] );

/* GCTP does not have a prototype for the following functions */
int inv_init(int insys,
             int inzone,
             const double *inparm,
             int indatum,
             char *fn27,
             char *fn83,
             int *iflg,
             int (*inv_trans[])(double, double, double*, double*));

int for_init(int outsys,
             int outzone,
             const double *outparm,
             int outdatum,
             char *fn27,
             char *fn83,
             int *iflg,
             int (*for_trans[])(double, double, double *, double *));

extern int sominv(double y,               /* (I) Y projection coordinate */
                  double x,               /* (I) X projection coordinate */
                  double *lon,            /* (O) Longitude */
                  double *lat);            /* (O) Latitude */

extern int somfor(double lon,             /* (I) Longitude                */
                  double lat,             /* (I) Latitude                 */
                  double *y,              /* (O) X projection coordinate  */
                  double *x);              /* (O) Y projection coordinate  */
#endif /* MISRCOORDQUERY_H */
