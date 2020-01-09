/*===========================================================================
=                                                                           =
=                              MisrMapQuery                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRMAPQUERY_H
#define MISRMAPQUERY_H

#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrUtil.h"

/** \brief Geographic Coordinates */
typedef struct {
  double lat;			/**< Latitude in decimal degrees */
  double lon;			/**< Longitude in decimal degrees */
} MTKt_GeoCoord;

#define MTKT_GEOCOORD_INIT { 0.0, 0.0 }

/** \brief SOM Coordinates */
typedef struct {
  double x;			/**< Som X in meters */
  double y;			/**< Som Y in meters */
} MTKt_SomCoord;

#define MTKT_SOMCOORD_INIT { 0.0, 0.0 }

/** \brief Geographic Region */
typedef struct {
  MTKt_GeoCoord ulc;		/**< Upper left corner of region in geographic */
  MTKt_GeoCoord urc;		/**< Upper right corner of region in geographic */
  MTKt_GeoCoord ctr;		/**< Center of region in geographic */
  MTKt_GeoCoord lrc;		/**< Lower right corner of geographic */
  MTKt_GeoCoord llc;		/**< Lower left corner of geographic */
} MTKt_GeoRegion;

#define MTKT_GEOREGION_INIT { MTKT_GEOCOORD_INIT, MTKT_GEOCOORD_INIT, \
                              MTKT_GEOCOORD_INIT, MTKT_GEOCOORD_INIT, \
                              MTKT_GEOCOORD_INIT }

/** \brief SOM Region */
typedef struct {
  int path;			/**< Path these coordinates */
  MTKt_SomCoord ulc;		/**< Upper left corner of region in som */
  MTKt_SomCoord ctr;		/**< Center of region in som */
  MTKt_SomCoord lrc;		/**< Lower right corner of region in som */
} MTKt_SomRegion;

#define MTKT_SOMREGION_INIT { 0, MTKT_SOMCOORD_INIT, MTKT_SOMCOORD_INIT, \
                              MTKT_SOMCOORD_INIT }

/** \brief Map Information */
typedef struct {
  int path;			/**< Path */
  int start_block;		/**< Start block */
  int end_block;		/**< End block */
  int resolution;		/**< Resolution */
  int resfactor;		/**< Resolution factor */
  int nline;			/**< Number of lines */
  int nsample;			/**< Number of samples */
  MTKt_boolean pixelcenter;	/**< Pixel registration center */
  MTKt_SomRegion som;		/**< Som region */
  MTKt_GeoRegion geo;		/**< Geographic region */
  MTKt_MisrProjParam pp;	/**< MISR projection parameters */
} MTKt_MapInfo;

#define MTKT_MAPINFO_INIT { 0, 0, 0, 0, 0, 0, 0, MTK_TRUE, \
                            MTKT_SOMREGION_INIT, MTKT_GEOREGION_INIT, \
                            MTKT_MISRPROJPARAM_INIT }

/** \brief Origin code */
typedef enum { 
  MTKe_ORIGIN_UL = 0,    /**< Upper Left (min X, max Y);  Line=Y, Sample=X */
  MTKe_ORIGIN_UR = 1,    /**< Upper Right (max X, max Y); Line=X, Sample=Y */
  MTKe_ORIGIN_LL = 2,    /**< Lower Left (min X, min Y);  Line=X, Sample=Y */
  MTKe_ORIGIN_LR = 3	 /**< Lower Right (max X, min Y); Line=Y, Sample=X */
} MTKt_OriginCode;

/** \brief Pixel registration code */
typedef enum {
  MTKe_PIX_REG_CENTER = 0,  /**< Center */
  MTKe_PIX_REG_CORNER = 1   /**< Corner */
} MTKt_PixRegCode;

/** \brief Generic map information */
typedef struct {
  double min_x; 		/**< Minimum X coord */
  double min_y;			/**< Minimum Y coord */
  double max_x; 		/**< Maximum X coord */
  double max_y;			/**< Maximum Y coord */
  int size_line;		/**< Size of map in lines 
				   (slowest changing dimension) */
  int size_sample;		/**< Size of map in samples 
				   (fastest changing dimension) */
  double resolution_x;          /**< Size of a pixel along the X axis */
  double resolution_y;          /**< Size of a pixel along the Y axis */
  double tline[4]; 		/**< Line dimension transform coefficients for 
				     converting between map coordinates
				     and pixel coordinates. */
  double tsample[4]; 		/**< Sample dimension transform coefficients for 
				     converting between map coordinates
				     and pixel coordinates. */
  MTKt_OriginCode origin_code;  /**< Origin code */
  MTKt_PixRegCode pix_reg_code; /**< Pixel registration code */
} MTKt_GenericMapInfo;


#define MTKT_GENERICMAPINFO_INIT { 0, 0, 0, 0, 0, 0, 0, 0, {0,0,0}, {0,0,0}, MTKe_ORIGIN_UL, MTKe_PIX_REG_CENTER }

/** \brief GCTP projection information */
typedef struct {
  int proj_code;
  int sphere_code;
  int zone_code;
  double proj_param[15];
} MTKt_GCTPProjInfo;

#define MTKT_GCTPPROJINFO_INIT { 0, 0, 0, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} }

MTKt_status MtkLSToSomXY( MTKt_MapInfo mapinfo,
			  float line,
			  float sample,
			  double *som_x,
			  double *som_y );

MTKt_status MtkLSToSomXYAry( MTKt_MapInfo mapinfo,
			     int nelement,
			     const float *line,
			     const float *sample,
			     double *som_x,
			     double *som_y );

MTKt_status MtkSomXYToLS( MTKt_MapInfo mapinfo,
			  double som_x,
			  double som_y,
			  float *line,
			  float *sample );

MTKt_status MtkSomXYToLSAry( MTKt_MapInfo mapinfo,
			     int nelement,
			     const double *som_x,
			     const double *som_y,
			     float *line,
			     float *sample );

MTKt_status MtkLatLonToLS( MTKt_MapInfo mapinfo,
			   double lat_dd,
			   double lon_dd,
			   float *line,
			   float *sample );

MTKt_status MtkLatLonToLSAry( MTKt_MapInfo mapinfo,
			      int nelement,
			      const double *lat_dd,
			      const double *lon_dd,
			      float *line,
			      float *sample );

MTKt_status MtkLSToLatLon( MTKt_MapInfo mapinfo,
			   float line,
			   float sample,
			   double *lat_dd,
			   double *lon_dd );

MTKt_status MtkLSToLatLonAry( MTKt_MapInfo mapinfo,
			      int nelement,
			      const float *line,
			      const float *sample,
			      double *lat_dd,
			      double *lon_dd );

MTKt_status MtkCreateLatLon( MTKt_MapInfo mapinfo,
			     MTKt_DataBuffer *latbuf,
			     MTKt_DataBuffer *lonbuf );

MTKt_status MtkGenericMapInfo(
  double Min_x, 
  double Min_y,  
  double Resolution_x, 
  double Resolution_y, 
  int Number_pixel_x,  
  int Number_pixel_y,  
  MTKt_OriginCode Origin_code, 
  MTKt_PixRegCode Pix_reg_code, 
  MTKt_GenericMapInfo *Map_info );

MTKt_status MtkGCTPProjInfo(
  int Proj_code,  
  int Sphere_code,
  int Zone_code,  
  double Proj_param[15],
  MTKt_GCTPProjInfo *Proj_info );

MTKt_status MtkGCTPCreateLatLon(
  const MTKt_GenericMapInfo *Map_info, 
  const MTKt_GCTPProjInfo *Proj_info,
  MTKt_DataBuffer *Latitude, 
  MTKt_DataBuffer *Longitude );

MTKt_status MtkGenericMapInfoRead(
  const char *Filename,  /**< [IN] Filename */
  MTKt_GenericMapInfo *Map_info /**< [OUT] Map information. */
);

MTKt_status MtkGCTPProjInfoRead(
  const char *Filename,  /**< [IN] Filename */
  MTKt_GCTPProjInfo *Proj_info /**< [OUT] Proj information. */
);

MTKt_status MtkChangeMapResolution(
  const MTKt_MapInfo *Map_info_in,  /**< [IN] Input map information */
  int Resolution,            /**< [IN] Desired output resolution. */
  MTKt_MapInfo *Map_info_out  /**< [OUT] Output map information. */
);

#endif /* MISRMAPQUERY_H */
