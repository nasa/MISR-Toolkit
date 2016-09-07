/*===========================================================================
=                                                                           =
=                              MisrSetRegion                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRSETREGION_H
#define MISRSETREGION_H

#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrMapQuery.h"


/** \brief Geographic Center */
typedef struct {
  MTKt_GeoCoord ctr;		/**< Center of region */
} MTKt_GeoCenter;

#define MTKT_GEOCENTER_INIT { MTKT_GEOCOORD_INIT }

/** \brief Geographic Extent */
typedef struct {
  double xlat;			/**< Som x or latitude extent in meters */
  double ylon;			/**< Som y or longitude extent in meters */
} MTKt_Extent;

#define MTKT_EXTENT_INIT { 0.0, 0.0 }

/** \brief Region of interest */
typedef struct {
  MTKt_GeoCenter geo;		/**< Region center coordinate in geographic lat/lon */
  MTKt_Extent hextent;		/**< Half of the region overall extent in meters (measured from geo.ctr) */
} MTKt_Region;

#define MTKT_REGION_INIT { MTKT_GEOCENTER_INIT, MTKT_EXTENT_INIT }


MTKt_status MtkSnapToGrid( int path,
			   int resolution,
			   MTKt_Region region,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkSetRegionByUlcLrc( double ulc_lat_dd,
				  double ulc_lon_dd,
				  double lrc_lat_dd,
				  double lrc_lon_dd,
				  MTKt_Region *region );

MTKt_status MtkSetRegionByPathBlockRange( int path_number,
					  int start_block,
					  int end_block,
					  MTKt_Region *region );

MTKt_status MtkSetRegionByLatLonExtent( double ctr_lat_dd,
					double ctr_lon_dd,
					double lat_extent,
					double lon_extent,
					const char *extent_units,
					MTKt_Region *region );

MTKt_status MtkSetRegionByPathSomUlcLrc(
					int path,
					double ulc_som_x,
					double ulc_som_y,
					double lrc_som_x,
					double lrc_som_y,
					MTKt_Region *region);

MTKt_status MtkSetRegionByGenericMapInfo(
  const MTKt_GenericMapInfo *Map_info, /**< [IN] Map information. */
  const MTKt_GCTPProjInfo *Proj_info,  /**< [IN] Projection information. */
  int Path,			       /**< [IN] Orbit path number. */
  MTKt_Region *Region /**< [OUT] Region */
					 );

#endif /* MISRSETREGION_H */
