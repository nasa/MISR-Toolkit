/*===========================================================================
=                                                                           =
=                              MisrReProject                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRREPROJECT_H
#define MISRREPROJECT_H

#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrMapQuery.h"

MTKt_status MtkCreateGeoGrid( double ulc_lat_dd,
			      double ulc_lon_dd,
			      double lrc_lat_dd,
			      double lrc_lon_dd,
			      double lat_cellsize_dd,
			      double lon_cellsize_dd,
			      MTKt_DataBuffer *latbuf,
			      MTKt_DataBuffer *lonbuf );

MTKt_status MtkTransformCoordinates( MTKt_MapInfo mapinfo,
				     MTKt_DataBuffer latbuf,
				     MTKt_DataBuffer lonbuf,
				     MTKt_DataBuffer *linebuf,
				     MTKt_DataBuffer *samplebuf );

MTKt_status MtkResampleNearestNeighbor(
				       MTKt_DataBuffer srcbuf,
				       MTKt_DataBuffer linebuf,
				       MTKt_DataBuffer samplebuf,
				       MTKt_DataBuffer *resampbuf );

MTKt_status MtkResampleCubicConvolution(
  const MTKt_DataBuffer *Source,
  const MTKt_DataBuffer *Source_mask, 
  const MTKt_DataBuffer *Line, 
  const MTKt_DataBuffer *Sample, 
  float A,
  MTKt_DataBuffer *Resampled, 
  MTKt_DataBuffer *Resampled_mask );

#endif /* MISRREPROJECT_H */
