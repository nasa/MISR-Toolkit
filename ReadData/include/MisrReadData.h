/*===========================================================================
=                                                                           =
=                              MisrReadData                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRREADDATA_H
#define MISRREADDATA_H

#include <MisrError.h>
#include <MisrSetRegion.h>
#include <MisrUtil.h>


MTKt_status MtkReadBlock( const char *filename,
			  const char *gridname,
			  const char *fieldname,
			  int block,
			  MTKt_DataBuffer *databuf );

MTKt_status MtkReadBlockNC( const char *filename,
			  const char *gridname,
			  const char *fieldname,
			  int block,
			  MTKt_DataBuffer *databuf );

MTKt_status MtkReadBlockHDF( const char *filename,
			  const char *gridname,
			  const char *fieldname,
			  int block,
			  MTKt_DataBuffer *databuf );

MTKt_status MtkReadBlockFid( int32 fid,
			     const char *gridname,
			     const char *fieldname,
			     int block,
			     MTKt_DataBuffer *databuf );

MTKt_status MtkReadBlockNcid( int ncid,
			     const char *gridname,
			     const char *fieldname,
			     int block,
			     MTKt_DataBuffer *databuf );

MTKt_status MtkReadBlockRange( const char *filename,
			       const char *gridname,
			       const char *fieldname,
			       int startblock,
			       int endblock,
			       MTKt_DataBuffer3D *databuf );

MTKt_status MtkReadBlockRangeNC( const char *filename,
			       const char *gridname,
			       const char *fieldname,
			       int startblock,
			       int endblock,
			       MTKt_DataBuffer3D *databuf );

MTKt_status MtkReadBlockRangeHDF( const char *filename,
			       const char *gridname,
			       const char *fieldname,
			       int startblock,
			       int endblock,
			       MTKt_DataBuffer3D *databuf );

MTKt_status MtkReadBlockRangeFid( int32 fid,
				  const char *gridname,
				  const char *fieldname,
				  int startblock,
				  int endblock,
				  MTKt_DataBuffer3D *databuf );

MTKt_status MtkReadBlockRangeNcid( int ncid,
				  const char *gridname,
				  const char *fieldname,
				  int startblock,
				  int endblock,
				  MTKt_DataBuffer3D *databuf );

MTKt_status MtkReadConv( const char *filename,
			 const char *gridname,
			 const char *fieldname,
			 MTKt_Region region,
			 MTKt_DataBuffer *databuf,
			 MTKt_MapInfo *mapinfo );

MTKt_status MtkReadConvFid( int32 fid,
			    const char *gridname,
			    const char *fieldname,
			    MTKt_Region region,
			    MTKt_DataBuffer *databuf,
			    MTKt_MapInfo *mapinfo );

MTKt_status MtkReadData( const char *filename,
			 const char *gridname,
			 const char *fieldname,
			 MTKt_Region region,
			 MTKt_DataBuffer *databuf,
			 MTKt_MapInfo *mapinfo );

MTKt_status MtkReadDataHDF( const char *filename,
			 const char *gridname,
			 const char *fieldname,
			 MTKt_Region region,
			 MTKt_DataBuffer *databuf,
			 MTKt_MapInfo *mapinfo );

MTKt_status MtkReadDataNC( const char *filename,
			 const char *gridname,
			 const char *fieldname,
			 MTKt_Region region,
			 MTKt_DataBuffer *databuf,
			 MTKt_MapInfo *mapinfo );

MTKt_status MtkReadDataFid( int32 fid,
			    const char *gridname,
			    const char *fieldname,
			    MTKt_Region region,
			    MTKt_DataBuffer *databuf,
			    MTKt_MapInfo *mapinfo );

MTKt_status MtkReadDataNcid( int ncid,
			    const char *gridname,
			    const char *fieldname,
			    MTKt_Region region,
			    MTKt_DataBuffer *databuf,
			    MTKt_MapInfo *mapinfo );

MTKt_status MtkReadRaw( const char *filename,
			const char *gridname,
			const char *fieldname,
			MTKt_Region region,
			MTKt_DataBuffer *databuf,
			MTKt_MapInfo *mapinfo );

MTKt_status MtkReadRawNC( const char *filename,
			const char *gridname,
			const char *fieldname,
			MTKt_Region region,
			MTKt_DataBuffer *databuf,
			MTKt_MapInfo *mapinfo );

MTKt_status MtkReadRawHDF( const char *filename,
			const char *gridname,
			const char *fieldname,
			MTKt_Region region,
			MTKt_DataBuffer *databuf,
			MTKt_MapInfo *mapinfo );

MTKt_status MtkReadRawFid( int32 fid,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadRawNcid( int ncid,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL1B2( const char *filename,
			 const char *gridname,
			 const char *fieldname,
			 MTKt_Region region,
			 MTKt_DataBuffer *databuf,
			 MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL1B2Fid( int32 Fid,
			    const char *gridname,
			    const char *fieldname,
			    MTKt_Region region,
			    MTKt_DataBuffer *databuf,
			    MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL2Land( const char *filename,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL2LandNC( const char *filename,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL2LandHDF( const char *filename,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL2LandFid( int32 fid,
			      const char *gridname,
			      const char *fieldname,
			      MTKt_Region region,
			      MTKt_DataBuffer *databuf,
			      MTKt_MapInfo *mapinfo );
            
MTKt_status MtkReadL2LandNcid( int ncid,
			      const char *gridname,
			      const char *fieldname,
			      MTKt_Region region,
			      MTKt_DataBuffer *databuf,
			      MTKt_MapInfo *mapinfo );
            
MTKt_status MtkReadL2TCCloud( const char *filename,
			   const char *gridname,
			   const char *fieldname,
			   MTKt_Region region,
			   MTKt_DataBuffer *databuf,
			   MTKt_MapInfo *mapinfo );

MTKt_status MtkReadL2TCCloudFid( int32 fid,
			      const char *gridname,
			      const char *fieldname,
			      MTKt_Region region,
			      MTKt_DataBuffer *databuf,
			      MTKt_MapInfo *mapinfo );            

#endif /* MISRREADDATA_H */
