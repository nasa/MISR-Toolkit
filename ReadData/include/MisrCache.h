/*===========================================================================
=                                                                           =
=                               MisrCache                                   =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRCACHE_H
#define MISRCACHE_H

#include "MisrReadData.h"
#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrUtil.h"

#define BLOCK_CACHE_MAX       2

/** \brief Block (Low Level) */
typedef struct {
  MTKt_boolean valid;   /**< Valid */
  MTKt_DataBuffer buf;  /**< Buffer */
} MTKt_Block;

#define MTKT_BLOCK_INIT { MTK_FALSE, MTKT_DATABUFFER_INIT }

/** \brief Cache (Low Level) */
typedef struct {
  int32 fid;                   /**< HDF-EOS file identifier */
  int ncid;                    /**< netCDF file identifier */
  char *gridname;              /**< Grid name */
  char *fieldname;             /**< Field name */
  int block_cnt;               /**< Block count */
  MTKt_DataBuffer fill;        /**< Fill */
  MTKt_Block block[NBLOCK+1];  /**< Block */
} MTKt_Cache;

#define MTKT_CACHE_INIT { FAIL, 0, NULL, NULL, 0, MTKT_DATABUFFER_INIT, {MTKT_BLOCK_INIT} }

MTKt_status MtkCacheInitFid( int32 fid,
			  const char *gridname,
			  const char *fieldname,
			  MTKt_Cache *cache );

MTKt_status MtkCacheInitNcid( int ncid,
			  const char *gridname,
			  const char *fieldname,
			  MTKt_Cache *cache );

MTKt_status MtkCachePixelGet( MTKt_Cache *cache,
			      int block,
			      float line,
			      float sample,
			      void *pixel );

MTKt_status MtkCacheLoad( MTKt_Cache *cache,
			  int block );

void MtkCacheFree( MTKt_Cache *cache );

#endif /* MISRCACHE_H */
