/*===========================================================================
=                                                                           =
=                              MisrFileQuery                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRFILEQUERY_H
#define MISRFILEQUERY_H

#include "MisrError.h"
#include "MisrUtil.h"
#include "MisrProjParam.h"

typedef enum MTKt_FileType {       /* File type */
  MTK_AGP,
  MTK_GP_GMP,
  MTK_GRP_RCCM,
  MTK_GRP_ELLIPSOID_GM,
  MTK_GRP_TERRAIN_GM,
  MTK_GRP_ELLIPSOID_LM,
  MTK_GRP_TERRAIN_LM,
  MTK_AS_AEROSOL,
  MTK_AS_LAND,
  MTK_TC_ALBEDO,
  MTK_TC_CLASSIFIERS,
  MTK_TC_STEREO,
  MTK_PP,
  MTK_CONVENTIONAL,
  MTK_UNKNOWN,
  MTK_TC_CLOUD,
  MTK_HR_BRF,
  MTK_HR_RPV,
  MTK_HR_TIP,
  MTK_CMV_NRT
} MTKt_FileType;

#define MTKT_FILE_TYPE_DESC { "AGP", "GP_GMP", "GRP_RCCM", \
                              "GRP_ELLIPSOID_GM", "GRP_TERRAIN_GM", \
                              "GRP_ELLIPSOID_LM", "GRP_TERRAIN_LM", \
                              "AS_AEROSOL", "AS_LAND", "TC_ALBEDO", \
                              "TC_CLASSIFIERS", "TC_STEREO", \
                              "PP", "CONVENTIONAL", "UNKNOWN","TC_CLOUD", \
                              "MISR_HR_BRF", "MISR_HR_RPV", "MISR_HR_TIP", \
                              "MISR_CMV_NRT"}

/** \brief Core Metadata */
typedef struct MtkCoreMetaData {
  union {
    char **s; /* Array of strings */
    int *i; /* Array of integers */
    double *d; /* Array of doubles */
  } data;
  int num_values;
  enum {
    MTKMETA_CHAR,
    MTKMETA_INT,
    MTKMETA_DOUBLE
  } datatype;
  void *dataptr;                /* Pointer data buffer */
} MtkCoreMetaData;

#define MTK_CORE_METADATA_INIT { {NULL}, 0, MTKMETA_CHAR, NULL }

#define NGRIDCELL 2

/** \brief Time Metadata */
typedef struct MTKt_TimeMetaData {
  MTKt_int32 path;
  MTKt_int32 start_block;
  MTKt_int32 end_block;
  MTKt_char8 camera[3];
  MTKt_int32 number_transform[NBLOCK + 1];
  MTKt_char8 ref_time[NBLOCK + 1][NGRIDCELL][MTKd_DATETIME_LEN];
  MTKt_int32 start_line[NBLOCK + 1][NGRIDCELL];
  MTKt_int32 number_line[NBLOCK + 1][NGRIDCELL];
  MTKt_double coeff_line[NBLOCK + 1][6][NGRIDCELL];
  MTKt_double som_ctr_x[NBLOCK + 1][NGRIDCELL]; /**< In terms of pixels */
  MTKt_double som_ctr_y[NBLOCK + 1][NGRIDCELL]; /**< In terms of pixels */
} MTKt_TimeMetaData;

#define MTKT_TIME_METADATA_INIT { 0, 0, 0, {'\0'}, {0}, {{{'\0'}}}, {{0}}, {{0}}, \
	                             {{{0.0}}}, {{0.0}}, {{0.0}} }

MTKt_status MtkFileToPath( const char *filename,
			   int *path );

MTKt_status MtkFileToPathNC( const char *filename,
			   int *path );

MTKt_status MtkFileToPathHDF( const char *filename,
			   int *path );

MTKt_status MtkFileToPathFid( int32 sid,
			      int *path );

MTKt_status MtkFileToPathNcid( int ncid,
			      int *path );

MTKt_status MtkFileToOrbit( const char *filename,
			    int *orbit );

MTKt_status MtkFileToOrbitNC( const char *filename,
			    int *orbit );

MTKt_status MtkFileToOrbitHDF( const char *filename,
			    int *orbit );

MTKt_status MtkFileToOrbitFid( int32 sd_id,
			       int *orbit );

MTKt_status MtkFileToOrbitNcid( int ncid,
			       int *orbit );

MTKt_status MtkFileToBlockRange( const char *filename,
				 int *start_block,
				 int *end_block );

MTKt_status MtkFileToBlockRangeNC( const char *filename,
				 int *start_block,
				 int *end_block );

MTKt_status MtkFileToBlockRangeHDF( const char *filename,
				 int *start_block,
				 int *end_block );

MTKt_status MtkFileToBlockRangeFid( int32 sid,
				    int *start_block,
				    int *end_block );

MTKt_status MtkFileToBlockRangeNcid( int ncid,
				    int *start_block,
				    int *end_block );

MTKt_status MtkFileGridToResolution( const char *filename,
				     const char *gridname,
				     int *resolution );

MTKt_status MtkFileGridToResolutionNC( const char *filename,
				     const char *gridname,
				     int *resolution );

MTKt_status MtkFileGridToResolutionHDF( const char *filename,
				     const char *gridname,
				     int *resolution );

MTKt_status MtkFileGridToResolutionFid( int32 fid,
				        const char *gridname,
				        int *resolution );

MTKt_status MtkFileGridToResolutionNcid( int ncid,
				        const char *gridname,
				        int *resolution );

MTKt_status MtkFileGridFieldToDataType( const char *filename,
					const char *gridname,
					const char *fieldname,
					MTKt_DataType *datatype );

MTKt_status MtkFileGridFieldToDataTypeNC( const char *filename,
					const char *gridname,
					const char *fieldname,
					MTKt_DataType *datatype );

MTKt_status MtkFileGridFieldToDataTypeHDF( const char *filename,
					const char *gridname,
					const char *fieldname,
					MTKt_DataType *datatype );

MTKt_status MtkFileGridFieldToDataTypeFid( int32 fid,
					   const char *gridname,
					   const char *fieldname,
					   MTKt_DataType *datatype );

MTKt_status MtkFileGridFieldToDataTypeNcid( int ncid,
					   const char *gridname,
					   const char *fieldname,
					   MTKt_DataType *datatype );

MTKt_status MtkFillValueGet( const char *filename,
			     const char *gridname,
			     const char *fieldname,
			     MTKt_DataBuffer *fillbuf );

MTKt_status MtkFillValueGetNC( const char *filename,
			     const char *gridname,
			     const char *fieldname,
			     MTKt_DataBuffer *fillbuf );

MTKt_status MtkFillValueGetHDF( const char *filename,
			     const char *gridname,
			     const char *fieldname,
			     MTKt_DataBuffer *fillbuf );

MTKt_status MtkFillValueGetFid( int32 fid,
				const char *gridname,
				const char *fieldname,
				MTKt_DataBuffer *fillbuf );

MTKt_status MtkFillValueGetNcid( int ncid,
				const char *gridname,
				const char *fieldname,
				MTKt_DataBuffer *fillbuf );

MTKt_status MtkFileToGridList( const char *filename,
                               int *ngrids,
                               char **gridlist[] );

MTKt_status MtkFileToGridListNC( const char *filename,
                               int *ngrids,
                               char **gridlist[] );

MTKt_status MtkFileToGridListHDF( const char *filename,
                               int *ngrids,
                               char **gridlist[] );

MTKt_status MtkFileToGridListFid( int32 fid,
				  int *ngrids,
				  char **gridlist[] );

MTKt_status MtkFileToGridListNcid( int ncid,
				  int *ngrids,
				  char **gridlist[] );
int32
GDinqgridfid(int32 fid, char *objectlist, int32 * strbufsize);


MTKt_status MtkFileGridToFieldList( const char *filename,
                                    const char *gridname,
				    int *nfields,
                                    char **fieldlist[] );

MTKt_status MtkFileGridToFieldListNC( const char *filename,
                                    const char *gridname,
				    int *nfields,
                                    char **fieldlist[] );

MTKt_status MtkFileGridToFieldListHDF( const char *filename,
                                    const char *gridname,
				    int *nfields,
                                    char **fieldlist[] );

MTKt_status MtkFileGridToFieldListFid( int32 Fid,
				       const char *gridname,
				       int *nfields,
				       char **fieldlist[] );

MTKt_status MtkFileGridToFieldListNcid( int ncid,
				       const char *gridname,
				       int *nfields,
				       char **fieldlist[] );

MTKt_status MtkFileGridToNativeFieldList( const char *filename,
					  const char *gridname,
					  int *nfields,
					  char **fieldlist[] );

MTKt_status MtkFileGridToNativeFieldListNC( const char *filename,
					  const char *gridname,
					  int *nfields,
					  char **fieldlist[] );

MTKt_status MtkFileGridToNativeFieldListHDF( const char *filename,
					  const char *gridname,
					  int *nfields,
					  char **fieldlist[] );

MTKt_status MtkFileGridToNativeFieldListFid( int32 Fid,
					     const char *gridname,
					     int *nfields,
					     char **fieldlist[] );

MTKt_status MtkFileGridToNativeFieldListNcid( int ncid,
					     const char *gridname,
					     int *nfields,
					     char **fieldlist[] );

MTKt_status MtkMakeFilename( const char *basedir,
			     const char *product,
			     const char *camera,
			     int path,
			     int orbit,
			     const char *version,
			     char **filename );

MTKt_status MtkFindFileList( const char *searchdir,
			     const char *product,
			     const char *camera,
			     const char *path,
			     const char *orbit,
			     const char *version,
			     int *filecnt,
			     char **filenames[] );

MTKt_status MtkFileLGID( const char *filename,
			 char **lgid );

MTKt_status MtkFileLGIDNC( const char *filename,
			 char **lgid );

MTKt_status MtkFileLGIDHDF( const char *filename,
			 char **lgid );

MTKt_status MtkFileLGIDFid( int32 sds_id,
			    char **lgid );

MTKt_status MtkFileLGIDNcid( int ncid,
			    char **lgid );

MTKt_status MtkFileType( const char *filename,
			 MTKt_FileType *filetype );

MTKt_status MtkFileTypeNC( const char *filename,
			 MTKt_FileType *filetype );

MTKt_status MtkFileTypeHDF( const char *filename,
			 MTKt_FileType *filetype );

MTKt_status MtkFileTypeFid( int32 Fid,
			    MTKt_FileType *filetype );

MTKt_status MtkFileTypeNcid( int ncid,
			    MTKt_FileType *filetype );

MTKt_status MtkFileVersion( const char *filename,
			    char *fileversion );

MTKt_status MtkFileVersionNC( const char *filename,
			    char *fileversion );

MTKt_status MtkFileVersionHDF( const char *filename,
			    char *fileversion );

MTKt_status MtkFileVersionFid( int32 sd_id,
			       char *fileversion );

MTKt_status MtkFileVersionNcid( int ncid,
			       char *fileversion );

MTKt_status MtkGridAttrGet( const char *filename,
			    const char *gridname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkGridAttrGetNC( const char *filename,
			    const char *gridname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkGridAttrGetHDF( const char *filename,
			    const char *gridname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkGridAttrGetFid( int32 fid,
			       const char *gridname,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );

MTKt_status MtkGridAttrGetNcid( int ncid,
			       const char *gridname,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileGridFieldToDimList( const char *filename,
				       const char *gridname,
				       const char *fieldname,
				       int *dimcnt,
				       char **dimlist[],
				       int **dimsize );

MTKt_status MtkFileGridFieldToDimListNC( const char *filename,
				       const char *gridname,
				       const char *fieldname,
				       int *dimcnt,
				       char **dimlist[],
				       int **dimsize );

MTKt_status MtkFileGridFieldToDimListHDF( const char *filename,
				       const char *gridname,
				       const char *fieldname,
				       int *dimcnt,
				       char **dimlist[],
				       int **dimsize );

MTKt_status MtkFileGridFieldToDimListFid( int32 Fid,
					  const char *gridname,
					  const char *fieldname,
					  int *dimcnt,
					  char **dimlist[],
					  int **dimsize );

MTKt_status MtkFileGridFieldToDimListNcid( int ncid,
					  const char *gridname,
					  const char *fieldname,
					  int *dimcnt,
					  char **dimlist[],
					  int **dimsize );

MTKt_status MtkFileCoreMetaDataRaw( const char *filename,
                                    char **coremeta );
MTKt_status MtkFileCoreMetaDataRawNC( const char *filename,
                                      char **coremeta );
MTKt_status MtkFileCoreMetaDataRawHDF( const char *filename,
                                       char **coremeta );

MTKt_status MtkFileCoreMetaDataRawFid( int32 sds_id,
				       char **coremeta );

MTKt_status MtkFileCoreMetaDataRawNcid( int ncid,
				       char **coremeta );

MTKt_status MtkFileCoreMetaDataQuery( const char *filename,
                                      int *nparam,
				      char ***paramlist );

MTKt_status MtkFileCoreMetaDataQueryNC( const char *filename,
                                      int *nparam,
				      char ***paramlist );

MTKt_status MtkFileCoreMetaDataQueryHDF( const char *filename,
                                      int *nparam,
				      char ***paramlist );

MTKt_status MtkFileCoreMetaDataQueryFid( int32 sd_id,
					 int *nparam,
					 char ***paramlist );

MTKt_status MtkFileCoreMetaDataQueryNcid( int ncid,
					 int *nparam,
					 char ***paramlist );

MTKt_status MtkFileCoreMetaDataGet( const char *filename,
				    const char *param,
				    MtkCoreMetaData *metadata );
MTKt_status MtkFileCoreMetaDataGetNC( const char *filename,
				    const char *param,
				    MtkCoreMetaData *metadata );
MTKt_status MtkFileCoreMetaDataGetHDF( const char *filename,
				    const char *param,
				    MtkCoreMetaData *metadata );

MTKt_status MtkFileCoreMetaDataGetFid( int32 sd_id,
				       const char *param,
				       MtkCoreMetaData *metadata );

MTKt_status MtkFileCoreMetaDataGetNcid( int ncid,
				       const char *param,
				       MtkCoreMetaData *metadata );

MTKt_status MtkCoreMetaDataFree( MtkCoreMetaData *metadata );

MTKt_status MtkFileAttrGet( const char *filename,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileAttrGetHDF( const char *filename,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileAttrGetNC( const char *filename,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileAttrGetFid( int32 sds_id,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileAttrGetNcid( int ncid,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );

MTKt_status MtkFileGridFieldCheck( const char *filename,
				   const char *gridname,
				   const char *fieldname );

MTKt_status MtkFileGridFieldCheckNC( const char *filename,
				   const char *gridname,
				   const char *fieldname );

MTKt_status MtkFileGridFieldCheckHDF( const char *filename,
				   const char *gridname,
				   const char *fieldname );

MTKt_status MtkFileGridFieldCheckFid( int32 Fid,
				      const char *gridname,
				      const char *fieldname );

MTKt_status MtkFileGridFieldCheckNcid( int ncid,
				      const char *gridname,
				      const char *fieldname );

MTKt_status MtkFileAttrList( const char *filename,
			     int *num_attrs,
			     char **attrlist[] );

MTKt_status MtkFileAttrListNC( const char *filename,
			     int *num_attrs,
			     char **attrlist[] );

MTKt_status MtkFileAttrListHDF( const char *filename,
			     int *num_attrs,
			     char **attrlist[] );

MTKt_status MtkFileAttrListFid( int32 sd_id,
				int *num_attrs,
				char **attrlist[] );

MTKt_status MtkFileAttrListNcid( int ncid,
				int *num_attrs,
				char **attrlist[] );

MTKt_status MtkGridAttrList( const char *filename,
			     const char *gridname,
			     int *num_attrs,
			     char **attrlist[] );
			     
MTKt_status MtkGridAttrListNC( const char *filename,
			     const char *gridname,
			     int *num_attrs,
			     char **attrlist[] );
			     
MTKt_status MtkGridAttrListHDF( const char *filename,
			     const char *gridname,
			     int *num_attrs,
			     char **attrlist[] );
			     
MTKt_status MtkGridAttrListFid( int32 fid,
				const char *gridname,
				int *num_attrs,
				char **attrlist[] );
        
MTKt_status MtkGridAttrListNcid( int ncid,
				const char *gridname,
				int *num_attrs,
				char **attrlist[] );
        
MTKt_status MtkFieldAttrList( const char *filename,
           const char *fieldname,
			     int *num_attrs,
			     char **attrlist[] );

MTKt_status MtkFieldAttrListNC( const char *filename,
           const char *fieldname,
			     int *num_attrs,
			     char **attrlist[] );

MTKt_status MtkFieldAttrListHDF( const char *filename,
           const char *fieldname,
			     int *num_attrs,
			     char **attrlist[] );
   
MTKt_status MtkFieldAttrListFid( int32 fid,
        const char *fieldname,
				int *num_attrs,
				char **attrlist[] );

MTKt_status MtkFieldAttrListNcid( int ncid,
        const char *fieldname,
				int *num_attrs,
				char **attrlist[] );

MTKt_status MtkFieldAttrGet( const char *filename,
          const char *fieldname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFieldAttrGetNC( const char *filename,
          const char *fieldname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFieldAttrGetHDF( const char *filename,
          const char *fieldname,
			    const char *attrname,
			    MTKt_DataBuffer *attrbuf );

MTKt_status MtkFieldAttrGetFid( int32 sds_id,
             const char *fieldname,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );                
			     
MTKt_status MtkFieldAttrGetNcid( int ncid,
             const char *fieldname,
			       const char *attrname,
			       MTKt_DataBuffer *attrbuf );                
			     
MTKt_status MtkFileBlockMetaList( const char *filename,
                 int *nblockmeta,
                 char ***blockmetalist );
                 
MTKt_status MtkFileBlockMetaListFid( int32 file_id,
				     int *nblockmeta,
				     char ***blockmetalist );
                 
MTKt_status MtkFileBlockMetaFieldList( const char *filename,
                 const char *blockmetaname,
                 int *nfields,
                 char **fieldlist[] );
                 
MTKt_status MtkFileBlockMetaFieldListFid( int32 file_id,
					  const char *blockmetaname,
					  int *nfields,
					  char **fieldlist[] );
                 
MTKt_status MtkFileBlockMetaFieldRead( const char *filename,
                 const char *blockmetaname,
                 const char *fieldname,
                 MTKt_DataBuffer *blockmetabuf );
                 
MTKt_status MtkFileBlockMetaFieldReadFid( int32 file_id,
					  const char *blockmetaname,
					  const char *fieldname,
					  MTKt_DataBuffer *blockmetabuf );
                 
MTKt_status MtkTimeMetaRead( const char *filename,
                             MTKt_TimeMetaData *time_metadata );

MTKt_status MtkTimeMetaReadFid( int32 hdf_id,
				int32 sd_id,
				MTKt_TimeMetaData *time_metadata );

#endif /* MISRFILEQUERY_H */
