/*===========================================================================
=                                                                           =
=                                MisrUtil                                   =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRUTIL_H
#define MISRUTIL_H

#include <netcdf.h>
#include <MisrError.h>
#include <mfhdf.h>
#ifndef _WIN32
#include <sys/types.h>
#endif

#define MAXDIMS 10
#define MAXSTR  80

#define MTKd_NDATATYPE 13
#define MTKd_DataType { "void", "char8", "uchar8", "int8", "uint8", \
                        "int16", "uint16", "int32", "uint32", \
                        "int64", "uint64", "float", "double" }
#define MTKd_DataSize { 0, 1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8 }

typedef enum {
  MTKe_void=0,
  MTKe_char8,
  MTKe_uchar8,
  MTKe_int8,
  MTKe_uint8,
  MTKe_int16,
  MTKe_uint16,
  MTKe_int32,
  MTKe_uint32,
  MTKe_int64,
  MTKe_uint64,
  MTKe_float,
  MTKe_double
} MTKt_DataType;

#ifdef _WIN32
typedef char MTKt_char8;
typedef unsigned char MTKt_uchar8;
typedef char MTKt_int8;
typedef unsigned char MTKt_uint8;
typedef short MTKt_int16;
typedef unsigned short MTKt_uint16;
typedef int MTKt_int32;
typedef unsigned int MTKt_uint32;
typedef long int MTKt_int64;
typedef unsigned long int MTKt_uint64;
typedef float MTKt_float;
typedef double MTKt_double;
#else
typedef char MTKt_char8;
typedef unsigned char MTKt_uchar8;
typedef int8_t MTKt_int8;
typedef u_int8_t MTKt_uint8;
typedef int16_t MTKt_int16;
typedef u_int16_t MTKt_uint16;
typedef int32_t MTKt_int32;
typedef u_int32_t MTKt_uint32;
typedef int64_t MTKt_int64;
typedef u_int64_t MTKt_uint64;
typedef float MTKt_float;
typedef double MTKt_double;
#endif

/** \brief 2-dimensional Data Buffer Type Union */
typedef union {
  void **v;
  MTKt_char8 **c8;
  MTKt_uchar8 **uc8;
  MTKt_int8 **i8;
  MTKt_uint8 **u8;
  MTKt_int16 **i16;
  MTKt_uint16 **u16;
  MTKt_int32 **i32;
  MTKt_uint32 **u32;
  MTKt_int64 **i64;
  MTKt_uint64 **u64;
  MTKt_float **f;
  MTKt_double **d;
} MTKt_DataBufferType;

/** \brief 2-dimensional Data Buffer */
typedef struct {
  int nline;			/**< Number of lines */
  int nsample;			/**< Number of samples */
  int datasize;			/**< Data element size (bytes) */
  MTKt_DataType datatype;	/**< Data type (enumeration) */
  MTKt_boolean imported;	/**< Imported dataptr flag */
  MTKt_DataBufferType data;	/**< Data type access union */
  void **vdata;			/**< Row major 2D array with Illiffe vector */
  void *dataptr;		/**< Pointer data buffer */
} MTKt_DataBuffer;

#define MTKT_DATABUFFER_INIT { 0, 0, 0, MTKe_void, MTK_FALSE, {0}, NULL, NULL }

/** \brief 3-dimensional Data Buffer Type Union */
typedef union {
  void ***v;
  MTKt_char8 ***c8;
  MTKt_uchar8 ***uc8;
  MTKt_int8 ***i8;
  MTKt_uint8 ***u8;
  MTKt_int16 ***i16;
  MTKt_uint16 ***u16;
  MTKt_int32 ***i32;
  MTKt_uint32 ***u32;
  MTKt_int64 ***i64;
  MTKt_uint64 ***u64;
  MTKt_float ***f;
  MTKt_double ***d;
} MTKt_DataBufferType3D;

/** \brief 3-dimensional Data Buffer */
typedef struct {
  int nblock;			/**< Number of blocks */
  int nline;			/**< Number of lines */
  int nsample;			/**< Number of samples */
  int datasize;			/**< Data element size (bytes) */
  MTKt_DataType datatype;	/**< Data type (enumeration) */
  MTKt_DataBufferType3D data;	/**< Data type access union */
  void ***vdata;	       	/**< Row major 3D array with Illiffe vector */
  void *dataptr;		/**< Pointer data buffer */
} MTKt_DataBuffer3D;

#define MTKT_DATABUFFER3D_INIT { 0, 0, 0, 0, 0, {0}, NULL, NULL }

/* Time conversion constants */
#define EPOCH_DAY 2448988.5
#define EPOCH_DAY_FRACTION  0.0003125
#define SECONDSperDAY 86400.0
#define SECONDSperHOUR 3600.0
#define SECONDSperMINUTE 60.0

#define MTKd_DATETIME_LEN 28
        
               /* JD      TAI - UTC */
#define LEAP_SECONDS {{ 2441317.5,  10.0 },  /* 1972 JAN  1 =JD 2441317.5  TAI-UTC=  10.0000000 S */ \
                      { 2441499.5,  11.0 },  /* 1972 JUL  1 =JD 2441499.5  TAI-UTC=  11.0000000 S */ \
                      { 2441683.5,  12.0 },  /* 1973 JAN  1 =JD 2441683.5  TAI-UTC=  12.0000000 S */ \
                      { 2442048.5,  13.0 },  /* 1974 JAN  1 =JD 2442048.5  TAI-UTC=  13.0000000 S */ \
                      { 2442413.5,  14.0 },  /* 1975 JAN  1 =JD 2442413.5  TAI-UTC=  14.0000000 S */ \
                      { 2442778.5,  15.0 },  /* 1976 JAN  1 =JD 2442778.5  TAI-UTC=  15.0000000 S */ \
                      { 2443144.5,  16.0 },  /* 1977 JAN  1 =JD 2443144.5  TAI-UTC=  16.0000000 S */ \
                      { 2443509.5,  17.0 },  /* 1978 JAN  1 =JD 2443509.5  TAI-UTC=  17.0000000 S */ \
                      { 2443874.5,  18.0 },  /* 1979 JAN  1 =JD 2443874.5  TAI-UTC=  18.0000000 S */ \
                      { 2444239.5,  19.0 },  /* 1980 JAN  1 =JD 2444239.5  TAI-UTC=  19.0000000 S */ \
                      { 2444786.5,  20.0 },  /* 1981 JUL  1 =JD 2444786.5  TAI-UTC=  20.0000000 S */ \
                      { 2445151.5,  21.0 },  /* 1982 JUL  1 =JD 2445151.5  TAI-UTC=  21.0000000 S */ \
                      { 2445516.5,  22.0 },  /* 1983 JUL  1 =JD 2445516.5  TAI-UTC=  22.0000000 S */ \
                      { 2446247.5,  23.0 },  /* 1985 JUL  1 =JD 2446247.5  TAI-UTC=  23.0000000 S */ \
                      { 2447161.5,  24.0 },  /* 1988 JAN  1 =JD 2447161.5  TAI-UTC=  24.0000000 S */ \
                      { 2447892.5,  25.0 },  /* 1990 JAN  1 =JD 2447892.5  TAI-UTC=  25.0000000 S */ \
                      { 2448257.5,  26.0 },  /* 1991 JAN  1 =JD 2448257.5  TAI-UTC=  26.0000000 S */ \
                      { 2448804.5,  27.0 },  /* 1992 JUL  1 =JD 2448804.5  TAI-UTC=  27.0000000 S */ \
                      { 2449169.5,  28.0 },  /* 1993 JUL  1 =JD 2449169.5  TAI-UTC=  28.0000000 S */ \
                      { 2449534.5,  29.0 },  /* 1994 JUL  1 =JD 2449534.5  TAI-UTC=  29.0000000 S */ \
                      { 2450083.5,  30.0 },  /* 1996 JAN  1 =JD 2450083.5  TAI-UTC=  30.0000000 S */ \
                      { 2450630.5,  31.0 },  /* 1997 JUL  1 =JD 2450630.5  TAI-UTC=  31.0000000 S */ \
                      { 2451179.5,  32.0 },  /* 1999 JAN  1 =JD 2451179.5  TAI-UTC=  32.0000000 S */ \
                      { 2453736.5,  33.0 },  /* 2006 JAN  1 =JD 2453736.5  TAI-UTC=  33.0000000 S */ \
                      { 2454832.5,  34.0 },  /* 2009 JAN  1 =JD 2454832.5  TAI-UTC=  34.0000000 S */ \
                      { 2456109.5,  35.0 },  /* 2012 JUL  1 =JD 2456109.5  TAI-UTC=  35.0000000 S */ \
                      { 2457204.5,  36.0 },  /* 2015 JUL  1 =JD 2457204.5  TAI-UTC=  36.0000000 S */ \
                      { 2457754.5,  37.0 }}  /* 2017 JAN  1 =JD 2457754.5  TAI-UTC=  37.0000000 S */

MTKt_status MtkDataBufferAllocate( int nline,
				   int nsample,
				   MTKt_DataType datatype,
				   MTKt_DataBuffer *databuf );

MTKt_status MtkDataBufferAllocate3D( int nblock,
				     int nline,
				     int nsample,
				     MTKt_DataType datatype,
				     MTKt_DataBuffer3D *databuf );

MTKt_status MtkDataBufferFree( MTKt_DataBuffer *databuf );

MTKt_status MtkDataBufferFree3D( MTKt_DataBuffer3D *databuf );

MTKt_status MtkDataBufferImport( int nline,
				 int nsample,
				 MTKt_DataType datatype,
				 void *dataptr,
				 MTKt_DataBuffer *databuf );

MTKt_status MtkHdfToMtkDataTypeConvert( int32 hdf_datatype,
                                        MTKt_DataType *datatype );

MTKt_status MtkNcToMtkDataTypeConvert( nc_type nc_datatype,
                                       MTKt_DataType *datatype );

MTKt_status MtkParseFieldname( const char *fieldname,
			       char **basefieldname,
			       int *ndim,
			       int **dimlist );

MTKt_status MtkStringListFree( int strcnt,
			       char **strlist[] );

MTKt_status MtkCalToJulian( int y,
                            int m,
                            int d,
                            int h,
                            int mn,
                            int s,
                            double *julian );

MTKt_status MtkJulianToCal( double jd,
                            int *year,
                            int *month,
                            int *day,
                            int *hour,
                            int *min,
                            int *sec );

MTKt_status MtkDateTimeToJulian( const char *datetime,
                                 double *jd );

MTKt_status MtkJulianToDateTime( double jd,
				 char datetime[MTKd_DATETIME_LEN] );
				 
MTKt_status MtkTaiJdToTai( double jdTAI[2],
                 double *secTAI93 );
                 
MTKt_status MtkTaiJdToUtcJd( double jdTAI[2],
                 double jdUTC[2] );
                 
MTKt_status MtkTaiToTaiJd( double secTAI93,
                 double jdTAI[2] );
                 
MTKt_status MtkTaiToUtc( double secTAI93,
                 char utc_datetime[MTKd_DATETIME_LEN] );
                 
MTKt_status MtkUtcJdToTaiJd( double jdUTC[2],
                 double jdTAI[2] );
                 
MTKt_status MtkUtcJdToUtc( double jdUTCin[2],
                 char utc_datetime[MTKd_DATETIME_LEN] );
                 
MTKt_status MtkUtcToTai( char utc_datetime[MTKd_DATETIME_LEN],
                 double *secTAI93 );
                 
MTKt_status MtkUtcToUtcJd( char utc_datetime[MTKd_DATETIME_LEN],
                 double jdUTC[2] );


typedef struct {
  int gid;  // netcdf group id
  int varid;  // netcdf variable id
} MTKt_ncvarid;

MTKt_status MtkNCVarId(int Ncid, const char *Name, MTKt_ncvarid *Var);

char *MtkVersion(void);


#ifdef _MSC_VER
char* win_strcasestr(const char *s, const char *find);
#define strncasecmp _strnicmp
#define strdup _strdup
#define strlwr _strlwr
#define strcasestr win_strcasestr
#endif

#endif /* MISRUTIL_H */
