/*===========================================================================
=                                                                           =
=                              MisrOrbitPath                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRORBITPATH_H
#define MISRORBITPATH_H

#include "MisrSetRegion.h"
#include "MisrError.h"

/* P(Nref) is the path number corresponding to a reference
   orbit number Nref.  It is assumed that Nref is the earliest
   orbit for which (a) the orbit has stabliized and (b) we
   would be interested in looking at data. */

/*static const int Nref = 1100;*/

/*   J(Nref) is the Julian date corresponding to the start of
   orbit Nref, or alternatively, the GMT date and time of
   the start of orbit Nref, fromwhich J(Nref) can be calculated. */

/* Start of orbit 1100 is 2000-03-02 21:14:00 */
/*static const double JNref = 2451606.38472;*/

/* Orbit Start Times */
#define MISR_ORBIT_REF_995 2451599.189583333 /* 995 24 Feb 2000 16:33:00 */
#define MISR_ORBIT_REF_DT 1000
#define MISR_ORBIT_REF { \
  2451599.51744, /* 1000  25 Feb 2000 00:25:07 */ \
  2451668.18819, /* 2000   3 May 2000 16:31 */    \
  2451736.85764, /* 3000   11 Jul 2000 08:35 */   \
  2451805.52639, /* 4000   18 Sep 2000 00:38 */   \
  2451874.19514, /* 5000   25 Nov 2000 16:41 */   \
  2451942.86389, /* 6000   2 Feb 2001 08:44 */    \
  2452011.53194, /* 7000   12 Apr 2001 00:46 */   \
  2452080.2,     /* 8000   19 Jun 2001 16:48 */   \
  2452148.86875, /* 9000   27 Aug 2001 08:51 */   \
  2452217.53681, /* 10000  4 Nov 2001 00:53 */    \
  2452286.20486, /* 11000  11 Jan 2002 16:55 */   \
  2452354.87361, /* 12000  21 Mar 2002 08:58 */   \
  2452423.54306, /* 13000  29 May 2002 01:02 */   \
  2452492.2125,  /* 14000  5 Aug 2002 17:06 */    \
  2452560.88194, /* 15000  13 Oct 2002 09:10 */   \
  2452629.55139, /* 16000  21 Dec 2002 01:14 */   \
  2452698.22083, /* 17000  27 Feb 2003 17:18 */   \
  2452766.88958, /* 18000  7 May 2003 09:21 */    \
  2452835.55903, /* 19000  15 Jul 2003 01:25 */   \
  2452904.22847, /* 20000  21 Sep 2003 17:29 */   \
  2452972.89861, /* 21000  29 Nov 2003 09:34 */   \
  2453041.56806, /* 22000  6 Feb 2004 01:38 */    \
  2453110.2375,  /* 23000  14 Apr 2004 17:42 */   \
  2453178.90694, /* 24000  22 Jun 2004 09:46 */   \
  2453247.57569, /* 25000  30 Aug 2004 01:49 */   \
  2453316.24514, /* 26000  6 Nov 2004 17:53 */    \
  2453384.91528, /* 27000  14 Jan 2005 09:58 */   \
  2453453.58472, /* 28000  24 Mar 2005 02:02 */   \
  2453522.25417, /* 29000  31 May 2005 18:06 */   \
  2453590.92361, /* 30000  8 Aug 2005 10:10 */    \
  2453659.59306, /* 31000  16 Oct 2005 02:14 */   \
  2453728.2625,  /* 32000  23 Dec 2005 18:18 */   \
  2453796.93194, /* 33000  2 Mar 2006 10:22 */    \
  2453865.60139, /* 34000  10 May 2006 02:26 */   \
  2453934.27152, /* 35000  17 Jul 2006 18:31 */   \
  2454002.94097, /* 36000  24 Sep 2006 10:35 */   \
  2454071.61042, /* 37000  2 Dec 2006 02:39 */    \
  2454140.28055, /* 38000  9 Feb 2007 18:44 */    \
  2454208.95,    /* 39000  17 Apr 2007 10:48 */   \
  2454277.61944, /* 40000  26 Jun 2007 02:52 */   \
  2454346.28889, /* 41000  2 Sep 2007 18:56 */    \
  2454414.95833, /* 42000  10 Nov 2007 11:00 */   \
  2454483.62778, /* 43000  18 Jan 2008 03:04 */   \
  2454552.29722, /* 44000  26 Mar 2008 19:08 */  \
  2454620.96736, /* 45000  3 Jun 2008 11:13 */   \
  2454689.63681, /* 46000  11 Aug 2008 03:17 */  \
  2454758.306250, /* 47000  18 Oct 2008 19:21 */ \
  2454826.975694, /* 48000  26 Dec 2008 11:25 */ \
  2454895.645139, /* 49000  05 Mar 2009 03:29 */ \
  2454964.314583, /* 50000  12 May 2009 19:33 */ \
  2455032.984028, /* 51000  20 Jul 2009 11:37 */ \
  2455101.653472, /* 52000  27 Sep 2009 03:41 */ \
  2455170.322917, /* 53000  04 Dec 2009 19:45 */ \
  2455238.992361, /* 54000  11 Jul 2010 11:49 */ \
  2455307.662500, /* 55000  21 Apr 2010 03:54 */ \
  2455376.331944, /* 56000  28 Jun 2010 19:58 */ \
  2455445.001389, /* 57000  05 Sep 2010 12:02 */ \
  2455513.670833, /* 58000  13 Nov 2010 04:06 */ \
  2455582.340278, /* 59000  20 Jan 2011 20:10 */ \
  2455651.009722, /* 60000  30 Mar 2011 12:14 */ \
  2455719.679167, /* 61000  07 Jun 2011 04:18 */ \
  2455788.348611, /* 62000  14 Aug 2011 20:22 */ \
  2455857.018750, /* 63000  22 Oct 2011 12:27 */ \
  2455925.688194, /* 64000  30 Dec 2011 04:31 */ \
  2455994.357639, /* 65000  07 Mar 2012 20:35 */ \
  2456063.027083, /* 66000  15 May 2012 12:39 */ \
  2456131.696528, /* 67000  23 Jul 2012 04:43 */ \
  2456200.365972, /* 68000  29 Sep 2012 20:47 */ \
  2456269.035417, /* 69000  07 Dec 2012 12:51 */ \
  2456337.704861, /* 70000  14 Feb 2013 04:55 */ \
  2456406.375000, /* 71000  23 Apr 2013 21:00 */ \
  2456475.044444, /* 72000  01 Jul 2013 13:04 */ \
  2456543.713889, /* 73000  08 Sep 2013 05:08 */ \
  2456612.383333  /* 74000  15 Nov 2013 21:12 */ \
}


MTKt_status MtkLatLonToPathList( double lat_dd,
				 double lon_dd,
				 int *pathcnt,
				 int **pathlist );

MTKt_status MtkRegionToPathList( MTKt_Region region,
				 int *pathcnt,
				 int **pathlist );

MTKt_status MtkRegionPathToBlockRange( MTKt_Region region,
				       int path,
				       int *start_block,
				       int *end_block );

MTKt_status MtkOrbitToPath( int orbit,
                            int *path );

MTKt_status MtkTimeToOrbitPath( const char *datetime,
                                int *orbit,
                                int *path );

MTKt_status MtkTimeRangeToOrbitList( const char *start_time,
                                     const char *end_time,
                                     int *orbitcnt,
                                     int **orbitlist );

MTKt_status MtkPathTimeRangeToOrbitList( int path,
                                         const char *start_time,
				         const char *end_time,
				         int *orbitcnt,
				         int **orbitlist );

MTKt_status MtkOrbitToTimeRange( int orbit,
                                 char start_time[MTKd_DATETIME_LEN],
				 char end_time[MTKd_DATETIME_LEN] );

#endif /* MISRORBITPATH_H */
