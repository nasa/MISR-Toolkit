/*===========================================================================
=                                                                           =
=                              MisrProjParam                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRPROJPARAM_H
#define MISRPROJPARAM_H

/* MISR's max resolution, line and sample for a block */

#define MAXRESOLUTION         275
#define MINRESOLUTION         17600
#define MAXNLINE              (RESOLUTION / MAXRESOLUTION * NLINE)
#define MAXNSAMPLE            (RESOLUTION / MAXRESOLUTION * NSAMPLE)

/* MISR projection parameters for path 1 at 1100 meter resolution. */
/* All other paths and resolutions are derived from these parameters. */

#define NRES                  6
#define NPATH                 233
#define PATHNUM               1
#define PROJCODE              22
#define ZONECODE              -1
#define SPHERECODE            12
#define PP1_SMAJOR            6378137.0
#define PP2_SMINOR            -0.006694348
#define PP3_UNUSED            0.0
#define PP4_INCANG            98018013.7520
#define PP5_ASCLONG           127045037.928240340
#define PP6_UNUSED            0.0
#define PP7_FE                0.0
#define PP8_FN                0.0
#define PP9_PSREV             98.88
#define PP10_LRAT             0.0
#define PP11_PFLAG            0.0
#define PP12_BLOCKS           180.0
#define PP13_SOMA             0.0
#define PP14_UNUSED           0.0
#define PP15_UNUSED           0.0
#define NBLOCK                180
#define NLINE                 128
#define NSAMPLE               512
#define RESOLUTION            1100
#define ULC_SOMX              7460750.0
#define ULC_SOMY              1090650.0
#define LRC_SOMX              7601550.0
#define LRC_SOMY              527450.0

/* Relative block offsets */

#define RELOFFSET { \
	0.0, 16.0, 0.0, 16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, \
	0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -16.0, 0.0, 0.0, 0.0, \
	-16.0, 0.0, 0.0, -16.0, 0.0, 0.0, -16.0, 0.0, -16.0, \
	0.0, -16.0, 0.0, -16.0, -16.0, 0.0, -16.0, 0.0, -16.0, \
	-16.0, 0.0, -16.0, -16.0, -16.0, 0.0, -16.0, -16.0, -16.0, \
	-16.0, 0.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -32.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -32.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, -16.0, -16.0, 0.0, -16.0, \
	-16.0, -16.0, -16.0, -16.0, 0.0, -16.0, -16.0, -16.0, 0.0, \
	-16.0, -16.0, 0.0, -16.0, 0.0, -16.0, -16.0, 0.0, -16.0, \
	0.0, -16.0, 0.0, 0.0, -16.0, 0.0, -16.0, 0.0, 0.0, -16.0, \
	0.0, 0.0, 0.0, 0.0, -16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
	16.0, 0.0, 0.0, 16.0, 0.0, 0.0, 16.0, 0.0 \
}

/* GCTP projection parameter array */

#define PROJPARAM { \
   PP1_SMAJOR, PP2_SMINOR, PP3_UNUSED, PP4_INCANG, PP5_ASCLONG, \
   PP6_UNUSED, PP7_FE, PP8_FN, PP9_PSREV, PP10_LRAT, PP11_PFLAG, \
   PP12_BLOCKS, PP13_SOMA, PP14_UNUSED, PP15_UNUSED \
}

/* Upper left corner and lower right corner of the first block */

#define ULC { ULC_SOMX, ULC_SOMY }
#define LRC { LRC_SOMX, LRC_SOMY }


/** \brief MISR Projection Parameters */

typedef struct {
  int path;			/**< MISR path number */
  long long projcode;		/**< GCTP projection code */
  long long zonecode;		/**< GCTP zone code */
  long long spherecode;		/**< GCTP sphere code */
  double projparam[15];		/**< GCTP projection parameters */
  double ulc[2];		/**< MISR ulc_xy of first block */
  double lrc[2];		/**< MISR lrc_xy of first block */
  int nblock;			/**< MISR number blocks */
  int nline;			/**< MISR number lines */
  int nsample;			/**< MISR number samples */
  float reloffset[179];		/**< MISR relative block offset */
  int resolution;		/**< MISR resolution */
} MTKt_MisrProjParam;

/* MTK MISR projection parameter initialization macro */

#define MTKT_MISRPROJPARAM_INIT { PATHNUM, PROJCODE, ZONECODE, SPHERECODE, \
   PROJPARAM, ULC, LRC, NBLOCK, NLINE, NSAMPLE, RELOFFSET, RESOLUTION }

#endif /* MISRPROJPARAM_H */
