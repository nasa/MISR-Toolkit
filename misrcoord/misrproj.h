#ifndef MISRPROJ_H
#define MISRPROJ_H

/* Defines */

#define STRLEN	   200
#define NBLOCK	   180
#define NOFFSET	   NBLOCK - 1
#define R2D	   57.2957795131
#define D2R        1.745329251994328e-2
#define NPROJ	   15

/* Prototypes */

int misr_init(
const int	   nblock,         /* Number of blocks */
const int          nline,          /* Number of lines in a block */
const int          nsample,        /* Number of samples in a block */
const float	   relOff[NOFFSET],/* Block offsets */
const double	   ulc_coord[],    /* Upper left corner coord. in meters */
const double	   lrc_coord[]     /* Lower right corner coord. in meters */
);

int misrfor(
const double	   x,		   /* Output SOM X coordinate */
const double	   y,		   /* Output SOM Y coordinate */
int*		   block,	   /* Input block */
float*		   line,	   /* Input line */
float*		   sample	   /* Input sample */
);

int misrinv(
const int	   block,	   /* Input block */
const float	   line,	   /* Input line */
const float	   sample,	   /* Input sample */
double*		   x,		   /* Output SOM X coordinate */
double*		   y		   /* Output SOM Y coordinate */
);

#endif /* MISRPROJ_H */
