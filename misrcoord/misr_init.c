#include "misrproj.h"		   /* Prototype for this function */
#include "errormacros.h"	   /* Error macros */

int nb;
int nl_var;
int ns;
float absOffset[NBLOCK]; 
float relOffset[NBLOCK-1]; 
double ulc[2];
double lrc[2];
double sx;
double sy;
double xc;
double yc;

#define FUNC_NAMEm "misr_init"

int misr_init(
const int	   nblock,         /* Number of blocks */
const int          nline,          /* Number of lines in a block */
const int          nsample,        /* Number of samples in a block */
const float	   relOff[NOFFSET],/* Block offsets */
const double	   ulc_coord[],    /* Upper left corner coord. in meters */
const double	   lrc_coord[]     /* Lower right corner coord. in meters */
)
{
  int		   i;		   /* Offset index */
  char		   msg[STRLEN];	   /* Warning message */

/* Argument checks */

  if (nblock < 1 || nblock > NBLOCK) {
    sprintf(msg,"nblock is out of range (1 < %d < %d)", nblock, NBLOCK);
    WRN_LOG_JUMP(msg);
  }

/* Convert relative offsets to absolute offsets */

  absOffset[0] = 0.0;
  for (i = 1; i < NBLOCK; i++) {
    absOffset[i] = absOffset[i-1] + relOff[i-1];
    relOffset[i-1] = relOff[i-1];
  }

/* Set ulc and lrc SOM coordinates */
/* Note; ulc y and lrc y are reversed in the structural metadata. */

  ulc[0] = ulc_coord[0];
  ulc[1] = lrc_coord[1];
  lrc[0] = lrc_coord[0];
  lrc[1] = ulc_coord[1];

/* Set number of blocks, lines and samples */

  nb = nblock;
  nl_var = nline;
  ns = nsample;

/* Compute pixel size in ulc/lrc units (meters) */

  sx = (lrc[0] - ulc[0]) / nl_var;
  sy = (lrc[1] - ulc[1]) / ns;

/* Adjust ulc to be in the center of the pixel */

  xc = ulc[0] + sx / 2.0;
  yc = ulc[1] + sy / 2.0;

  return(0);

 ERROR_HANDLE:
  return(1);
}
