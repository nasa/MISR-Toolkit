#include "misrproj.h"		   /* Prototype for this function */
#include "errormacros.h"	   /* Error macros */

extern int nb;
extern int nl_var;
extern int ns;
extern float absOffset[NBLOCK];
extern double ulc[2];
extern double lrc[2];
extern double sx;
extern double sy;
extern double xc;
extern double yc;

#define FUNC_NAMEm "misrinv"

int misrinv(
const int	   block,	   /* Input block */
const float	   line,	   /* Input line */
const float	   sample,	   /* Input sample */
double*		   x,		   /* Output SOM X coordinate */
double*		   y		   /* Output SOM Y coordinate */
)
{
  int		   n;		   /* Number of line to current block */
  char		   msg[STRLEN];	   /* Warning message */

/* Check Arguments */

  if (block < 1 || block > NBLOCK) {
    sprintf(msg, "block is out of range (0 < %d < %d)", block, nb);
    WRN_LOG_JUMP(msg);
  }

  if (line < -0.5 || line > nl_var - 0.5) {
    sprintf(msg, "line is out of range (0 < %e < %d)", line, nl_var);
    WRN_LOG_JUMP(msg);
  }

  if (sample < -0.5 || sample > ns - 0.5) {
    sprintf(msg, "sample is out of range (0 < %e < %d)", sample, ns);
    WRN_LOG_JUMP(msg);
  }

/* Compute SOM x/y coordinates in ulc/lrc units (meters) */

  n = (int)((block - 1) * nl_var * sx);
  *x = (double)(xc + n + (line * sx));
  *y = (double)(yc + ((sample + absOffset[block-1]) * sy));

  return(0);

 ERROR_HANDLE:

  *x = -1e-9;
  *y = -1e-9;
  return(1);
}

