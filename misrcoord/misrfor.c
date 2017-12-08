#include "misrproj.h"		   /* Prototype for this function */
#include "errormacros.h"	   /* Error macros */
#include <math.h>		   /* Prototype for floor */

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

#define FUNC_NAMEm "misrfor"

int misrfor(
const double	   x,		   /* Output SOM X coordinate */
const double	   y,		   /* Output SOM Y coordinate */
int*		   block,	   /* Input block */
float*		   line,	   /* Input line */
float*		   sample	   /* Input sample */
)
{
  float		   i;		   /* Intermediate X coordinate */
  float		   j;		   /* Intermediate Y coordinate */
  int		   b;		   /* Intermediate block */
  float		   l;		   /* Intermediate line */
  float		   s;		   /* Intermediate sample */
  char		   msg[STRLEN];	   /* Warning message */

/* Compute intermediate coordinates */

  i = (float)((x - xc) / sx);
  j = (float)((y - yc) / sy);

/* Check for very small numbers in i and j and assume they are zero */

  i = (fabs(i) < 1E-5 ? 0.0 : i);
  j = (fabs(j) < 1E-5 ? 0.0 : j);

/* Compute block and check range */

  b = (int)(floor((i + 0.5) / nl_var)) + 1;
  if (b < 1 || b > nb) {
    sprintf(msg, "block is out of range (1 < %d < %d)", b, nb);
    WRN_LOG_JUMP(msg);
  }

/* Compute line and check range */

  l = (float)(i - ((b - 1) * nl_var));
  if (l < -0.5 || l > nl_var - 0.5) {
    sprintf(msg, "line is out of range (0 < %e < %d)", l, nl_var);
    WRN_LOG_JUMP(msg);
  }

/* Compute sample and check range */

  s = (float)(j - absOffset[b-1]);
  if (s < -0.5 || s > ns - 0.5) {
    sprintf(msg, "sample is out of range (0 < %e < %d)", s, ns);
    WRN_LOG_JUMP(msg);
  }

/* Set return values */

  *block = b;
  *line = l;
  *sample = s;

  return(0);

 ERROR_HANDLE:

  *block = -1;
  *line = -1.0;
  *sample = -1.0;
  return(1);
}
