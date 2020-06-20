/*===========================================================================
=                                                                           =
=                               strcasestr                                  =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

============================================================================*/

#include "MisrUtil.h"
#include <string.h>
#include <ctype.h>

/* Since Windows does not include strcasestr, we carry our own implementation */
char* win_strcasestr(const char *s, const char *find)
{
  char c, sc;
  size_t len;

  if ((c = *find++) != 0) {
    c = tolower((unsigned char)c);
    len = strlen(find);
    do {
      do {
	if ((sc = *s++) == 0)
	  return NULL;
      } while ((char)tolower((unsigned)sc) != c);
    } while (strncasecmp(s, find, len) != 0);
    s--;
  }
  return ((char *)s);
}
