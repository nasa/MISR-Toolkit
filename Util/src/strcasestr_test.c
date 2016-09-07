/*===========================================================================
=                                                                           =
=                              strcasestr_test                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include <string.h>

int main()
{
  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int cn = 0;			/* Column number */
  const char *largestring = "Foo Bar Baz";
  char smallstring[80] = "Bar";
  char *ptr;

  MTK_PRINT_STATUS(cn,"Testing strcasestr");

  /* string found with same case */
  ptr = strcasestr(largestring, smallstring);
  if (strcmp(ptr,"Bar Baz") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* string found with different case */
  strcpy(smallstring,"bar");
  ptr = strcasestr(largestring, smallstring);
  if (strcmp(ptr,"Bar Baz") == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* small is empty string */
  strcpy(smallstring,"");
  ptr = strcasestr(largestring, smallstring);
  if (strcmp(ptr,largestring) == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

   if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
