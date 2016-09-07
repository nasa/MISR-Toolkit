/*===========================================================================
=                                                                           =
=                             strncasecmp_test                              =
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
  const char *s1 = "kilometers";
  char s2[80];
  int result;

  MTK_PRINT_STATUS(cn,"Testing strncasecmp");

  /* string compare with same case */
  strcpy(s2, "kilometers");
  result = strncasecmp(s1, s2, strlen(s1));
  if (result == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* string compare with different case */
  strcpy(s2,"KILOMETERS");
  result = strncasecmp(s1, s2, strlen(s1));
  if (result == 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* string compare with s2 larger */
  strcpy(s2,"KILOMETERSABC");
  result = strncasecmp(s1, s2, strlen(s2));
  if (result < 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* string compare with s2 smaller */
  strcpy(s2,"KILO");
  result = strncasecmp(s1, s2, strlen(s1));
  if (result > 0) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* s2 is empty string */
  strcpy(s2,"");
  result = strncasecmp(s1, s2, strlen(s1));
  if (result > 0) {
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
