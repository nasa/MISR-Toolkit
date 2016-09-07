/*===========================================================================
=                                                                           =
=                       MtkTimeRangeToOrbitList_test                        =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrError.h"
#include <stdio.h>
#include <stdlib.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  int cn = 0;			/* Column number */
  int orbit_count;
  int *orbit_list;
  int i;
  const int ol_expected[] = {11312, 11313, 11314, 11315}; /* Orbit List Expected */
  const int ol_expected2[] = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007};
  const int ol_expected3[] = {38105, 38106, 38107, 38108, 38109, 38110, 38111, 38112, 38113, 38114, 38115, 38116, 38117, 38118, 38119};

  MTK_PRINT_STATUS(cn,"Testing MtkTimeRangeToOrbitList");

  /* Normal Call */
  status = MtkTimeRangeToOrbitList("2002-02-02T04:00:00Z","2002-02-02T09:00:00Z",
                                   &orbit_count,&orbit_list);
  if (status == MTK_SUCCESS)
  {
    if (orbit_count != sizeof ol_expected / sizeof *ol_expected)
      data_ok = MTK_FALSE;

    for (i = 0; i < sizeof ol_expected / sizeof *ol_expected; ++i)
      if (orbit_list[i] != ol_expected[i])
      {
        data_ok = MTK_FALSE;
        break;
      }

    free(orbit_list);
  }
  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeRangeToOrbitList("2000-01-01T00:00:00Z","2000-02-25T12:00:00Z",
                                   &orbit_count,&orbit_list);

  if (status == MTK_SUCCESS)
  {
    if (orbit_count != sizeof ol_expected2 / sizeof *ol_expected2)
      data_ok = MTK_FALSE;

    for (i = 0; i < sizeof ol_expected2 / sizeof *ol_expected2; ++i)
      if (orbit_list[i] != ol_expected2[i])
      {
        data_ok = MTK_FALSE;
        break;
      }

    free(orbit_list);
  }
  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal Call */
  status = MtkTimeRangeToOrbitList("2007-02-16T00:00:00Z","2007-02-17T00:00:00Z",
                                   &orbit_count,&orbit_list);

  if (status == MTK_SUCCESS)
  {
    if (orbit_count != sizeof ol_expected3 / sizeof *ol_expected3)
      data_ok = MTK_FALSE;

    for (i = 0; i < sizeof ol_expected3 / sizeof *ol_expected3; ++i)
      if (orbit_list[i] != ol_expected3[i])
      {
        data_ok = MTK_FALSE;
        break;
      }

    free(orbit_list);
  }
  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkTimeRangeToOrbitList(NULL,"2002-02-02T09:00:00Z",
                                   &orbit_count,&orbit_list);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeRangeToOrbitList("2002-02-02T04:00:00Z",NULL,
                                   &orbit_count,&orbit_list);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeRangeToOrbitList("2002-02-02T04:00:00Z","2002-02-02T09:00:00Z",
                                   NULL,&orbit_list);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeRangeToOrbitList("2002-02-02T04:00:00Z","2002-02-02T09:00:00Z",
                                   &orbit_count,NULL);
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkTimeRangeToOrbitList("2002-02-02T09:00:00Z","2002-02-02T04:00:00Z",
                                   &orbit_count,&orbit_list);
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
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
