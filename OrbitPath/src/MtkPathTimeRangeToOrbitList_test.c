/*===========================================================================
=                                                                           =
=                     MtkPathTimeRangeToOrbitList_test                      =
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
  int cn = 0;
  int num_orbits;
  int *orbit_list;
  int olist_expected[] = {11338, 11571, 11804, 12037, 12270, 12503};
  int olist_expected2[] = {1319, 1552, 1785};
  int i;

  MTK_PRINT_STATUS(cn,"Testing MtkPathTimeRangeToOrbitList");

  /* Normal Call */
  status = MtkPathTimeRangeToOrbitList( 78, "2002-02-02T02:00:00Z",
					"2002-05-02T02:00:00Z", &num_orbits,
				        &orbit_list );
  if (status == MTK_SUCCESS)
  {
    if (num_orbits != sizeof olist_expected / sizeof *olist_expected)
      data_ok = MTK_FALSE;

    for (i = 0; i < sizeof olist_expected / sizeof *olist_expected; ++i)
      if (orbit_list[i] != olist_expected[i])
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

  /* Start time before start of mission */
  data_ok = MTK_TRUE;
  status = MtkPathTimeRangeToOrbitList( 78, "1999-02-02T02:00:00Z",
					"2000-05-02T02:00:00Z", &num_orbits,
				        &orbit_list );
  if (status == MTK_SUCCESS)
  {
    if (num_orbits != sizeof olist_expected2 / sizeof *olist_expected2)
      data_ok = MTK_FALSE;

    for (i = 0; i < sizeof olist_expected2 / sizeof *olist_expected2; ++i)
      if (orbit_list[i] != olist_expected2[i])
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
  status = MtkPathTimeRangeToOrbitList( 0, "2002-02-02T02:00:00Z",
					"2002-05-02T02:00:00Z", &num_orbits,
				        &orbit_list );
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 234, "2002-02-02T02:00:00Z",
					"2002-05-02T02:00:00", &num_orbits,
				        &orbit_list );
  if (status == MTK_BAD_ARGUMENT)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 78, NULL,
					"2002-05-02T02:00:00Z", &num_orbits,
				        &orbit_list );
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 78, "2002-02-02T02:00:00Z",
					NULL, &num_orbits,
				        &orbit_list );
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 78, "2002-02-02T02:00:00Z",
					"20020502020000", NULL,
				        &orbit_list );
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 78, "2002-02-02T02:00:00Z",
					"2002-05-02T02:00:00Z", &num_orbits,
				        NULL );
  if (status == MTK_NULLPTR)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkPathTimeRangeToOrbitList( 78, "2002-05-02T02:00:00Z",
                                        "2002-02-02T02:00:00Z", &num_orbits,
				        &orbit_list );
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
