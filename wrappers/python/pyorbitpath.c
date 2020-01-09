/*===========================================================================
=                                                                           =
=                               PyOrbitPath                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <Python.h>
#include "MisrToolkit.h"
#include <stdlib.h>
#include "pyMtk.h"

PyObject* LatLonToPathList(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double lat_dd; /* Latitude */
   double lon_dd; /* Longitude */
   int pathcnt;   /* Path Count */
   int *pathlist; /* Path List */
   int i;

   if (!PyArg_ParseTuple(args,"dd",&lat_dd,&lon_dd))
      return NULL;

   status = MtkLatLonToPathList(lat_dd, lon_dd, &pathcnt, &pathlist);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkLatLonToPathList Failed");
      return NULL;
   }

   result = PyList_New(pathcnt);
   for (i = 0; i < pathcnt; ++i)
     PyList_SetItem(result, i, PyInt_FromLong(pathlist[i]));

   free(pathlist);
   return result;
}

PyObject* OrbitToPath(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int orbit; /* Orbit Number */
   int path;  /* Path Number */

   if (!PyArg_ParseTuple(args,"i",&orbit))
      return NULL;

   status = MtkOrbitToPath(orbit,&path);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkOrbitToPath Failed");
      return NULL;
   }

   result = Py_BuildValue("i",path);
   return result;
}

PyObject* PathTimeRangeToOrbitList(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;         /* Path */
   char *start_time; /* Start Time */
   char *end_time;   /* End Time */
   int orbitcnt;    /* Orbit Count */
   int *orbitlist;  /* Orbit List */
   int i;

   if (!PyArg_ParseTuple(args,"iss",&path,&start_time,&end_time))
      return NULL;

   status = MtkPathTimeRangeToOrbitList(path,start_time,end_time,
					&orbitcnt,&orbitlist);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkPathTimeRangeToOrbitList Failed");
      return NULL;
   }

   result = PyList_New(orbitcnt);
   for (i = 0; i < orbitcnt; ++i)
     PyList_SetItem(result, i, PyInt_FromLong(orbitlist[i]));

   free(orbitlist);
   return result;
}

PyObject* TimeRangeToOrbitList(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *start_time; /* Start Time */
   char *end_time; /* End Time */
   int orbitcnt; /* Orbit Count */
   int *orbitlist; /* Orbit List */
   int i;

   if (!PyArg_ParseTuple(args,"ss",&start_time,&end_time))
      return NULL;

   status = MtkTimeRangeToOrbitList(start_time,end_time,&orbitcnt,&orbitlist);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkTimeRangeToOrbitList Failed");
      return NULL;
   }

   result = PyList_New(orbitcnt);
   for (i = 0; i < orbitcnt; ++i)
     PyList_SetItem(result, i, PyInt_FromLong(orbitlist[i]));

   free(orbitlist);
   return result;
}

PyObject* TimeToOrbitPath(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *datetime;  /* YYYY-MM-DDThh:mm:ssZ */
   int orbit; /* Orbit Number */
   int path; /* Path */

   if (!PyArg_ParseTuple(args,"s",&datetime))
      return NULL;

   status = MtkTimeToOrbitPath(datetime,&orbit,&path);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkTimeToOrbitPath Failed");
      return NULL;
   }

   result = Py_BuildValue("ii",orbit,path);
   return result;
}

PyObject* OrbitToTimeRange(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char start_time[MTKd_DATETIME_LEN];  /* YYYY-MM-DDThh:mm:ssZ */
   char end_time[MTKd_DATETIME_LEN];
   int orbit; /* Orbit Number */

   if (!PyArg_ParseTuple(args,"i",&orbit))
      return NULL;

   status = MtkOrbitToTimeRange(orbit,start_time,end_time);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkOrbitToTimeRange Failed");
      return NULL;
   }

   result = Py_BuildValue("ss",start_time,end_time);
   return result;
}


PyMethodDef orbitpath_methods[] = {
   {"latlon_to_path_list", (PyCFunction)LatLonToPathList, METH_VARARGS,
    "Get list of paths that cover a particular latitude and longitude."},
   {"orbit_to_path", (PyCFunction)OrbitToPath, METH_VARARGS,
    "Given orbit number return path number."},
   {"path_time_range_to_orbit_list", (PyCFunction)PathTimeRangeToOrbitList, METH_VARARGS,
    "Given path and time range return list of orbits on path."},
   {"time_range_to_orbit_list", (PyCFunction)TimeRangeToOrbitList, METH_VARARGS,
    "Given start time and end time return list of orbits."},
   {"time_to_orbit_path", (PyCFunction)TimeToOrbitPath, METH_VARARGS,
    "Given time return orbit number and path number."},
   {"orbit_to_time_range", (PyCFunction)OrbitToTimeRange, METH_VARARGS,
    "Given orbit number return time range."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};
