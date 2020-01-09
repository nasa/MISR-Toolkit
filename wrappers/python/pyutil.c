/*===========================================================================
=                                                                           =
=                                 PyUtil                                    =
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
#include "pyMtk.h"

static PyObject* ParseFieldname(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *fieldname;             /* Field name */
   char *basefieldname;		/* Base field name */
   int ndim;                    /* Number dimensions */
   int *dimlist;		/* Dimension list */
   PyObject *py_dimlist;
   int i;

   if (!PyArg_ParseTuple(args,"s",&fieldname))
      return NULL;

   status = MtkParseFieldname(fieldname,&basefieldname,&ndim,&dimlist);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkParseFieldname Failed");
      return NULL;
   }

   py_dimlist = PyList_New(ndim);
   for (i = 0; i < ndim; ++i)
     PyList_SetItem(py_dimlist, i, PyInt_FromLong(dimlist[ndim]));

   result = Py_BuildValue("sN",basefieldname,py_dimlist);
   free(basefieldname);
   free(dimlist);
   return result;
}

static PyObject* JulianToCal(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double jd; /* Julian date */
   int year; /* Year */
   int month; /* Month */
   int day; /* Day */
   int hour; /* Hour */
   int min; /* Minutes */
   int sec; /* Seconds */

   if (!PyArg_ParseTuple(args,"d",&jd))
      return NULL;

   status = MtkJulianToCal(jd,&year,&month,&day,&hour,&min,&sec);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkJulianToCal Failed");
      return NULL;
   }

   result = Py_BuildValue("iiiiii",year,month,day,hour,min,sec);

   return result;
}

static PyObject* CalToJulian(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int y; /* Year */
   int m; /* Month */
   int d; /* Day */
   int h; /* Hour */
   int mn; /* Minutes */
   int s; /* Seconds */
   double julian; /* Julian date */

   if (!PyArg_ParseTuple(args,"iiiiii",&y,&m,&d,&h,&mn,&s))
      return NULL;

   status = MtkCalToJulian(y,m,d,h,mn,s,&julian);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkCalToJulian Failed");
      return NULL;
   }

   result = Py_BuildValue("d",julian);
   return result;
}

static PyObject* DateTimeToJulian(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *datetime; /* Date and time ISO 8601 format */
   double julian; /* Julian date */

   if (!PyArg_ParseTuple(args,"s",&datetime))
      return NULL;

   status = MtkDateTimeToJulian(datetime,&julian);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDateTimeToJulian Failed");
      return NULL;
   }

   result = Py_BuildValue("d",julian);
   return result;
}

static PyObject* JulianToDateTime(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double jd; /* Julian date */
   char datetime[MTKd_DATETIME_LEN]; /* Date and time ISO 8601 format */

   if (!PyArg_ParseTuple(args,"d",&jd))
      return NULL;

   status = MtkJulianToDateTime(jd,datetime);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkJulianToDateTime Failed");
      return NULL;
   }

   result = Py_BuildValue("s",datetime);
   return result;
}

static PyObject* Version(PyObject *self)
{
   PyObject *result;

   result = Py_BuildValue("s",MtkVersion());
   return result;
}

PyMethodDef util_methods[] = {
   {"parse_fieldname", (PyCFunction)ParseFieldname, METH_VARARGS,
    "Parses extra dimensions from fieldnames."},
   {"julian_to_cal", (PyCFunction)JulianToCal, METH_VARARGS,
    "Convert Julian date to calendar date."},
   {"cal_to_julian", (PyCFunction)CalToJulian, METH_VARARGS,
    "Convert calendar date to Julian date."},
   {"datetime_to_julian", (PyCFunction)DateTimeToJulian, METH_VARARGS,
    "Convert date and time (ISO 8601) to Julian date."},
   {"julian_to_datetime", (PyCFunction)JulianToDateTime, METH_VARARGS,
    "Convert Julian date to date and time (ISO 8601)."},
   {"version", (PyCFunction)Version, METH_NOARGS,
    "MISR Toolkit version."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};
