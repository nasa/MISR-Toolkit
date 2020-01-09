/*===========================================================================
=                                                                           =
=                               PyUnitConv                                  =
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

PyObject* DdToDegMinSec(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dd;  /* Decimal Degrees */
   int deg;    /* Degrees */
   int min;    /* Minutes */
   double sec; /* Seconds */

   if (!PyArg_ParseTuple(args,"d",&dd))
      return NULL;

   status = MtkDdToDegMinSec(dd,&deg,&min,&sec);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDdToDegMinSec Failed");
      return NULL;
   }

   result = Py_BuildValue("iid",deg,min,sec);
   return result;
}

PyObject* DmsToDd(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dms; /* Packed degrees, minutes, seconds */
   double dd;  /* Decimal degrees */

   if (!PyArg_ParseTuple(args,"d",&dms))
      return NULL;

   status = MtkDmsToDd(dms,&dd);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDmsToDd Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dd);
   return result;
}

PyObject* DdToDms(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dd;  /* Decimal degrees */
   double dms; /* Packed degrees, minutes, seconds */

   if (!PyArg_ParseTuple(args,"d",&dd))
      return NULL;

   status = MtkDdToDms(dd,&dms);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDdToDms Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dms);
   return result;
}

PyObject* DmsToDegMinSec(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dms; /* Packed degrees, minutes, seconds */
   int deg;    /* Degrees */
   int min;    /* Minutes */
   double sec; /* Seconds */

   if (!PyArg_ParseTuple(args,"d",&dms))
      return NULL;

   status = MtkDmsToDegMinSec(dms,&deg,&min,&sec);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDmsToDegMinSec Failed");
      return NULL;
   }

   result = Py_BuildValue("iid",deg,min,sec);
   return result;
}

PyObject* DdToRad(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dd;  /* Decimal Degrees */
   double rad; /* Radians */

   if (!PyArg_ParseTuple(args,"d",&dd))
      return NULL;

   status = MtkDdToRad(dd,&rad);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDdToRad Failed");
      return NULL;
   }

   result = Py_BuildValue("d",rad);
   return result;
}

PyObject* DmsToRad(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double dms; /* Packed degrees, minutes, seconds */
   double rad; /* Radians */

   if (!PyArg_ParseTuple(args,"d",&dms))
      return NULL;

   status = MtkDmsToRad(dms,&rad);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDmsToRad Failed");
      return NULL;
   }

   result = Py_BuildValue("d",rad);
   return result;
}

PyObject* DegMinSecToDd(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int deg;    /* Degrees */
   int min;    /* Minutes */
   double sec; /* Seconds */
   double dd;  /* Decimal degrees */

   if (!PyArg_ParseTuple(args,"iid",&deg,&min,&sec))
      return NULL;

   status = MtkDegMinSecToDd(deg,min,sec,&dd);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDegMinSecToDd Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dd);
   return result;
}

PyObject* RadToDd(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double rad; /* Radians */
   double dd;  /* Decimal degrees */ 

   if (!PyArg_ParseTuple(args,"d",&rad))
      return NULL;

   status = MtkRadToDd(rad,&dd);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkRadToDd Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dd);
   return result;
}

PyObject* DegMinSecToDms(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int deg;    /* Degrees */
   int min;    /* Minutes */
   double sec; /* Seconds */
   double dms; /* Packed degrees, minutes, seconds */

   if (!PyArg_ParseTuple(args,"iid",&deg,&min,&sec))
      return NULL;

   status = MtkDegMinSecToDms(deg,min,sec,&dms);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDegMinSecToDms Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dms);
   return result;
}

PyObject* RadToDegMinSec(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double rad; /* Radians */
   int deg;    /* Degrees */
   int min;    /* Minutes */
   double sec; /* Seconds */

   if (!PyArg_ParseTuple(args,"d",&rad))
      return NULL;

   status = MtkRadToDegMinSec(rad,&deg,&min,&sec);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkRadToDegMinSec Failed");
      return NULL;
   }

   result = Py_BuildValue("iid",deg,min,sec);
   return result;
}

PyObject* DegMinSecToRad(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int deg;     /* Degrees */
   int min;     /* Minutes */
   double sec;  /* Seconds */
   double rad; /* Radians */

   if (!PyArg_ParseTuple(args,"iid",&deg,&min,&sec))
      return NULL;

   status = MtkDegMinSecToRad(deg,min,sec,&rad);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkDegMinSecToRad Failed");
      return NULL;
   }

   result = Py_BuildValue("d",rad);
   return result;
}

PyObject* RadToDms(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double rad; /* Radians */
   double dms; /* Degrees, minutes, seconds */

   if (!PyArg_ParseTuple(args,"d",&rad))
      return NULL;

   status = MtkRadToDms(rad,&dms);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkRadToDms Failed");
      return NULL;
   }

   result = Py_BuildValue("d",dms);
   return result;
}

PyMethodDef unitconv_methods[] = {
   {"dd_to_deg_min_sec", (PyCFunction)DdToDegMinSec, METH_VARARGS,
    "Convert decimal degrees to unpacked degrees, minutes, seconds."},
   {"dms_to_dd", (PyCFunction)DmsToDd, METH_VARARGS,
    "Convert packed degrees, minutes, seconds to decimal degrees."},
   {"dd_to_dms", (PyCFunction)DdToDms, METH_VARARGS,
    "Convert decimal degrees to packed degrees, minutes, seconds."},
   {"dms_to_deg_min_sec", (PyCFunction)DmsToDegMinSec, METH_VARARGS,
    "Convert packed degrees, minutes, seconds to unpacked."},
   {"dd_to_rad", (PyCFunction)DdToRad, METH_VARARGS,
    "Convert decimal degrees to radians."},
   {"dms_to_rad", (PyCFunction)DmsToRad, METH_VARARGS,
    "Convert packed degrees, minutes, seconds to Radians."},
   {"deg_min_sec_to_dd", (PyCFunction)DegMinSecToDd, METH_VARARGS,
    "Convert unpacked degrees, minutes, seconds to decimal degrees."},
   {"rad_to_dd", (PyCFunction)RadToDd, METH_VARARGS,
    "Convert radians to decimal degrees."},
   {"deg_min_sec_to_dms", (PyCFunction)DegMinSecToDms, METH_VARARGS,
    "Convert unpacked Degrees, minutes, seconds to packed."},
   {"rad_to_deg_min_sec", (PyCFunction)RadToDegMinSec, METH_VARARGS,
    "Convert radians to unpacked degrees, minutes, seconds."},
   {"deg_min_sec_to_rad",(PyCFunction) DegMinSecToRad, METH_VARARGS,
    "Convert unpacked degrees, minutes, seconds to radians."},
   {"rad_to_dms", (PyCFunction)RadToDms, METH_VARARGS,
    "Convert radians to packed degrees, minutes, seconds."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};
