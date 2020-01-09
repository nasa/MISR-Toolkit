/*===========================================================================
=                                                                           =
=                               MtkGeoCoord                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <Python.h>
#include "structmember.h"
#include "pyMtk.h"

static void
MtkGeoCoord_dealloc(MtkGeoCoord* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkGeoCoord_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkGeoCoord *self;
   MTKt_GeoCoord gc = MTKT_GEOCOORD_INIT;

   self = (MtkGeoCoord *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->gc = gc;
   }

   return (PyObject *)self;
}

int
MtkGeoCoord_init(MtkGeoCoord *self, PyObject *args, PyObject *kwds)
{
   MTKt_GeoCoord gc = MTKT_GEOCOORD_INIT;
   self->gc = gc;

   return 0;
}

static PyObject *
MtkGeoCoord_repr(MtkGeoCoord *self)
{
   char temp[100];
   sprintf(temp,"(%f,%f)",self->gc.lat,self->gc.lon);
   return PyString_FromString(temp);
}

static PyMemberDef MtkGeoCoord_members[] = {
    {"lat", T_DOUBLE, offsetof(MtkGeoCoord, gc.lat), READONLY,
     "Latitude in decimal degrees."},
    {"lon", T_DOUBLE, offsetof(MtkGeoCoord, gc.lon), READONLY,
     "Longitude in decimal degrees."},
    {NULL}  /* Sentinel */
};

PyTypeObject MtkGeoCoordType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkGeoCoord", /*tp_name*/
    sizeof(MtkGeoCoord),       /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkGeoCoord_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)MtkGeoCoord_repr,/*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "MtkGeoCoord objects",     /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    MtkGeoCoord_members,       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkGeoCoord_init, /* tp_init */
    0,                         /* tp_alloc */
    MtkGeoCoord_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkgeocoord_methods[] = {
    {NULL}  /* Sentinel */
};
