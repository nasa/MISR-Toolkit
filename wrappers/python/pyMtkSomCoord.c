/*===========================================================================
=                                                                           =
=                               MtkSomCoord                                 =
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
MtkSomCoord_dealloc(MtkSomCoord* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkSomCoord_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkSomCoord *self;
   MTKt_SomCoord som_coord = MTKT_SOMCOORD_INIT;

   self = (MtkSomCoord *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->som_coord = som_coord;
   }

   return (PyObject *)self;
}

int
MtkSomCoord_init(MtkSomCoord *self, PyObject *args, PyObject *kwds)
{
   MTKt_SomCoord som_coord = MTKT_SOMCOORD_INIT;
   self->som_coord = som_coord;

   return 0;
}

static PyObject *
MtkSomCoord_repr(MtkSomCoord *self)
{
   char temp[100];
   sprintf(temp,"(%f,%f)",self->som_coord.x,self->som_coord.y);
   return PyString_FromString(temp);
}

static PyMemberDef MtkSomCoord_members[] = {
    {"x", T_DOUBLE, offsetof(MtkSomCoord, som_coord.x), READONLY,
     "Som X."},
    {"y", T_DOUBLE, offsetof(MtkSomCoord, som_coord.y), READONLY,
     "Som Y."},
    {NULL}  /* Sentinel */
};

PyTypeObject MtkSomCoordType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkSomCoord", /*tp_name*/
    sizeof(MtkSomCoord),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkSomCoord_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)MtkSomCoord_repr,/*tp_repr*/
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
    "MtkSomCoord objects",     /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    MtkSomCoord_members,       /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkSomCoord_init, /* tp_init */
    0,                         /* tp_alloc */
    MtkSomCoord_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtksomcoord_methods[] = {
    {NULL}  /* Sentinel */
};
