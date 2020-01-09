/*===========================================================================
=                                                                           =
=                               MtkSomRegion                                 =
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

int MtkSomCoord_init(MtkSomCoord *self, PyObject *args, PyObject *kwds);
extern PyTypeObject MtkSomCoordType;

static void
MtkSomRegion_dealloc(MtkSomRegion* self)
{
    Py_XDECREF(self->ulc);
    Py_XDECREF(self->ctr);
    Py_XDECREF(self->lrc);
    Py_TYPE(self)->tp_free((PyObject*)self); 
}

static PyObject *
MtkSomRegion_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkSomRegion *self;

   self = (MtkSomRegion *)type->tp_alloc(type, 0);
   if (self != NULL)
   {  
      self->path = 0;
      self->ulc = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
      MtkSomCoord_init(self->ulc,NULL,NULL);
      self->ctr = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
      MtkSomCoord_init(self->ctr,NULL,NULL);
      self->lrc = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
      MtkSomCoord_init(self->lrc,NULL,NULL);

      if (self->ulc == NULL || self->ctr == NULL || self->lrc == NULL)
      {
         PyErr_Format(PyExc_StandardError, "Problem initializing MtkSomRegion.");
         return NULL;
      }
   }

   return (PyObject *)self;
}

int
MtkSomRegion_init(MtkSomRegion *self, PyObject *args, PyObject *kwds)
{
   self->path = 0;
   self->ulc = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
   MtkSomCoord_init(self->ulc,NULL,NULL);
   self->ctr = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
   MtkSomCoord_init(self->ctr,NULL,NULL);
   self->lrc = (MtkSomCoord*)PyObject_New(MtkSomCoord, &MtkSomCoordType);
   MtkSomCoord_init(self->lrc,NULL,NULL);

   if (self->ulc == NULL || self->ctr == NULL || self->lrc == NULL)
   {
      PyErr_Format(PyExc_StandardError, "Problem initializing MtkSomRegion.");
      return -1;
   }
  
   return 0;
}

static PyObject *
MtkSomRegion_getulc(MtkSomRegion *self, void *closure)
{
   Py_XINCREF(self->ulc);
   return (PyObject*)self->ulc;
}

MTK_READ_ONLY_ATTR(MtkSomRegion, ulc)

static PyObject *
MtkSomRegion_getctr(MtkSomRegion *self, void *closure)
{
   Py_XINCREF(self->ctr);
   return (PyObject*)self->ctr;
}

MTK_READ_ONLY_ATTR(MtkSomRegion, ctr)

static PyObject *
MtkSomRegion_getlrc(MtkSomRegion *self, void *closure)
{
   Py_XINCREF(self->lrc);
   return (PyObject*)self->lrc;
}

MTK_READ_ONLY_ATTR(MtkSomRegion, lrc)

static PyGetSetDef MtkSomRegion_getseters[] = {
    {"ulc", (getter)MtkSomRegion_getulc, (setter)MtkSomRegion_setulc,
     "Upper left coordinate.", NULL},
    {"ctr", (getter)MtkSomRegion_getctr, (setter)MtkSomRegion_setctr,
     "Center coordinate.", NULL},
    {"lrc", (getter)MtkSomRegion_getlrc, (setter)MtkSomRegion_setlrc,
     "Lower right coordinate.", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkSomRegion_members[] = {
    {"path", T_INT, offsetof(MtkSomRegion, path), READONLY,
     "Path number"},
    {NULL}  /* Sentinel */
};

static PyMethodDef MtkSomRegion_methods[] = {

    {NULL}  /* Sentinel */
};

PyTypeObject MtkSomRegionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkSomRegion", /*tp_name*/
    sizeof(MtkSomRegion),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkSomRegion_dealloc,/*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
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
    "MtkSomRegion objects",    /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkSomRegion_methods,    /* tp_methods */
    MtkSomRegion_members,      /* tp_members */
    MtkSomRegion_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkSomRegion_init,/* tp_init */
    0,                         /* tp_alloc */
    MtkSomRegion_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtksomregion_methods[] = {
    {NULL}  /* Sentinel */
};
