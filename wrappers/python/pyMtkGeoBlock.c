/*===========================================================================
=                                                                           =
=                               MtkGeoBlock                                 =
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

extern PyTypeObject MtkGeoCoordType;

static void
MtkGeoBlock_dealloc(MtkGeoBlock* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
    Py_XDECREF(self->ulc);
    Py_XDECREF(self->ctr);
    Py_XDECREF(self->lrc);
}

static PyObject *
MtkGeoBlock_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkGeoBlock *self;
   MTKt_GeoCoord gc = MTKT_GEOCOORD_INIT;

   self = (MtkGeoBlock *)type->tp_alloc(type, 0);
   if (self != NULL)
   {  
      self->block_number = 0;
      self->ulc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      self->ulc->gc = gc;
      self->urc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      self->urc->gc = gc;      
      self->ctr = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      self->ctr->gc = gc;
      self->lrc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      self->lrc->gc = gc;
      self->llc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      self->llc->gc = gc;

      if (self->ulc == NULL || self->urc == NULL || self->ctr == NULL ||
          self->lrc == NULL || self->llc == NULL)
      {
         PyErr_Format(PyExc_StandardError, "Problem initializing MtkGeoBlock.");
         return NULL;
      }
   }

   return (PyObject *)self;
}

int
MtkGeoBlock_init(MtkGeoBlock *self, PyObject *args, PyObject *kwds)
{
   MTKt_GeoCoord gc = MTKT_GEOCOORD_INIT;
   self->block_number = 0;
   self->ulc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   self->ulc->gc = gc;
   self->urc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   self->urc->gc = gc;   
   self->ctr = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   self->ctr->gc = gc;
   self->lrc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   self->lrc->gc = gc;
   self->llc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   self->llc->gc = gc;
      
   if (self->ulc == NULL || self->urc == NULL || self->ctr == NULL ||
       self->lrc == NULL || self->llc == NULL)
   {
      PyErr_Format(PyExc_StandardError, "Problem initializing MtkGeoBlock.");
      return -1;
   }
  
   return 0;
}

static PyObject *
MtkGeoBlock_getulc(MtkGeoBlock *self, void *closure)
{
   Py_XINCREF(self->ulc);
   return (PyObject*)self->ulc;
}

MTK_READ_ONLY_ATTR(MtkGeoBlock, ulc)

static PyObject *
MtkGeoBlock_geturc(MtkGeoBlock *self, void *closure)
{
   Py_XINCREF(self->urc);
   return (PyObject*)self->urc;
}

MTK_READ_ONLY_ATTR(MtkGeoBlock, urc)

static PyObject *
MtkGeoBlock_getctr(MtkGeoBlock *self, void *closure)
{
   Py_XINCREF(self->ctr);
   return (PyObject*)self->ctr;
}

MTK_READ_ONLY_ATTR(MtkGeoBlock, ctr)

static PyObject *
MtkGeoBlock_getlrc(MtkGeoBlock *self, void *closure)
{
   Py_XINCREF(self->lrc);
   return (PyObject*)self->lrc;
}

MTK_READ_ONLY_ATTR(MtkGeoBlock, lrc)

static PyObject *
MtkGeoBlock_getllc(MtkGeoBlock *self, void *closure)
{
   Py_XINCREF(self->llc);
   return (PyObject*)self->llc;
}

MTK_READ_ONLY_ATTR(MtkGeoBlock, llc)

static PyGetSetDef MtkGeoBlock_getseters[] = {
    {"ulc", (getter)MtkGeoBlock_getulc, (setter)MtkGeoBlock_setulc,
     "Upper left coordinate.", NULL},
    {"urc", (getter)MtkGeoBlock_geturc, (setter)MtkGeoBlock_seturc,
     "Upper right coordinate.", NULL},
    {"ctr", (getter)MtkGeoBlock_getctr, (setter)MtkGeoBlock_setctr,
     "Center coordinate.", NULL},
    {"lrc", (getter)MtkGeoBlock_getlrc, (setter)MtkGeoBlock_setlrc,
     "Lower right coordinate.", NULL},
    {"llc", (getter)MtkGeoBlock_getllc, (setter)MtkGeoBlock_setllc,
     "Lower left coordinate.", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkGeoBlock_members[] = {
    {"block", T_INT, offsetof(MtkGeoBlock, block_number), READONLY,
     "Block number"},
    {NULL}  /* Sentinel */
};


static PyMethodDef MtkGeoBlock_methods[] = {

    {NULL}  /* Sentinel */
};


PyTypeObject MtkGeoBlockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkGeoBlock", /*tp_name*/
    sizeof(MtkGeoBlock),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkGeoBlock_dealloc,/*tp_dealloc*/
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
    "MtkGeoBlock objects",    /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkGeoBlock_methods,    /* tp_methods */
    MtkGeoBlock_members,      /* tp_members */
    MtkGeoBlock_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkGeoBlock_init,/* tp_init */
    0,                         /* tp_alloc */
    MtkGeoBlock_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkgeoblock_methods[] = {
    {NULL}  /* Sentinel */
};
