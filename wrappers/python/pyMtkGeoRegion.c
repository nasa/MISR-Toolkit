/*===========================================================================
=                                                                           =
=                               MtkGeoRegion                                 =
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

int MtkGeoCoord_init(MtkGeoCoord *self, PyObject *args, PyObject *kwds);
extern PyTypeObject MtkGeoCoordType;

static void
MtkGeoRegion_dealloc(MtkGeoRegion* self)
{
    Py_XDECREF(self->ulc);
    Py_XDECREF(self->ctr);
    Py_XDECREF(self->lrc);
    Py_TYPE(self)->tp_free((PyObject*)self); 
}

static PyObject *
MtkGeoRegion_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkGeoRegion *self;

   self = (MtkGeoRegion *)type->tp_alloc(type, 0);
   if (self != NULL)
   {  
      self->ulc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      MtkGeoCoord_init(self->ulc,NULL,NULL);
      self->urc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      MtkGeoCoord_init(self->urc,NULL,NULL);
      self->ctr = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      MtkGeoCoord_init(self->ctr,NULL,NULL);
      self->lrc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      MtkGeoCoord_init(self->lrc,NULL,NULL);
      self->llc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
      MtkGeoCoord_init(self->llc,NULL,NULL);

      if (self->ulc == NULL || self->urc == NULL || self->ctr == NULL ||
          self->lrc == NULL || self->llc == NULL)
      {
         PyErr_Format(PyExc_StandardError, "Problem initializing MtkGeoRegion.");
         return NULL;
      }
   }

   return (PyObject *)self;
}

int
MtkGeoRegion_init(MtkGeoRegion *self, PyObject *args, PyObject *kwds)
{
   self->ulc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   MtkGeoCoord_init(self->ulc,NULL,NULL);
   self->urc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   MtkGeoCoord_init(self->urc,NULL,NULL);
   self->ctr = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   MtkGeoCoord_init(self->ctr,NULL,NULL);
   self->lrc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   MtkGeoCoord_init(self->lrc,NULL,NULL);
   self->llc = (MtkGeoCoord*)PyObject_New(MtkGeoCoord, &MtkGeoCoordType);
   MtkGeoCoord_init(self->llc,NULL,NULL);

   if (self->ulc == NULL || self->urc == NULL || self->ctr == NULL ||
       self->lrc == NULL || self->llc == NULL)
   {
      PyErr_Format(PyExc_StandardError, "Problem initializing MtkGeoRegion.");
      return -1;
   }
  
   return 0;
}

static PyObject *
MtkGeoRegion_getulc(MtkGeoRegion *self, void *closure)
{
   Py_XINCREF(self->ulc);
   return (PyObject*)self->ulc;
}

MTK_READ_ONLY_ATTR(MtkGeoRegion, ulc)

static PyObject *
MtkGeoRegion_geturc(MtkGeoRegion *self, void *closure)
{
   Py_XINCREF(self->urc);
   return (PyObject*)self->urc;
}

MTK_READ_ONLY_ATTR(MtkGeoRegion, urc)

static PyObject *
MtkGeoRegion_getctr(MtkGeoRegion *self, void *closure)
{
   Py_XINCREF(self->ctr);
   return (PyObject*)self->ctr;
}

MTK_READ_ONLY_ATTR(MtkGeoRegion, ctr)

static PyObject *
MtkGeoRegion_getlrc(MtkGeoRegion *self, void *closure)
{
   Py_XINCREF(self->lrc);
   return (PyObject*)self->lrc;
}

MTK_READ_ONLY_ATTR(MtkGeoRegion, lrc)

static PyObject *
MtkGeoRegion_getllc(MtkGeoRegion *self, void *closure)
{
   Py_XINCREF(self->llc);
   return (PyObject*)self->llc;
}

MTK_READ_ONLY_ATTR(MtkGeoRegion, llc)

static PyGetSetDef MtkGeoRegion_getseters[] = {
    {"ulc", (getter)MtkGeoRegion_getulc, (setter)MtkGeoRegion_setulc,
     "Upper left coordinate.", NULL},
    {"urc", (getter)MtkGeoRegion_geturc, (setter)MtkGeoRegion_seturc,
     "Upper right coordinate.", NULL},
    {"ctr", (getter)MtkGeoRegion_getctr, (setter)MtkGeoRegion_setctr,
     "Center coordinate.", NULL},
    {"lrc", (getter)MtkGeoRegion_getlrc, (setter)MtkGeoRegion_setlrc,
     "Lower right coordinate.", NULL},
    {"llc", (getter)MtkGeoRegion_getllc, (setter)MtkGeoRegion_setllc,
     "Lower left coordinate.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef MtkGeoRegion_methods[] = {

    {NULL}  /* Sentinel */
};

PyTypeObject MtkGeoRegionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkGeoRegion", /*tp_name*/
    sizeof(MtkGeoRegion),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkGeoRegion_dealloc,/*tp_dealloc*/
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
    "MtkGeoRegion objects",    /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkGeoRegion_methods,    /* tp_methods */
    0,                          /* tp_members */
    MtkGeoRegion_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkGeoRegion_init,/* tp_init */
    0,                         /* tp_alloc */
    MtkGeoRegion_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkgeoregion_methods[] = {
    {NULL}  /* Sentinel */
};
