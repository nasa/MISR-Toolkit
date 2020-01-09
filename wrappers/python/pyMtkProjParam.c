/*===========================================================================
=                                                                           =
=                              MtkProjParam                                 =
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
#include "structmember.h"
#include "pyMtk.h"

static void
MtkProjParam_dealloc(MtkProjParam* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkProjParam_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkProjParam *self;
   MTKt_MisrProjParam proj = MTKT_MISRPROJPARAM_INIT;

   self = (MtkProjParam *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->pp = proj;
   }

   return (PyObject *)self;
}

int
MtkProjParam_init(MtkProjParam *self, PyObject *args, PyObject *kwds)
{
   MTKt_MisrProjParam proj = MTKT_MISRPROJPARAM_INIT;
   self->pp = proj;

   return 0;
}

static PyObject *
MtkProjParam_getprojparam(MtkProjParam *self, void *closure)
{
   PyObject *result;
   int i;

   result = PyTuple_New(15);
   if (result == NULL)
      return NULL;

   for (i = 0; i < 15; ++i)
      PyTuple_SetItem(result, i, Py_BuildValue("d",self->pp.projparam[i]));

   return result;
}

MTK_READ_ONLY_ATTR(MtkProjParam, projparam)

static PyObject *
MtkProjParam_getulc(MtkProjParam *self, void *closure)
{
   PyObject *result;
   int i;

   result = PyTuple_New(2);
   if (result == NULL)
      return NULL;

   for (i = 0; i < 2; ++i)
      PyTuple_SetItem(result, i, Py_BuildValue("d",self->pp.ulc[i]));

   return result;
}

MTK_READ_ONLY_ATTR(MtkProjParam, ulc)

static PyObject *
MtkProjParam_getlrc(MtkProjParam *self, void *closure)
{
   PyObject *result;
   int i;

   result = PyTuple_New(2);
   if (result == NULL)
      return NULL;

   for (i = 0; i < 2; ++i)
      PyTuple_SetItem(result, i, Py_BuildValue("d",self->pp.lrc[i]));

   return result;
}

MTK_READ_ONLY_ATTR(MtkProjParam, lrc)

static PyObject *
MtkProjParam_getreloffset(MtkProjParam *self, void *closure)
{
   PyObject *result;
   int i;

   result = PyTuple_New(179);
   if (result == NULL)
      return NULL;

   for (i = 0; i < 179; ++i)
      PyTuple_SetItem(result, i, Py_BuildValue("f",self->pp.reloffset[i]));

   return result;
}

MTK_READ_ONLY_ATTR(MtkProjParam, reloffset)


static PyGetSetDef MtkProjParam_getseters[] = {
    {"projparam", (getter)MtkProjParam_getprojparam, (setter)MtkProjParam_setprojparam,
     "projparam", NULL},
    {"ulc", (getter)MtkProjParam_getulc, (setter)MtkProjParam_setulc,
     "ulc", NULL},
    {"lrc", (getter)MtkProjParam_getlrc, (setter)MtkProjParam_setlrc,
     "lrc", NULL},
    {"reloffset", (getter)MtkProjParam_getreloffset, (setter)MtkProjParam_setreloffset,
     "reloffset", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkProjParam_members[] = {
    {"path", T_INT, offsetof(MtkProjParam, pp.path), READONLY, "path"},
    {"projcode", T_LONG, offsetof(MtkProjParam, pp.projcode), READONLY,
     "projcode"},
    {"zonecode", T_LONG, offsetof(MtkProjParam, pp.zonecode), READONLY,
     "zonecode"},
    {"spherecode", T_LONG, offsetof(MtkProjParam, pp.spherecode), READONLY,
     "spherecode"},
    {"nblock", T_INT, offsetof(MtkProjParam, pp.nblock), READONLY, "nblock"},
    {"nline", T_INT, offsetof(MtkProjParam, pp.nline), READONLY, "nline"},
    {"nsample", T_INT, offsetof(MtkProjParam, pp.nsample), READONLY,
     "nsample"},
    {"resolution", T_INT, offsetof(MtkProjParam, pp.resolution), READONLY,
     "resolution"},
    {NULL}  /* Sentinel */
};

PyTypeObject MtkProjParamType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkProjParam", /*tp_name*/
    sizeof(MtkProjParam),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkProjParam_dealloc,/*tp_dealloc*/
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
    "MtkProjParam objects",    /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    MtkProjParam_members,      /* tp_members */
    MtkProjParam_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkProjParam_init,                         /* tp_init */
    0,                         /* tp_alloc */
    MtkProjParam_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkprojparam_methods[] = {
    {NULL}  /* Sentinel */
};
