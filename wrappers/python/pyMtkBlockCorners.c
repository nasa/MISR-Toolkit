/*===========================================================================
=                                                                           =
=                            MtkBlockCorners                                =
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

extern PyTypeObject MtkGeoBlockType;
extern int MtkGeoBlock_init(MtkGeoBlock *self, PyObject *args, PyObject *kwds);

static void
MtkBlockCorners_dealloc(MtkBlockCorners* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkBlockCorners_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkBlockCorners *self;

   self = (MtkBlockCorners *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->path = 0;
      self->start_block = 0;
      self->end_block = 0;
   }

   return (PyObject *)self;
}

int
MtkBlockCorners_init(MtkBlockCorners *self, PyObject *args, PyObject *kwds)
{
   int i;

   self->path = 0;
   self->start_block = 0;
   self->end_block = 0;

   for (i = 0; i < NBLOCK + 1; ++i)
   {
      self->gb[i] = (MtkGeoBlock*)PyObject_New(MtkGeoBlock, &MtkGeoBlockType);
      if (self->gb[i] == NULL)
	 return -1;
      MtkGeoBlock_init(self->gb[i],NULL,NULL);
   }
  
   return 0;
}

static PyObject *
MtkBlockCorners_getblock(MtkBlockCorners *self, void *closure)
{
   PyObject *result;
   int i;

   result = PyTuple_New(NBLOCK + 1);
   if (result == NULL)
      return NULL;

   for (i = 0; i < NBLOCK + 1; ++i)
   {
      /* INCREF because PyTuple_SetItem steals a reference. */
      Py_INCREF(self->gb[i]); 
      PyTuple_SetItem(result, i, (PyObject*)self->gb[i]);
   }

   return result;
}

MTK_READ_ONLY_ATTR(MtkBlockCorners, block)


static PyGetSetDef MtkBlockCorners_getseters[] = {
    {"block", (getter)MtkBlockCorners_getblock, (setter)MtkBlockCorners_setblock,
     "Block coordinates indexed by 1-based block number.", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkBlockCorners_members[] = {
    {"path", T_INT, offsetof(MtkBlockCorners, path), READONLY, "path"},
    {"start_block", T_INT, offsetof(MtkBlockCorners, start_block), READONLY,
     "start block"},
    {"end_block", T_INT, offsetof(MtkBlockCorners, end_block), READONLY,
     "end block"},
    {NULL}  /* Sentinel */
};

PyTypeObject MtkBlockCornersType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkBlockCorners", /*tp_name*/
    sizeof(MtkBlockCorners),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkBlockCorners_dealloc,/*tp_dealloc*/
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
    "MtkBlockCorners objects",    /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    MtkBlockCorners_members,      /* tp_members */
    MtkBlockCorners_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkBlockCorners_init, /* tp_init */
    0,                         /* tp_alloc */
    MtkBlockCorners_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkblockcorners_methods[] = {
    {NULL}  /* Sentinel */
};
