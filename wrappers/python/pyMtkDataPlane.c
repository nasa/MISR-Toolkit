/*===========================================================================
=                                                                           =
=                               MtkDataPlane                                =
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
#define PY_ARRAY_UNIQUE_SYMBOL PY_MTK_EXT
#define NO_IMPORT_ARRAY
#include <numpy/numpyconfig.h>

#if (NPY_API_VERSION >= 0x00000007)
// NumPy >= 1.7
#   define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
// NumPy < 1.7
#   define NPY_ARRAY_IN_ARRAY   NPY_IN_ARRAY
#endif
#include "numpy/arrayobject.h"
#include "MisrToolkit.h"
#include "pyMtk.h"

extern PyTypeObject MtkMapInfoType;
extern int MtkMapInfo_init(MtkMapInfo *self, PyObject *args, PyObject *kwds);
extern int MtkMapInfo_copy(MtkMapInfo *self, MTKt_MapInfo mapinfo);

static void
DataPlane_dealloc(DataPlane* self)
{
   MtkDataBufferFree(&self->databuf);
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
DataPlane_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   DataPlane *self;
   MTKt_DataBuffer db_init = MTKT_DATABUFFER_INIT;
   MTKt_MapInfo mi_init = MTKT_MAPINFO_INIT;

   self = (DataPlane *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->databuf = db_init;
      self->mapinfo = mi_init;
   }

   return (PyObject*)self;
}

int
DataPlane_init(DataPlane *self, PyObject *args, PyObject *kwds)
{
   MTKt_DataBuffer db_init = MTKT_DATABUFFER_INIT;
   MTKt_MapInfo mi_init = MTKT_MAPINFO_INIT;

   self->databuf = db_init;
   self->mapinfo = mi_init;
   
   return 0;	
}


static PyObject *
DataPlane_data(DataPlane *self)
{
   PyObject *result;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dims[2] = {self->databuf.nline, self->databuf.nsample};
   PyArrayObject *data = NULL;

   /* Determine NumPy data type */
   switch (self->databuf.datatype)
   {
      case MTKe_void : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_VOID);
	     break;
      case MTKe_char8 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_BYTE);
	     break;
      case MTKe_uchar8 :data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UBYTE);
	     break;
      case MTKe_int8 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_BYTE);
	     break;
      case MTKe_uint8 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UBYTE);
	     break;
      case MTKe_int16 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_SHORT);
	     break;
      case MTKe_uint16 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_USHORT);
	     break;
      case MTKe_int32 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT);
	     break;
      case MTKe_uint32 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT);
	     break;
      case MTKe_int64 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_LONG);
	     break;
      case MTKe_uint64 : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_ULONG);
	     break;
      case MTKe_float : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT);
	     break;
      case MTKe_double : data = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	     break;
   }

   if (data == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(data), self->databuf.dataptr, self->databuf.datasize *
          self->databuf.nline * self->databuf.nsample);

   result = PyArray_Return(data);
   return result;
}

static PyObject *
DataPlane_mapinfo(DataPlane *self)
{
  /*PyObject *result;*/
   MtkMapInfo *mapinfo;

   mapinfo = (MtkMapInfo*)PyObject_New(MtkMapInfo, &MtkMapInfoType);
   MtkMapInfo_init(mapinfo,NULL,NULL);

   /* Copy data into MtkMapInfo */
   MtkMapInfo_copy(mapinfo,self->mapinfo);

   return (PyObject*)mapinfo;
}

static PyMethodDef DataPlane_methods[] = {
   {"data", (PyCFunction)DataPlane_data, METH_NOARGS,
    "Return NumPy with data read."},
   {"mapinfo", (PyCFunction)DataPlane_mapinfo, METH_NOARGS,
    "Return MtkMapInfo."},
   {NULL}  /* Sentinel */
};

PyTypeObject DataPlaneType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkDataPlane",      /*tp_name*/
    sizeof(DataPlane),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)DataPlane_dealloc,/*tp_dealloc*/
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
    "DataPlane objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    DataPlane_methods,            /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)DataPlane_init,   /* tp_init */
    0,                         /* tp_alloc */
    DataPlane_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef dataplane_methods[] = {
    {NULL}  /* Sentinel */
};
