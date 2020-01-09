/*===========================================================================
=                                                                           =
=                               MtkRegCoeff                                =
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
#include <numpy/numpyconfig.h>

#if (NPY_API_VERSION >= 0x00000007)
// NumPy >= 1.7
#   define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
// NumPy < 1.7
#   define NPY_ARRAY_IN_ARRAY   NPY_IN_ARRAY
#endif
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "MisrToolkit.h"
#include "pyMtk.h"
#include "pyhelpers.h"

static void
RegCoeff_dealloc(RegCoeff* self)
{
   MtkDataBufferFree(&self->valid_mask);
   MtkDataBufferFree(&self->slope);
   MtkDataBufferFree(&self->intercept);
   MtkDataBufferFree(&self->correlation);            
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
RegCoeff_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   RegCoeff *self;
   MTKt_DataBuffer mask_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer slope_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer intr_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer corr_init = MTKT_DATABUFFER_INIT;

   self = (RegCoeff *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->valid_mask = mask_init;
      self->slope = slope_init;
      self->intercept = intr_init;
      self->correlation = corr_init;
   }

   return (PyObject*)self;
}

int
RegCoeff_init(RegCoeff *self, PyObject *args, PyObject *kwds)
{
   MTKt_DataBuffer mask_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer slope_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer intr_init = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer corr_init = MTKT_DATABUFFER_INIT;
   self->valid_mask = mask_init;
   self->slope = slope_init;
   self->intercept = intr_init;
   self->correlation = corr_init;      
   return 0;	
}

static PyObject *
RegCoeff_valid_mask(RegCoeff *self) {
    PyObject *result;
    PyObject *mask_arr = NULL;
    Mtk_DataBufferToPyArray(&self->valid_mask, &mask_arr, NPY_UBYTE);    
    result = PyArray_Return((PyArrayObject *)mask_arr);
    return result;
}

static PyObject *
RegCoeff_slope(RegCoeff *self) {
    PyObject *result;
    PyObject *slope_arr = NULL;
    Mtk_DataBufferToPyArray(&self->slope, &slope_arr, NPY_FLOAT32);    
    result = PyArray_Return((PyArrayObject *)slope_arr);
    return result;
}

static PyObject *
RegCoeff_intercept(RegCoeff *self) {
    PyObject *result;
    PyObject *intercept_arr = NULL;
    Mtk_DataBufferToPyArray(&self->intercept, &intercept_arr, NPY_FLOAT32);    
    result = PyArray_Return((PyArrayObject *)intercept_arr);
    return result;
}
static PyObject *
RegCoeff_correlation(RegCoeff *self) {
    PyObject *result;
    PyObject *correlation_arr = NULL;
    Mtk_DataBufferToPyArray(&self->correlation, &correlation_arr, NPY_FLOAT32);    
    result = PyArray_Return((PyArrayObject *)correlation_arr);
    return result;
}
    

static PyMethodDef RegCoeff_methods[] = {
    {"valid_mask", (PyCFunction)RegCoeff_valid_mask, METH_NOARGS,
     "Return valid mask numpy array from Regression Coefficients structure."},
    {"slope", (PyCFunction)RegCoeff_slope, METH_NOARGS,
     "Return slope numpy array from Regression Coefficients structure."},
    {"intercept", (PyCFunction)RegCoeff_intercept, METH_NOARGS,
     "Return intercept numpy array from Regression Coefficients structure."},
    {"correlation", (PyCFunction)RegCoeff_correlation, METH_NOARGS,
     "Return correlation numpy array from Regression Coefficients structure."},
     
     
   {NULL}  /* Sentinel */
};

PyTypeObject RegCoeffType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkRegCoeff",      /*tp_name*/
    sizeof(RegCoeff),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)RegCoeff_dealloc,/*tp_dealloc*/
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
    "RegCoeff objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    RegCoeff_methods,            /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)RegCoeff_init,   /* tp_init */
    0,                         /* tp_alloc */
    RegCoeff_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef regcoeff_methods[] = {
    {NULL}  /* Sentinel */
};
