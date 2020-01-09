/*===========================================================================
=                                                                           =
=                              MtkTimeMetaData                              =
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

static void
MtkTimeMetaData_dealloc(MtkTimeMetaData* self)
{
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkTimeMetaData_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkTimeMetaData *self;
   MTKt_TimeMetaData time_metadata = MTKT_TIME_METADATA_INIT;

   self = (MtkTimeMetaData *)type->tp_alloc(type, 0);
   if (self != NULL)
      self->time_metadata = time_metadata;

   return (PyObject*)self;
}

int
MtkTimeMetaData_init(MtkTimeMetaData *self, PyObject *args, PyObject *kwds)
{
   MTKt_TimeMetaData time_metadata = MTKT_TIME_METADATA_INIT;
   
   self->time_metadata = time_metadata;

   return 0;
}

static PyObject *
MtkTimeMetaData_PixelTime(MtkTimeMetaData *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   double som_x;
   double som_y;
   char pixel_time[MTKd_DATETIME_LEN];

   if (!PyArg_ParseTuple(args,"dd",&som_x,&som_y))
      return NULL;

   status = MtkPixelTime(self->time_metadata, som_x, som_y, pixel_time);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkPixelTime Failed");
      return NULL;
   }

   result = Py_BuildValue("s",pixel_time);
   return result;
}

static PyObject *
MtkTimeMetaData_getcamera(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;

   result = PyString_FromString(self->time_metadata.camera);
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, camera)

static PyObject *
MtkTimeMetaData_getnumber_transform(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *number_transform;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];

   dim_size[0] = NBLOCK + 1;
   number_transform = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_INT);

   if (number_transform == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(number_transform),self->time_metadata.number_transform,
          sizeof(MTKt_int32) * (NBLOCK + 1)); 
   
   result = Py_BuildValue("N",PyArray_Return(number_transform));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, number_transform)

static PyObject *
MtkTimeMetaData_getref_time(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyObject *temp_list;
   int i;
   int j;
   
   result = PyList_New(NBLOCK + 1);
   for (i = 0; i < NBLOCK + 1; ++i)
   {
   	 temp_list = PyList_New(NGRIDCELL);
   	 for (j = 0; j < NGRIDCELL; ++j)
   	    PyList_SetItem(temp_list, j, PyString_FromString(self->time_metadata.ref_time[i][j])); 

     PyList_SetItem(result, i, temp_list);
   }
   
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, ref_time)

static PyObject *
MtkTimeMetaData_getstart_line(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *start_line;
   const int dim = 2;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[2];

   dim_size[0] = NBLOCK + 1;
   dim_size[1] = NGRIDCELL;
   start_line = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_INT);

   if (start_line == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(start_line),self->time_metadata.start_line,
          sizeof(MTKt_int32) * (NBLOCK + 1) * NGRIDCELL); 
   
   result = Py_BuildValue("N",PyArray_Return(start_line));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, start_line)

static PyObject *
MtkTimeMetaData_getnumber_line(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *number_line;
   const int dim = 2;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[2];

   dim_size[0] = NBLOCK + 1;
   dim_size[1] = NGRIDCELL;
   number_line = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_INT);

   if (number_line == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(number_line),self->time_metadata.number_line,
          sizeof(MTKt_int32) * (NBLOCK + 1) * NGRIDCELL); 
   
   result = Py_BuildValue("N",PyArray_Return(number_line));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, number_line)

static PyObject *
MtkTimeMetaData_getcoeff_line(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *coeff_line;
   const int dim = 3;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[3];

   dim_size[0] = NBLOCK + 1;
   dim_size[1] = 6;
   dim_size[2] = NGRIDCELL;
   coeff_line = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

   if (coeff_line == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(coeff_line),self->time_metadata.coeff_line,
          sizeof(MTKt_double) * (NBLOCK + 1) * 6 * NGRIDCELL); 
   
   result = Py_BuildValue("N",PyArray_Return(coeff_line));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, coeff_line)

static PyObject *
MtkTimeMetaData_getsom_ctr_x(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *som_ctr_x;
   const int dim = 2;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[2];

   dim_size[0] = NBLOCK + 1;
   dim_size[1] = NGRIDCELL;
   som_ctr_x = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

   if (som_ctr_x == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(som_ctr_x),self->time_metadata.som_ctr_x,
          sizeof(MTKt_double) * (NBLOCK + 1) * NGRIDCELL); 
   
   result = Py_BuildValue("N",PyArray_Return(som_ctr_x));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, som_ctr_x)

static PyObject *
MtkTimeMetaData_getsom_ctr_y(MtkTimeMetaData *self, void *closure)
{
   PyObject *result;
   PyArrayObject *som_ctr_y;
   const int dim = 2;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[2];

   dim_size[0] = NBLOCK + 1;
   dim_size[1] = NGRIDCELL;
   som_ctr_y = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

   if (som_ctr_y == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(som_ctr_y),self->time_metadata.som_ctr_y,
          sizeof(MTKt_double) * (NBLOCK + 1) * NGRIDCELL); 
   
   result = Py_BuildValue("N",PyArray_Return(som_ctr_y));
   return result;
}

MTK_READ_ONLY_ATTR(MtkTimeMetaData, som_ctr_y)

static PyGetSetDef MtkTimeMetaData_getseters[] = {
	{"camera", (getter)MtkTimeMetaData_getcamera, (setter)MtkTimeMetaData_setcamera,
     "Camera", NULL},
    {"number_transform",
     (getter)MtkTimeMetaData_getnumber_transform, (setter)MtkTimeMetaData_setnumber_transform,
     "Number of transforms.", NULL},
    {"ref_time", (getter)MtkTimeMetaData_getref_time, (setter)MtkTimeMetaData_setref_time,
     "Reference time.", NULL},
    {"start_line", 
     (getter)MtkTimeMetaData_getstart_line, (setter)MtkTimeMetaData_setstart_line,
     "Starting line.", NULL},
    {"number_line", 
     (getter)MtkTimeMetaData_getnumber_line, (setter)MtkTimeMetaData_setnumber_line,
     "Number of lines.", NULL},
    {"coeff_line", 
     (getter)MtkTimeMetaData_getcoeff_line, (setter)MtkTimeMetaData_setcoeff_line,
     "Line transform coefficients.", NULL},
    {"som_ctr_x", 
     (getter)MtkTimeMetaData_getsom_ctr_x, (setter)MtkTimeMetaData_setsom_ctr_x,
     "SOM X center coordinates.", NULL},
    {"som_ctr_y", 
     (getter)MtkTimeMetaData_getsom_ctr_y, (setter)MtkTimeMetaData_setsom_ctr_y,
     "SOM Y center coordinates.", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkTimeMetaData_members[] = {
    {"path", T_INT, offsetof(MtkTimeMetaData, time_metadata.path), READONLY,
     "Path number"},
    {"start_block", T_INT, offsetof(MtkTimeMetaData, time_metadata.start_block), READONLY,
     "Start Block Number"},
    {"end_block", T_INT, offsetof(MtkTimeMetaData, time_metadata.end_block), READONLY,
     "End Block Number"},
    {NULL}  /* Sentinel */
};

static PyMethodDef MtkTimeMetaData_methods[] = {
   {"pixel_time", (PyCFunction)MtkTimeMetaData_PixelTime, METH_VARARGS,
    "Pixel time of Som X and Som Y."}, 
   {NULL}  /* Sentinel */
};

PyTypeObject MtkTimeMetaDataType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkTimeMetaData",  /*tp_name*/
    sizeof(MtkTimeMetaData),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkTimeMetaData_dealloc,/*tp_dealloc*/
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
    "MtkTimeMetaData objects",      /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkTimeMetaData_methods,        /* tp_methods */
    MtkTimeMetaData_members,        /* tp_members */
    MtkTimeMetaData_getseters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkTimeMetaData_init, /* tp_init */
    0,                         /* tp_alloc */
    MtkTimeMetaData_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtktimemetadata_methods[] = {
    {NULL}  /* Sentinel */
};
