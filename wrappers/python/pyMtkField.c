/*===========================================================================
=                                                                           =
=                                 MtkField                                  =
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
#include "pyMtk.h"
#include "numpy/arrayobject.h"
#include "MisrToolkit.h"
#include <stdlib.h>


extern PyTypeObject RegionType;
extern PyTypeObject DataPlaneType;
int DataPlane_init(DataPlane *self, PyObject *args, PyObject *kwds);

static void
MtkField_dealloc(MtkField* self)
{
   Py_XDECREF(self->filename);
   Py_XDECREF(self->gridname);
   Py_XDECREF(self->fieldname);

   Py_XDECREF(self->file_id);

   Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
MtkField_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkField *self;
   static char *kwlist[] = {"fieldname", NULL};

   self = (MtkField *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
        #if PY_MAJOR_VERSION >= 3
            if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist,
                                             &self->fieldname)) {
        #else
            if (!PyArg_ParseTupleAndKeywords(args, kwds, "S", kwlist,
                                             &self->fieldname)) {
        #endif
              return NULL;
            }

      Py_INCREF(self->fieldname);
   }

   return (PyObject*)self;
}

static int
MtkField_init(MtkField *self, PyObject *args, PyObject *kwds)
{
   char *file_name;
   char *grid_name;
   char *field_name;

   file_name = PyString_AsString(self->filename);
   if (file_name == NULL)
      return -1;

   grid_name = PyString_AsString(self->gridname);
   if (grid_name == NULL)
      return -1;

   field_name = PyString_AsString(self->fieldname);
   if (field_name == NULL)
      return -1;

   return 0;
}

static PyObject *
ReadData(MtkField *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   Region *region;
   DataPlane *data;
   int start_block;
   int end_block;
   PyArrayObject *data3d = NULL;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dims[3];
   PyObject *arg1; /* region or start_block */
   PyObject *arg2; /* end_block */
   MTKt_DataBuffer3D databuf;
   char *err_msg[] = MTK_ERR_DESC;

   if (!PyArg_ParseTuple(args,"O|O",&arg1,&arg2))
      return NULL;

   if (PyTuple_Size(args) == 1) /* Read using MtkRegion */
   {
      region = (Region*)arg1;
      if (!PyObject_TypeCheck((PyObject*)region, &RegionType))
      {
         PyErr_SetString(PyExc_TypeError, "Argument is not a Region.");
         return NULL;
      }

      data = PyObject_New(DataPlane, &DataPlaneType);
      DataPlane_init(data,NULL,NULL);

      if (self->file_id->ncid > 0) {
        status = MtkReadDataNcid(self->file_id->ncid,
                                PyString_AsString(self->gridname),
                                PyString_AsString(self->fieldname),
                                region->region,
                                &data->databuf,
                                &data->mapinfo);
      } else {
        status = MtkReadDataFid(self->file_id->fid,
                                PyString_AsString(self->gridname),
                                PyString_AsString(self->fieldname),
                                region->region,
                                &data->databuf,
                                &data->mapinfo);
      }
      if (status != MTK_SUCCESS)
      {
         PyErr_Format(PyExc_StandardError, "MtkReadData Failed: %s",err_msg[status]); 
         Py_DECREF(data);

         return NULL;
      }

      return (PyObject*)data;
   }

   /* Read using block range. */
   if (!(PyInt_Check(arg1) && PyInt_Check(arg2)))
   {
      PyErr_SetString(PyExc_TypeError, "Arguments must be Int.");
      return NULL;
   }

   start_block = (int) PyInt_AsLong(arg1);
   end_block = (int) PyInt_AsLong(arg2);

   if (self->file_id->ncid > 0) {
     status = MtkReadBlockRangeNcid(self->file_id->ncid,
                                   PyString_AsString(self->gridname),
                                   PyString_AsString(self->fieldname),
                                   start_block, end_block, &databuf);
   } else {
     status = MtkReadBlockRangeFid(self->file_id->fid,
                                   PyString_AsString(self->gridname),
                                   PyString_AsString(self->fieldname),
                                   start_block, end_block, &databuf);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_Format(PyExc_StandardError, "MtkReadBlockRange Failed: %s",err_msg[status]);
      return NULL;
   }

   dims[0] = databuf.nblock;
   dims[1] = databuf.nline;
   dims[2] = databuf.nsample;

   /* Determine NumPy data type */
   switch (databuf.datatype)
   {
      case MTKe_void : data3d = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_VOID);
	     break;
      case MTKe_char8 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_BYTE);
	     break;
      case MTKe_uchar8 :data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UBYTE);
	     break;
      case MTKe_int8 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_BYTE);
	     break;
      case MTKe_uint8 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UBYTE);
	     break;
      case MTKe_int16 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_SHORT);
	     break;
      case MTKe_uint16 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_USHORT);
	     break;
      case MTKe_int32 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_INT);
	     break;
      case MTKe_uint32 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_UINT);
	     break;
      case MTKe_int64 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_LONG);
	     break;
      case MTKe_uint64 : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_ULONG);
	     break;
      case MTKe_float : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT);
	     break;
      case MTKe_double : data3d = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
	     break;
   }

   if (data3d == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
     return NULL;
   }

   memcpy(PyArray_DATA(data3d), databuf.dataptr, databuf.datasize * databuf.nblock *
          databuf.nline * databuf.nsample);

   MtkDataBufferFree3D(&databuf);

   result = PyArray_Return(data3d);
   return result;
}

static PyObject *
MtkField_getdata_type(MtkField *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   MTKt_DataType datatype;
   char *types[] = MTKd_DataType;

   if (self->file_id->ncid > 0) {
     status = MtkFileGridFieldToDataTypeNcid(self->file_id->ncid,
                                            PyString_AsString(self->gridname),
                                            PyString_AsString(self->fieldname),
                                            &datatype);
   } else {
     status = MtkFileGridFieldToDataTypeFid(self->file_id->fid,
                                            PyString_AsString(self->gridname),
                                            PyString_AsString(self->fieldname),
                                            &datatype);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileGridFieldToDataType Failed");
      return NULL;
   }

   result = PyString_FromString(types[datatype]);
   return result;
}

MTK_READ_ONLY_ATTR(MtkField, data_type)

static PyObject *
MtkField_getfill_value(MtkField *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   MTKt_DataBuffer fillbuf = MTKT_DATABUFFER_INIT;

   if (self->file_id->ncid > 0) {
     status = MtkFillValueGetNcid(self->file_id->ncid,
                                 PyString_AsString(self->gridname),
                                 PyString_AsString(self->fieldname),
                                 &fillbuf);
   } else {
     status = MtkFillValueGetFid(self->file_id->fid,
                                 PyString_AsString(self->gridname),
                                 PyString_AsString(self->fieldname),
                                 &fillbuf);
   }
   if (status != MTK_SUCCESS) /* File doesn't have a fill value */
   {
      result = Py_BuildValue("i", 0);
      return result;
   }

   switch (fillbuf.datatype)
   {
      case MTKe_char8 : result = Py_BuildValue("i", fillbuf.data.c8[0][0]);
	 break;
      case MTKe_uchar8 : result = Py_BuildValue("i", fillbuf.data.uc8[0][0]);
	 break;
      case MTKe_int8 : result = Py_BuildValue("i", fillbuf.data.i8[0][0]);
	 break;
      case MTKe_uint8 : result = Py_BuildValue("i", fillbuf.data.u8[0][0]);
	 break;
      case MTKe_int16 : result = Py_BuildValue("i", fillbuf.data.i16[0][0]);
	 break;
      case MTKe_uint16 : result = Py_BuildValue("i", fillbuf.data.u16[0][0]);
	 break;
      case MTKe_int32 : result = Py_BuildValue("i", fillbuf.data.i32[0][0]);
	 break;
      case MTKe_uint32 : result = Py_BuildValue("i", fillbuf.data.u32[0][0]);
	 break;
      case MTKe_int64 : result = Py_BuildValue("l", fillbuf.data.i64[0][0]);
	 break;
      case MTKe_uint64 : result = Py_BuildValue("l", fillbuf.data.u64[0][0]);
	 break;
      case MTKe_float : result = Py_BuildValue("f",  fillbuf.data.f[0][0]);
	 break;
      case MTKe_double : result = Py_BuildValue("d",  fillbuf.data.d[0][0]);
	 break;
      default : result = Py_BuildValue("i", 0);
	break;
   }

   MtkDataBufferFree(&fillbuf);

   return result;
}

MTK_READ_ONLY_ATTR(MtkField, fill_value)

static PyObject *
MtkField_getfield_name(MtkField *self, void *closure)
{ 
   Py_INCREF(self->fieldname);
   return self->fieldname;
}

MTK_READ_ONLY_ATTR(MtkField, field_name)

static PyObject *
AttrGet(MtkField *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;
   char *fieldname;
   char *attrname;
   MTKt_DataBuffer attrbuf;

   if (!PyArg_ParseTuple(args,"s",&attrname))
  return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   fieldname = PyString_AsString(self->fieldname);
   if (fieldname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFieldAttrGetNcid(self->file_id->ncid,fieldname,attrname,&attrbuf);
   } else {
     status = MtkFieldAttrGetFid(self->file_id->fid,fieldname,attrname,&attrbuf);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFieldAttrGet Failed");
      return NULL;
   }

   switch (attrbuf.datatype)
   {
      case MTKe_void : result = NULL;
  break;
      case  MTKe_char8 : result = Py_BuildValue("c",attrbuf.data.c8[0][0]);
  break;
      case  MTKe_uchar8 : result = Py_BuildValue("c",attrbuf.data.uc8[0][0]);
  break;
      case  MTKe_int8 : result = Py_BuildValue("i",attrbuf.data.i8[0][0]);
  break;
      case  MTKe_uint8 : result = Py_BuildValue("i",attrbuf.data.u8[0][0]);
  break;
      case  MTKe_int16 : result = Py_BuildValue("i",attrbuf.data.i16[0][0]);
  break;
      case  MTKe_uint16 : result = Py_BuildValue("i",attrbuf.data.u16[0][0]);
  break;
      case  MTKe_int32 : result = Py_BuildValue("i",attrbuf.data.i32[0][0]);
  break;
      case  MTKe_uint32 : result = Py_BuildValue("i",attrbuf.data.u32[0][0]);
  break;
      case  MTKe_int64 : result = Py_BuildValue("l",attrbuf.data.i64[0][0]);
  break;
      case  MTKe_uint64 : result = Py_BuildValue("l",attrbuf.data.u64[0][0]);
  break;
      case  MTKe_float : result = Py_BuildValue("f",attrbuf.data.f[0][0]);
  break;
      case  MTKe_double : result = Py_BuildValue("f",attrbuf.data.d[0][0]);
  break;
      default : result = NULL;
        break;
   }

   MtkDataBufferFree(&attrbuf);

   return result;
}


static PyObject *
MtkField_getattr_list(MtkField *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;   /* Filename */
   char *fieldname;   /* Field name */
   int num_attrs;    /* Number of attributes */
   char **attrlist; /* Attribute list */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   fieldname = PyString_AsString(self->fieldname);
   if (fieldname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFieldAttrListNcid(self->file_id->ncid,fieldname,&num_attrs,&attrlist);
   } else {
     status = MtkFieldAttrListFid(self->file_id->fid,fieldname,&num_attrs,&attrlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFieldAttrList Failed");
      return NULL;
   }

   result = PyList_New(num_attrs);
   for (i = 0; i < num_attrs; ++i)
     PyList_SetItem(result, i, PyString_FromString(attrlist[i]));

   MtkStringListFree(num_attrs,&attrlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkField, attr_list)

static PyGetSetDef MtkField_getseters[] = {
   {"data_type", (getter)MtkField_getdata_type, (setter)MtkField_setdata_type,
    "Data type of field.", NULL},
   {"fill_value", (getter)MtkField_getfill_value, (setter)MtkField_setfill_value,
    "Fill value.", NULL},
   {"field_name", (getter)MtkField_getfield_name, (setter)MtkField_setfield_name,
    "Field name.", NULL},
    {"attr_list", (getter)MtkField_getattr_list, (setter)MtkField_setattr_list,
        "List of attribute names.", NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef MtkField_methods[] = {
   {"read", (PyCFunction)ReadData, METH_VARARGS,
    "Read data from field."},
    {"attr_get", (PyCFunction)AttrGet, METH_VARARGS,
        "Get a field attribute."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyTypeObject MtkFieldType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkField",      /*tp_name*/
    sizeof(MtkField),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkField_dealloc,/*tp_dealloc*/
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
    "MtkField objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkField_methods,          /* tp_methods */
    0,                         /* tp_members */
    MtkField_getseters,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkField_init,    /* tp_init */
    0,                         /* tp_alloc */
    MtkField_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkfield_methods[] = {
    {NULL}  /* Sentinel */
};
