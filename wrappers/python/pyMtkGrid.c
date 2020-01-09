/*===========================================================================
=                                                                           =
=                                 MtkGrid                                   =
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
#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdlib.h>
#include "pyMtk.h"

extern PyTypeObject MtkFieldType;

static void
MtkGrid_dealloc(MtkGrid* self)
{
   int i;

   Py_XDECREF(self->filename);
   Py_XDECREF(self->gridname);

   for (i = 0; i < self->num_fields; ++i) {

     Py_XDECREF(self->fields[i]);

   }

   Py_XDECREF(self->file_id);

   PyMem_Free(self->fields);
   Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
MtkGrid_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkGrid *self;
   static char *kwlist[] = {"gridname", NULL};

   self = (MtkGrid *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
       #if PY_MAJOR_VERSION >= 3
           if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist,
                                            &self->gridname)) {
       #else
           if (!PyArg_ParseTupleAndKeywords(args, kwds, "S", kwlist,
                                            &self->gridname)) {
       #endif
               return NULL;
           }

       Py_INCREF(self->gridname);
   }

   return (PyObject*)self;
}

MtkGrid *
grid_init(MtkGrid *self, const char *filename, const char *gridname, MtkFileId *file_id)
{
   MTKt_status status;
   int num_fields;
   char **fieldlist;

   self->filename = PyString_FromString(filename);
   if (self->filename == NULL)
      return NULL;

   self->gridname = PyString_FromString(gridname);
   if (self->gridname == NULL)
      return NULL;

   if (file_id->ncid > 0) {
     status = MtkFileGridToFieldListNcid(file_id->ncid, gridname, &num_fields, &fieldlist);
   } else {
     status = MtkFileGridToFieldListFid(file_id->fid, gridname, &num_fields, &fieldlist);
   }
   if (status != MTK_SUCCESS)
      return NULL;

   /* Create MtkField Pointers */
   self->fields = PyMem_New(MtkField*, num_fields);
   self->num_fields = 0;
   self->max_fields = num_fields;
   self->file_id = file_id;

   Py_INCREF(self->file_id);

   MtkStringListFree(num_fields, &fieldlist);

   return self;
}

static PyObject *
Field(MtkGrid *self, PyObject *args)
{
   char *filename;
   char *gridname;
   char *fieldname;
   char *err_msg[] = MTK_ERR_DESC;
   int i;
   MTKt_status status;

   if (!PyArg_ParseTuple(args,"s",&fieldname))
	return NULL;

   /* Check if already created MtkField for this field */
   for (i = 0; i < self->num_fields; ++i)
      if (strcmp(fieldname,PyString_AsString(self->fields[i]->fieldname)) == 0)
      {

         Py_INCREF(self->fields[i]);
         return (PyObject*)self->fields[i];
      }
   
   /* Allocate memory for more fields if needed */
   if (self->num_fields == self->max_fields)
   {
      self->max_fields += 10;
      PyMem_Resize(self->fields, MtkField*, self->max_fields);
   }

   filename =PyString_AsString(self->filename);
   gridname = PyString_AsString(self->gridname);

   if (self->file_id->ncid > 0) {
     status = MtkFileGridFieldCheckNcid(self->file_id->ncid, gridname, fieldname);
   } else {
     status = MtkFileGridFieldCheckFid(self->file_id->fid, gridname, fieldname);
   }
   if (status == MTK_SUCCESS) {

     i = self->num_fields;

     self->fields[i] = PyObject_New(MtkField, &MtkFieldType);
     self->fields[i]->filename = PyString_FromString(filename);
     self->fields[i]->gridname = PyString_FromString(gridname);
     self->fields[i]->fieldname = PyString_FromString(fieldname);
     self->fields[i]->file_id = self->file_id;

     Py_INCREF(self->fields[i]->file_id);

     ++self->num_fields;
     
     Py_INCREF(self->fields[i]);
     return (PyObject*)self->fields[i];

   } else {

     PyErr_Format(PyExc_NameError, "Field: %s %s", fieldname, err_msg[status]);

     return NULL;
   }
}

static PyObject *
MtkGrid_getfield_list(MtkGrid *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;   /* Filename */
   char *gridname;   /* Gridname */
   int nfields;      /* Number of Fields */
   char **fieldlist; /* List of Fields */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   gridname = PyString_AsString(self->gridname);
   if (gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileGridToFieldListNcid(self->file_id->ncid,gridname,&nfields,&fieldlist);
   } else {
     status = MtkFileGridToFieldListFid(self->file_id->fid,gridname,&nfields,&fieldlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileGridToFieldList Failed");
      return NULL;
   }

   result = PyList_New(nfields);
   for (i = 0; i < nfields; ++i)
     PyList_SetItem(result, i, PyString_FromString(fieldlist[i]));

   MtkStringListFree(nfields,&fieldlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkGrid, field_list)

static PyObject *
MtkGrid_getnative_field_list(MtkGrid *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;   /* Filename */
   char *gridname;   /* Gridname */
   int nfields;      /* Number of Fields */
   char **fieldlist; /* List of Fields */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   gridname = PyString_AsString(self->gridname);
   if (gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileGridToNativeFieldListNcid(self->file_id->ncid,gridname,&nfields,&fieldlist);
   } else {
     status = MtkFileGridToNativeFieldListFid(self->file_id->fid,gridname,&nfields,&fieldlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileGridToFieldList Failed");
      return NULL;
   }

   result = PyList_New(nfields);
   for (i = 0; i < nfields; ++i)
     PyList_SetItem(result, i, PyString_FromString(fieldlist[i]));

   MtkStringListFree(nfields,&fieldlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkGrid, native_field_list)

static PyObject *
FieldDims(MtkGrid *self, PyObject *args)
{
   PyObject *result;
   PyObject *temp = NULL;
   MTKt_status status;
   char *filename;
   char *gridname;
   char *fieldname;
   int dimcnt;
   char **dimlist = NULL;
   int *dimsize = NULL;
   int i;
   
   if (!PyArg_ParseTuple(args,"s",&fieldname))
	 return NULL;
   
   filename = PyString_AsString(self->filename);
   gridname = PyString_AsString(self->gridname);
   
   if (filename == NULL || gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileGridFieldToDimListNcid(self->file_id->ncid, gridname, fieldname, &dimcnt,
                                            &dimlist, &dimsize);
   } else {
     status = MtkFileGridFieldToDimListFid(self->file_id->fid, gridname, fieldname, &dimcnt,
                                           &dimlist, &dimsize);
   }
   if (status != MTK_SUCCESS)
   {
      if (self->file_id->ncid > 0) {
        status = MtkFileGridFieldCheckNcid(self->file_id->ncid, gridname, fieldname);
      } else {
        status = MtkFileGridFieldCheckFid(self->file_id->fid, gridname, fieldname);
      }
      if (status == MTK_INVALID_FIELD)
        PyErr_Format(PyExc_NameError, "Field: %s Not Found.", fieldname);
      else
        PyErr_SetString(PyExc_StandardError, "MtkFileGridFieldToDimList Failed");
      
      return NULL;
   }

   result = PyList_New(dimcnt);
   if (result == NULL)
   {
      PyErr_SetString(PyExc_MemoryError, "Couldn't create list.");
      goto ERROR_HANDLE;
   }

   for (i = 0; i < dimcnt; ++i)
   {
   	  temp = PyTuple_New(2);
   	  PyTuple_SetItem(temp,0,PyString_FromString(dimlist[i]));
   	  PyTuple_SetItem(temp,1,PyInt_FromLong(dimsize[i]));
   	    	  
      if (PyList_SetItem(result,i,temp))
      {
         PyErr_SetString(PyExc_MemoryError, "Couldn't insert item into"
                                             " list.");
         goto ERROR_HANDLE;
      }  
   }

   MtkStringListFree(dimcnt,&dimlist);
   free(dimsize);

   return result;
   
ERROR_HANDLE:
   MtkStringListFree(dimcnt,&dimlist);
   free(dimsize);
   Py_XDECREF(temp);
   Py_XDECREF(result);
   return NULL;   
}

static PyObject *
MtkGrid_getresolution(MtkGrid *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename; /* File name */
   char *gridname; /* Grid name */
   int resolution; /* Resolution */ 

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   gridname = PyString_AsString(self->gridname);
   if (gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileGridToResolutionNcid(self->file_id->ncid,gridname,&resolution);
   } else {
     status = MtkFileGridToResolutionFid(self->file_id->fid,gridname,&resolution);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileGridToResolution Failed");
      return NULL;
   }

   result = Py_BuildValue("i",resolution);
   return result;
}

MTK_READ_ONLY_ATTR(MtkGrid, resolution)

static PyObject *
MtkGrid_getgrid_name(MtkGrid *self, void *closure)
{
   Py_INCREF(self->gridname);
   return self->gridname;
}

MTK_READ_ONLY_ATTR(MtkGrid, grid_name)

static PyObject *
AttrGet(MtkGrid *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;
   char *gridname;
   char *attrname;
   MTKt_DataBuffer attrbuf;

   if (!PyArg_ParseTuple(args,"s",&attrname))
	return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   gridname = PyString_AsString(self->gridname);
   if (gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkGridAttrGetNcid(self->file_id->ncid,gridname,attrname,&attrbuf);
   } else {
     status = MtkGridAttrGetFid(self->file_id->fid,gridname,attrname,&attrbuf);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkGridAttrGet Failed");
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
MtkGrid_getattr_list(MtkGrid *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;   /* Filename */
   char *gridname;   /* Gridname */
   int num_attrs;    /* Number of attributes */
   char **attrlist; /* Attribute list */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
      return NULL;

   gridname = PyString_AsString(self->gridname);
   if (gridname == NULL)
      return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkGridAttrListNcid(self->file_id->ncid,gridname,&num_attrs,&attrlist);
   } else {
     status = MtkGridAttrListFid(self->file_id->fid,gridname,&num_attrs,&attrlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkGridAttrList Failed");
      return NULL;
   }

   result = PyList_New(num_attrs);
   for (i = 0; i < num_attrs; ++i)
     PyList_SetItem(result, i, PyString_FromString(attrlist[i]));

   MtkStringListFree(num_attrs,&attrlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkGrid, attr_list)

static PyGetSetDef MtkGrid_getseters[] = {
   {"field_list", (getter)MtkGrid_getfield_list, (setter)MtkGrid_setfield_list,
    "List of field names.", NULL},
   {"native_field_list", (getter)MtkGrid_getnative_field_list,
    (setter)MtkGrid_setnative_field_list, "List of native field names.", NULL},
   {"grid_name", (getter)MtkGrid_getgrid_name, (setter)MtkGrid_setgrid_name,
    "Grid name.", NULL},
   {"attr_list", (getter)MtkGrid_getattr_list, (setter)MtkGrid_setattr_list,
    "List of attribute names.", NULL},
   {"resolution", (getter)MtkGrid_getresolution, (setter)MtkGrid_setresolution,
    "Resolution of grid in meters.", NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef MtkGrid_methods[] = {
   {"field", (PyCFunction)Field, METH_VARARGS,
    "Return MtkField."},
   {"field_dims", (PyCFunction)FieldDims, METH_VARARGS,
    "Get extra dimension names and sizes"},
   {"attr_get", (PyCFunction)AttrGet, METH_VARARGS,
    "Get a grid attribute."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyTypeObject MtkGridType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkGrid",      /*tp_name*/
    sizeof(MtkGrid),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkGrid_dealloc,/*tp_dealloc*/
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
    "MtkGrid objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkGrid_methods,           /* tp_methods */
    0,                         /* tp_members */
    MtkGrid_getseters,         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    MtkGrid_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkgrid_methods[] = {
    {NULL}  /* Sentinel */
};
