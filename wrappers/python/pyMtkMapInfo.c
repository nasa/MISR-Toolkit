/*===========================================================================
=                                                                           =
=                                MtkMapInfo                                 =
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

extern PyTypeObject MtkSomRegionType;
extern int MtkSomRegion_init(MtkSomRegion *self, PyObject *args, PyObject *kwds);
extern PyTypeObject MtkGeoRegionType;
extern int MtkGeoRegion_init(MtkGeoRegion *self, PyObject *args, PyObject *kwds);
extern PyTypeObject MtkProjParamType;
extern int MtkProjParam_init(MtkProjParam *self, PyObject *args, PyObject *kwds);

static void
MtkMapInfo_dealloc(MtkMapInfo* self)
{
   Py_XDECREF(self->pp);
   Py_XDECREF(self->som);
   Py_XDECREF(self->geo);
   Py_XDECREF(self->pixelcenter);
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkMapInfo_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkMapInfo *self;

   self = (MtkMapInfo *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->pixelcenter = PyBool_FromLong(0);
      self->som = (MtkSomRegion*)PyObject_New(MtkSomRegion, &MtkSomRegionType);
      MtkSomRegion_init(self->som,NULL,NULL);
      self->geo = (MtkGeoRegion*)PyObject_New(MtkGeoRegion, &MtkGeoRegionType);
      MtkGeoRegion_init(self->geo,NULL,NULL);
      self->pp = (MtkProjParam*)PyObject_New(MtkProjParam, &MtkProjParamType);
      MtkProjParam_init(self->pp,NULL,NULL);

      if (self->pixelcenter == NULL || self->som == NULL || self->pp == NULL)
      {
         PyErr_Format(PyExc_StandardError, "Problem initializing MtkMapInfo.");
         return NULL;
      }
   }

   return (PyObject*)self;
}

int
MtkMapInfo_init(MtkMapInfo *self, PyObject *args, PyObject *kwds)
{
   self->pixelcenter = PyBool_FromLong(0);
   self->som = (MtkSomRegion*)PyObject_New(MtkSomRegion, &MtkSomRegionType);
   MtkSomRegion_init(self->som,NULL,NULL);
   self->geo = (MtkGeoRegion*)PyObject_New(MtkGeoRegion, &MtkGeoRegionType);
   MtkGeoRegion_init(self->geo,NULL,NULL);
   self->pp = (MtkProjParam*)PyObject_New(MtkProjParam, &MtkProjParamType);
   MtkProjParam_init(self->pp,NULL,NULL);

   if (self->pixelcenter == NULL || self->som == NULL || self->pp == NULL)
   {
      PyErr_Format(PyExc_StandardError, "Problem initializing MtkMapInfo.");
      return -1;
   }

   return 0;
}

/* Copy MTKt_MapInfo into MtkMapInfo */
int MtkMapInfo_copy(MtkMapInfo *self, MTKt_MapInfo mapinfo)
{
   Py_XDECREF(self->pixelcenter);
   self->pixelcenter = PyBool_FromLong(mapinfo.pixelcenter);
   self->som->path =  mapinfo.som.path;
   self->som->ulc->som_coord = mapinfo.som.ulc;
   self->som->ctr->som_coord = mapinfo.som.ctr;
   self->som->lrc->som_coord = mapinfo.som.lrc;
   self->geo->ulc->gc = mapinfo.geo.ulc;
   self->geo->urc->gc = mapinfo.geo.urc;
   self->geo->ctr->gc = mapinfo.geo.ctr;
   self->geo->lrc->gc = mapinfo.geo.lrc;
   self->geo->llc->gc = mapinfo.geo.llc;   
   self->pp->pp = mapinfo.pp;
   self->mapinfo = mapinfo;

   return 0;
}

static PyObject *
MtkMapInfo_LSToSomXY(MtkMapInfo *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int nelement;
   PyObject *oline, *osample;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   float line = 0.0;
   float sample = 0.0;
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x;
   double som_y;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"OO",&oline,&osample))
      return NULL;

   if (PyArray_Check(oline) && PyArray_Check(osample))
   {
      use_array_func = 1;
      line_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oline, NPY_FLOAT, 1, 1);
      sample_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osample, NPY_FLOAT, 1, 1);

      if (line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      if (PyArray_DIM(line_arr, 0) != PyArray_DIM(sample_arr,0))
      {
         PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	 goto ERROR_HANDLE;
      }
   }
   else if ((PyFloat_Check(oline) || PyInt_Check(oline)) &&
            (PyFloat_Check(osample) || PyInt_Check(osample)))
   {
      use_array_func = 0;
      line = (float)PyFloat_AsDouble(oline);
      sample = (float)PyFloat_AsDouble(osample);
   }
   else
   {
       PyErr_SetString(PyExc_TypeError, "Incorrect argument types.");
       goto ERROR_HANDLE;
   }

   if (use_array_func)
   {
      dim_size[0] = nelement = (int) PyArray_DIM(line_arr, 0);
      som_x_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      som_y_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (som_x_arr == NULL || som_y_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }


      status = MtkLSToSomXYAry(self->mapinfo, nelement,
			 (float*)PyArray_DATA(line_arr), (float*)PyArray_DATA(sample_arr),
			 (double*)PyArray_DATA(som_x_arr), (double*)PyArray_DATA(som_y_arr) );
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLSToSomXYAry Failed");
         goto ERROR_HANDLE;
      }

      Py_DECREF(line_arr);
      Py_DECREF(sample_arr);
      result = Py_BuildValue("NN",PyArray_Return(som_x_arr),
                                  PyArray_Return(som_y_arr));
      return result;
   }
   else
   {
      status = MtkLSToSomXY(self->mapinfo,line,sample,&som_x,&som_y);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLSToSomXY Failed");
         goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",som_x,som_y);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   return NULL;
}

static PyObject *
MtkMapInfo_SomXYToLS(MtkMapInfo *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int nelement;
   PyObject *osom_x, *osom_y;
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x = 0.0;
   double som_y = 0.0;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   float line;
   float sample;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"OO",&osom_x,&osom_y))
      return NULL;

   if (PyArray_Check(osom_x) && PyArray_Check(osom_y))
   {
      use_array_func = 1;
      som_x_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osom_x, NPY_DOUBLE, 1, 1);
      som_y_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osom_y, NPY_DOUBLE, 1, 1);

      if (som_x_arr == NULL || som_y_arr == NULL)
      {
	goto ERROR_HANDLE;
      }

      if ( PyArray_DIM(som_x_arr, 0) != PyArray_DIM(som_y_arr,0) )
      {
        PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	goto ERROR_HANDLE;
      }
   }
   else if (PyFloat_Check(osom_x) && PyFloat_Check(osom_y))
   {
      use_array_func = 0;
      som_x = PyFloat_AsDouble(osom_x);
      som_y = PyFloat_AsDouble(osom_y);
   }
   else
   {
      PyErr_SetString(PyExc_TypeError, "Incorrect argument types.");
      goto ERROR_HANDLE;
   }

   if (use_array_func)
   {
      dim_size[0] = nelement = (int) PyArray_DIM(som_x_arr, 0);
      line_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);
      sample_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);

      if (line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkSomXYToLSAry(self->mapinfo,nelement,
			     (double*)PyArray_DATA(som_x_arr), (double*)PyArray_DATA(som_y_arr),
			     (float*)PyArray_DATA(line_arr), (float*)PyArray_DATA(sample_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToLSAry Failed");
	 goto ERROR_HANDLE;
      }

      Py_DECREF(som_x_arr);
      Py_DECREF(som_y_arr);
      result = Py_BuildValue("NN",PyArray_Return(line_arr),
                             PyArray_Return(sample_arr));
      return result;
   }
   else
   {
      status = MtkSomXYToLS(self->mapinfo,som_x,som_y,&line,&sample);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToLS Failed");
	 goto ERROR_HANDLE;
      }

      result = Py_BuildValue("ff",line,sample);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   return NULL;
}

static PyObject *
MtkMapInfo_LatLonToLS(MtkMapInfo *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int nelement;
   PyObject *olat_dd, *olon_dd;
   PyArrayObject *lat_dd_arr = NULL;
   PyArrayObject *lon_dd_arr = NULL;
   double lat_dd = 0.0;
   double lon_dd = 0.0;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   float line = 0.0;
   float sample = 0.0;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"OO",&olat_dd,&olon_dd))
      return NULL;

   if (PyArray_Check(olat_dd) && PyArray_Check(olon_dd))
   {
      use_array_func = 1;
      lat_dd_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      olat_dd, NPY_DOUBLE, 1, 1);
      lon_dd_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      olon_dd, NPY_DOUBLE, 1, 1);

      if (lat_dd_arr == NULL || lon_dd_arr == NULL)
      {
	goto ERROR_HANDLE;
      }

      if ( PyArray_DIM(lat_dd_arr,0) != PyArray_DIM(lon_dd_arr,0) )
      {
        PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	goto ERROR_HANDLE;
      }
   }
   else if (PyFloat_Check(olat_dd) && PyFloat_Check(olon_dd))
   {
      use_array_func = 0;
      lat_dd = PyFloat_AsDouble(olat_dd);
      lon_dd = PyFloat_AsDouble(olon_dd);
   }
   else
   {
      PyErr_SetString(PyExc_TypeError, "Incorrect argument types.");
      goto ERROR_HANDLE;
   }

   if (use_array_func)
   {
      dim_size[0] = nelement = (int) PyArray_DIM(lat_dd_arr,0);
      line_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);
      sample_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);

      if (line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }


      status = MtkLatLonToLSAry(self->mapinfo,nelement,
		      (double*)PyArray_DATA(lat_dd_arr), (double*)PyArray_DATA(lon_dd_arr),
		      (float*)PyArray_DATA(line_arr),(float*)PyArray_DATA(sample_arr) );
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToLSAry Failed");
	 goto ERROR_HANDLE;
      }

      Py_DECREF(lat_dd_arr);
      Py_DECREF(lon_dd_arr);
      result = Py_BuildValue("NN",PyArray_Return(line_arr),
                             PyArray_Return(sample_arr));
      return result;
   }
   else
   {
      status = MtkLatLonToLS(self->mapinfo,lat_dd,lon_dd,&line,&sample );
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToLS Failed");
	 goto ERROR_HANDLE;
      }

      result = Py_BuildValue("ff",line,sample);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   return NULL;
}

static PyObject *
MtkMapInfo_LSToLatLon(MtkMapInfo *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int nelement;
   PyObject *oline, *osample;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   float line = 0.0;
   float sample = 0.0;
   PyArrayObject *lat_dd_arr = NULL;
   PyArrayObject *lon_dd_arr = NULL;
   double lat_dd;
   double lon_dd;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"OO",&oline,&osample))
      return NULL;

   if (PyArray_Check(oline) && PyArray_Check(osample))
   {
      use_array_func = 1;
      line_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oline, NPY_FLOAT, 1, 1);
      sample_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osample, NPY_FLOAT, 1, 1);

      if (line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      if ( PyArray_DIM(line_arr,0) !=  PyArray_DIM(sample_arr,0) )
      {
         PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	 goto ERROR_HANDLE;
      }
   }
   else if ((PyFloat_Check(oline) || PyInt_Check(oline)) &&
            (PyFloat_Check(osample) || PyInt_Check(osample)))
   {
      use_array_func = 0;
      line = (float)PyFloat_AsDouble(oline);
      sample = (float)PyFloat_AsDouble(osample);
   }
   else
   {
       PyErr_SetString(PyExc_TypeError, "Incorrect argument types.");
       goto ERROR_HANDLE;
   }

   if (use_array_func)
   {
      dim_size[0] = nelement = (int) PyArray_DIM(line_arr,0);
      lat_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      lon_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (lat_dd_arr == NULL || lon_dd_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkLSToLatLonAry(self->mapinfo,nelement,
		      (float*)PyArray_DATA(line_arr), (float*)PyArray_DATA(sample_arr),
		      (double*)PyArray_DATA(lat_dd_arr), (double*)PyArray_DATA(lon_dd_arr) );
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLSToLatLonAry Failed");
         goto ERROR_HANDLE;
      }

      Py_DECREF(line_arr);
      Py_DECREF(sample_arr);
      result = Py_BuildValue("NN",PyArray_Return(lat_dd_arr),
                             PyArray_Return(lon_dd_arr));
      return result;
   }
   else
   {
      status = MtkLSToLatLon(self->mapinfo,line,sample,&lat_dd,&lon_dd);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLSToLatLon Failed");
         goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",lat_dd,lon_dd);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   return NULL;
}

static PyObject *
MtkMapInfo_CreateLatLon(MtkMapInfo *self)
{
   PyObject *result;
   MTKt_status status;
   PyArrayObject *lat_dd_arr = NULL;
   PyArrayObject *lon_dd_arr = NULL;
   const int dim = 2;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[2];
   MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
   MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;

   status = MtkCreateLatLon(self->mapinfo, &latbuf, &lonbuf);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkCreateLatLon Failed");
      goto ERROR_HANDLE;
   }

   dim_size[0] = latbuf.nline;
   dim_size[1] = latbuf.nsample;
   
   lat_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
   lon_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

   if (lat_dd_arr == NULL || lon_dd_arr == NULL)
   {
     PyErr_SetString(PyExc_MemoryError, "Could not create NumPy.");
	 goto ERROR_HANDLE;
   }

   
   memcpy(PyArray_DATA(lat_dd_arr),latbuf.dataptr,latbuf.datasize *
          latbuf.nline * latbuf.nsample);
          
   memcpy(PyArray_DATA(lon_dd_arr),lonbuf.dataptr,lonbuf.datasize *
          lonbuf.nline * lonbuf.nsample);
   
   MtkDataBufferFree(&latbuf);
   MtkDataBufferFree(&lonbuf);
   
   result = Py_BuildValue("NN",PyArray_Return(lat_dd_arr),
                             PyArray_Return(lon_dd_arr));
   return result;
   
ERROR_HANDLE:
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   MtkDataBufferFree(&latbuf);
   MtkDataBufferFree(&lonbuf);
   return NULL;
}

static PyObject *
MtkMapInfo_getpixelcenter(MtkMapInfo *self, void *closure)
{
   Py_XINCREF(self->pixelcenter);
   return self->pixelcenter;
}

MTK_READ_ONLY_ATTR(MtkMapInfo, pixelcenter)

static PyObject *
MtkMapInfo_getpp(MtkMapInfo *self, void *closure)
{
   Py_XINCREF(self->pp);

   if (self->pp == NULL)
   {
      PyErr_Format(PyExc_StandardError, "PP NULL");
      return NULL;

   }

   return (PyObject*)self->pp;
}

MTK_READ_ONLY_ATTR(MtkMapInfo, pp)

static PyObject *
MtkMapInfo_getsom(MtkMapInfo *self, void *closure)
{
   Py_XINCREF(self->som);
   return (PyObject*)self->som;
}

MTK_READ_ONLY_ATTR(MtkMapInfo, som)

static PyObject *
MtkMapInfo_getgeo(MtkMapInfo *self, void *closure)
{
   Py_XINCREF(self->geo);
   return (PyObject*)self->geo;
}

MTK_READ_ONLY_ATTR(MtkMapInfo, geo)

static PyGetSetDef MtkMapInfo_getseters[] = {
    {"pixelcenter", (getter)MtkMapInfo_getpixelcenter, (setter)MtkMapInfo_setpixelcenter,
     "Pixel Center.", NULL},
    {"pp", (getter)MtkMapInfo_getpp, (setter)MtkMapInfo_setpp,
     "Projection Parameters.", NULL},
    {"som", (getter)MtkMapInfo_getsom, (setter)MtkMapInfo_setsom,
     "Som coordinates.", NULL},
    {"geo", (getter)MtkMapInfo_getgeo, (setter)MtkMapInfo_setgeo,
     "Geo coordinates.", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef MtkMapInfo_members[] = {
    {"path", T_INT, offsetof(MtkMapInfo, mapinfo.path), READONLY,
     "Path number"},
    {"start_block", T_INT, offsetof(MtkMapInfo, mapinfo.start_block), READONLY,
     "Start Block Number"},
    {"end_block", T_INT, offsetof(MtkMapInfo, mapinfo.end_block), READONLY,
     "End Block Number"},
    {"resolution", T_INT, offsetof(MtkMapInfo, mapinfo.resolution), READONLY,
     "Resolution"},
    {"resfactor", T_INT, offsetof(MtkMapInfo, mapinfo.resfactor), READONLY,
     "Resfactor"},
    {"nline", T_INT, offsetof(MtkMapInfo, mapinfo.nline), READONLY,
     "Number Lines"},
    {"nsample", T_INT, offsetof(MtkMapInfo, mapinfo.nsample), READONLY,
     "Number Samples"},
    {NULL}  /* Sentinel */
};

static PyMethodDef MtkMapInfo_methods[] = {
   {"ls_to_somxy", (PyCFunction)MtkMapInfo_LSToSomXY, METH_VARARGS,
    "Line and Sample To Som X and Som Y."},
   {"somxy_to_ls", (PyCFunction)MtkMapInfo_SomXYToLS, METH_VARARGS,
    "Som X and Som Y To Line and Sample."},
   {"latlon_to_ls", (PyCFunction)MtkMapInfo_LatLonToLS, METH_VARARGS,
    "Lat and Lon to Line Sample."},
   {"ls_to_latlon", (PyCFunction)MtkMapInfo_LSToLatLon, METH_VARARGS,
    "Line and Sample To Lat and Lon."},
   {"create_latlon", (PyCFunction)MtkMapInfo_CreateLatLon, METH_NOARGS,
    "Create a latitude array and a longitude array in decimal degrees."},    
   {NULL}  /* Sentinel */
};

PyTypeObject MtkMapInfoType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkMapInfo",  /*tp_name*/
    sizeof(MtkMapInfo),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkMapInfo_dealloc,/*tp_dealloc*/
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
    "MtkMapInfo objects",      /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkMapInfo_methods,        /* tp_methods */
    MtkMapInfo_members,        /* tp_members */
    MtkMapInfo_getseters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkMapInfo_init, /* tp_init */
    0,                         /* tp_alloc */
    MtkMapInfo_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkmapinfo_methods[] = {
    {NULL}  /* Sentinel */
};
