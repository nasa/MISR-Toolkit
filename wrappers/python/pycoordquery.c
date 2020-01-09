/*===========================================================================
=                                                                           =
=                              PyCoordQuery                                 =
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

extern PyTypeObject MtkProjParamType;
extern PyTypeObject MtkBlockCornersType;
int MtkBlockCorners_init(MtkBlockCorners *self, PyObject *args, PyObject *kwds);

static PyObject *
BlsToLatLon(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;              /* Path */
   int resolution_meters; /* Resolution */
   int nelement;          /* Number of elements */
   PyObject *oblock, *oline, *osample; /* Temp */
   PyArrayObject *block_arr = NULL;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   int block = 0;
   float line = 0.0;
   float sample = 0.0;
   PyArrayObject *lat_dd_arr = NULL; /* Latitude Decimal Degrees */
   PyArrayObject *lon_dd_arr = NULL; /* Longitude Decimal Degrees */
   double lat_dd;
   double lon_dd;
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iiOOO",&path,&resolution_meters,
                         &oblock,&oline,&osample))
      return NULL;

   if (PyArray_Check(oblock) && PyArray_Check(oline) &&
       PyArray_Check(osample))
   {
      use_array_func = 1;
      block_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oblock, NPY_INT, 1, 1);
      line_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oline, NPY_FLOAT, 1, 1);
      sample_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osample, NPY_FLOAT, 1, 1);

      if (block_arr == NULL || line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      if ( PyArray_DIM(block_arr,0) != PyArray_DIM(line_arr,0) ||
          PyArray_DIM(block_arr,0) !=  PyArray_DIM(sample_arr,0) )
      {
         PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	 goto ERROR_HANDLE;
      }
   }
   else if (PyInt_Check(oblock) && (PyFloat_Check(oline) || PyInt_Check(oline)) &&
            (PyFloat_Check(osample) || PyInt_Check(osample)))
   {
      use_array_func = 0;
      block = (int)PyInt_AsLong(oblock);
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
      dim_size[0] = nelement = (int) PyArray_DIM(block_arr,0);
      lat_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      lon_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (lat_dd_arr == NULL || lon_dd_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkBlsToLatLonAry(path,resolution_meters,nelement,
                              (int*)PyArray_DATA(block_arr),(float*)PyArray_DATA(line_arr),
                              (float*)PyArray_DATA(sample_arr),(double*)PyArray_DATA(lat_dd_arr),
                              (double*)PyArray_DATA(lon_dd_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkBlsToLatLonAry Failed");
         goto ERROR_HANDLE;
      }

      Py_DECREF(block_arr);
      Py_DECREF(line_arr);
      Py_DECREF(sample_arr);
      result = Py_BuildValue("NN",PyArray_Return(lat_dd_arr),
                             PyArray_Return(lon_dd_arr));
      return result;
   }
   else
   {
      status = MtkBlsToLatLon(path,resolution_meters,block,line,sample,
                              &lat_dd,&lon_dd);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkBlsToLatLon Failed");
         goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",lat_dd,lon_dd);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(block_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   return NULL;
}

static PyObject *
BlsToSomXY(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;              /* Path */
   int resolution_meters; /* Resolution Meters */
   int nelement;          /* Number of elements */
   PyObject *oblock, *oline, *osample; /* Temp */
   PyArrayObject *block_arr = NULL;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   int block = 0;             /* Block */
   float line = 0;            /* Line */
   float sample = 0;          /* Sample */
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x;         /* SOM X */
   double som_y;         /* SOM Y */ 
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iiOOO",&path,&resolution_meters,
                         &oblock,&oline,&osample))
      return NULL;

   if (PyArray_Check(oblock) && PyArray_Check(oline) &&
       PyArray_Check(osample))
   {
      use_array_func = 1;
      block_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oblock, NPY_INT, 1, 1);
      line_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      oline, NPY_FLOAT, 1, 1);
      sample_arr = (PyArrayObject*)PyArray_ContiguousFromObject(
                      osample, NPY_FLOAT, 1, 1);

      if (block_arr == NULL || line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      if (PyArray_DIM(block_arr,0) != PyArray_DIM(line_arr,0) ||
          PyArray_DIM(block_arr,0) !=  PyArray_DIM(sample_arr,0))
      {
         PyErr_SetString(PyExc_ValueError, "Array dimensions not equal.");
	 goto ERROR_HANDLE;
      }
   }
   else if (PyInt_Check(oblock) && (PyFloat_Check(oline) || PyInt_Check(oline)) &&
            (PyFloat_Check(osample) || PyInt_Check(osample)))
   {
      use_array_func = 0;
      block = (int)PyInt_AsLong(oblock);
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
      dim_size[0] = nelement = (int) PyArray_DIM(block_arr,0);
      som_x_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      som_y_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (som_x_arr == NULL || som_y_arr == NULL)
      {
	goto ERROR_HANDLE;
      }

      status = MtkBlsToSomXYAry(path,resolution_meters,nelement,
                              (int*)PyArray_DATA(block_arr),(float*)PyArray_DATA(line_arr),
                              (float*)PyArray_DATA(sample_arr),
                              (double*)PyArray_DATA(som_x_arr),
                              (double*)PyArray_DATA(som_y_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkBlsToSomXYAry Failed");
         goto ERROR_HANDLE;
      }

      Py_DECREF(block_arr);
      Py_DECREF(line_arr);
      Py_DECREF(sample_arr);
      result = Py_BuildValue("NN",PyArray_Return(som_x_arr),
                             PyArray_Return(som_y_arr));
      return result;
   }
   else
   {
      status = MtkBlsToSomXY(path,resolution_meters,block,line,sample,
                              &som_x,&som_y);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkBlsToSomXY Failed");
         goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",som_x,som_y);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(block_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   return NULL;
}

static PyObject *
LatLonToBls(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;              /* Path */
   int resolution_meters; /* Resolution */
   int nelement;          /* Number of elements */
   PyObject *olat_dd, *olon_dd;
   PyArrayObject *lat_dd_arr = NULL;
   PyArrayObject *lon_dd_arr = NULL;
   double lat_dd = 0.0;  /* Latitude */
   double lon_dd = 0.0;  /* Longitude */
   PyArrayObject *block_arr = NULL;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   int block = 0;             /* Block */
   float line = 0.0;          /* Line */
   float sample = 0.0;        /* Sample */
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iiOO",&path,&resolution_meters,&olat_dd,
                         &olon_dd))
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

      if (PyArray_DIM(lat_dd_arr,0) != PyArray_DIM(lon_dd_arr,0))
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
      block_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_INT);
      line_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);
      sample_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);

      if (block_arr == NULL || line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkLatLonToBlsAry(path,resolution_meters,nelement,
			         (double*)PyArray_DATA(lat_dd_arr),
			         (double*)PyArray_DATA(lon_dd_arr),
			         (int*)PyArray_DATA(block_arr),
			         (float*)PyArray_DATA(line_arr),
			         (float*)PyArray_DATA(sample_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToBlsAry Failed");
	 goto ERROR_HANDLE;
      }

      Py_DECREF(lat_dd_arr);
      Py_DECREF(lon_dd_arr);
      result = Py_BuildValue("NNN",PyArray_Return(block_arr),
                             PyArray_Return(line_arr),
                             PyArray_Return(sample_arr));
      return result;
   }
   else
   {
      status = MtkLatLonToBls(path,resolution_meters,lat_dd,lon_dd,
                              &block,&line,&sample);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToBls Failed");
	 goto ERROR_HANDLE;
      }

      result = Py_BuildValue("iff",block,line,sample);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   Py_XDECREF(block_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   return NULL;
}

static PyObject *
LatLonToSomXY(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;             /* Path */
   int nelement;         /* Number of elements */
   PyObject *olat_dd, *olon_dd;
   PyArrayObject *lat_dd_arr = NULL;
   PyArrayObject *lon_dd_arr = NULL;
   double lat_dd = 0.0;  /* Latitude */
   double lon_dd = 0.0;  /* Longitude */
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x;        /* SOM X */
   double som_y;        /* SOM Y */
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iOO",&path,&olat_dd,&olon_dd))
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

      if (PyArray_DIM(lat_dd_arr,0) != PyArray_DIM(lon_dd_arr,0))
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
      som_x_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      som_y_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (som_x_arr == NULL || som_y_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkLatLonToSomXYAry(path,nelement,(double*)PyArray_DATA(lat_dd_arr),
			           (double*)PyArray_DATA(lon_dd_arr),
			           (double*)PyArray_DATA(som_x_arr),
			           (double*)PyArray_DATA(som_y_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToSomXYAry Failed");
	 goto ERROR_HANDLE;
      }

      Py_DECREF(lat_dd_arr);
      Py_DECREF(lon_dd_arr);
      result = Py_BuildValue("NN",PyArray_Return(som_x_arr),
                             PyArray_Return(som_y_arr));
      return result;
   }
   else
   {
      status = MtkLatLonToSomXY(path,lat_dd,lon_dd,
                              &som_x,&som_y);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkLatLonToSomXY Failed");
	 goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",som_x,som_y);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   return NULL;
}

static PyObject *
PathToProjParam(PyObject *self, PyObject *args)
{
  /*PyObject *result;*/
   MTKt_status status;
   int path;
   int resolution_meters;
   MtkProjParam *proj;

   if (!PyArg_ParseTuple(args,"ii",&path,&resolution_meters))
      return NULL;

   proj = PyObject_New(MtkProjParam, &MtkProjParamType);

   status = MtkPathToProjParam(path,resolution_meters,&proj->pp);
   if (status != MTK_SUCCESS)
   {
      Py_XDECREF(proj);
      PyErr_SetString(PyExc_StandardError, "MtkPathToProjParam Failed");
      return NULL;
   }

   return (PyObject*)proj;
}

static PyObject *
SomXYToBls(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;              /* Path */
   int resolution_meters; /* Resolution */
   int nelement;          /* Number of elements */
   PyObject *osom_x, *osom_y;
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x = 0.0;   /* SOM X */
   double som_y = 0.0;   /* SOM Y */
   PyArrayObject *block_arr = NULL;
   PyArrayObject *line_arr = NULL;
   PyArrayObject *sample_arr = NULL;
   int block;            /* Block */
   float line;           /* Line */
   float sample;         /* Sample */
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iiOO",&path,&resolution_meters,&osom_x,
                         &osom_y))
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

      if (PyArray_DIM(som_x_arr,0) != PyArray_DIM(som_y_arr,0))
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
      dim_size[0] = nelement = (int) PyArray_DIM(som_x_arr,0);
      block_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_INT);
      line_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);
      sample_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_FLOAT);

      if (block_arr == NULL || line_arr == NULL || sample_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkSomXYToBlsAry(path,resolution_meters,nelement,
			        (double*)PyArray_DATA(som_x_arr),
			        (double*)PyArray_DATA(som_y_arr),
			        (int*)PyArray_DATA(block_arr),
			        (float*)PyArray_DATA(line_arr),
			        (float*)PyArray_DATA(sample_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToBlsAry Failed");
	 goto ERROR_HANDLE;
      }

      Py_DECREF(som_x_arr);
      Py_DECREF(som_y_arr);
      result = Py_BuildValue("NNN",PyArray_Return(block_arr),
                             PyArray_Return(line_arr),
                             PyArray_Return(sample_arr));
      return result;
   }
   else
   {
      status = MtkSomXYToBls(path,resolution_meters,som_x,som_y,
                              &block,&line,&sample);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToBls Failed");
	 goto ERROR_HANDLE;
      }

      result = Py_BuildValue("iff",block,line,sample);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   Py_XDECREF(block_arr);
   Py_XDECREF(line_arr);
   Py_XDECREF(sample_arr);
   return NULL;
}

static PyObject *
SomXYToLatLon(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;            /* Path */
   int nelement;        /* Number of elements */
   PyObject *osom_x, *osom_y; /* Temp */
   PyArrayObject *som_x_arr = NULL;
   PyArrayObject *som_y_arr = NULL;
   double som_x = 0.0; /* SOM X */
   double som_y = 0.0; /* SOM Y */
   PyArrayObject *lat_dd_arr = NULL; /* Latitude Decimal Degrees */
   PyArrayObject *lon_dd_arr = NULL; /* Longitude Decimal Degrees */
   double lat_dd;      /* Latitude */
   double lon_dd;      /* Longitude */
   const int dim = 1;
   /* Using npy_intp (defined in arrayobject.h) instead of int
      so pointer size is correct on 64-bit systems. */
   npy_intp dim_size[1];
   int use_array_func = 0;

   if (!PyArg_ParseTuple(args,"iOO",&path,&osom_x,&osom_y))
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

      if (PyArray_DIM(som_x_arr,0) != PyArray_DIM(som_y_arr,0))
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
      dim_size[0] = nelement = (int) PyArray_DIM(som_x_arr,0);
      lat_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);
      lon_dd_arr = (PyArrayObject*)PyArray_SimpleNew(dim,dim_size,NPY_DOUBLE);

      if (lat_dd_arr == NULL || lon_dd_arr == NULL)
      {
	 goto ERROR_HANDLE;
      }

      status = MtkSomXYToLatLonAry(path,nelement,(double*)PyArray_DATA(som_x_arr),
                                   (double*)PyArray_DATA(som_y_arr),
                                   (double*)PyArray_DATA(lat_dd_arr),
                                   (double*)PyArray_DATA(lon_dd_arr));
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToLatLonAry Failed");
         goto ERROR_HANDLE;
      }

      Py_DECREF(som_x_arr);
      Py_DECREF(som_y_arr);
      result = Py_BuildValue("NN",PyArray_Return(lat_dd_arr),
                             PyArray_Return(lon_dd_arr));
      return result;
   }
   else
   {
      status = MtkSomXYToLatLon(path,som_x,som_y,&lat_dd,&lon_dd);
      if (status != MTK_SUCCESS)
      {
         PyErr_SetString(PyExc_StandardError, "MtkSomXYToLatLon Failed");
         goto ERROR_HANDLE;
      }

      result = Py_BuildValue("dd",lat_dd,lon_dd);
      return result;
   }

ERROR_HANDLE:
   Py_XDECREF(som_x_arr);
   Py_XDECREF(som_y_arr);
   Py_XDECREF(lat_dd_arr);
   Py_XDECREF(lon_dd_arr);
   return NULL;
}

static PyObject *
PathBlockRangeToBlockCorners(PyObject *self, PyObject *args)
{
  /*PyObject *result;*/
   MTKt_status status;
   int path;            /* Path */
   int start_block;     /* Start Block */
   int end_block;       /* End Block */
   MTKt_BlockCorners block_corners = MTKT_BLOCKCORNERS_INIT;
   MtkBlockCorners *bc;
   int i;

   if (!PyArg_ParseTuple(args,"iii",&path,&start_block,&end_block))
     return NULL;

   status = MtkPathBlockRangeToBlockCorners(path, start_block,end_block,
					    &block_corners);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkPathBlockRangeToBlockCorners Failed");
      return NULL;
   }

   bc = (MtkBlockCorners*)PyObject_New(MtkBlockCorners, &MtkBlockCornersType);
   MtkBlockCorners_init(bc,NULL,NULL);

   bc->path = block_corners.path;
   bc->start_block = block_corners.start_block;
   bc->end_block = block_corners.end_block;

   for (i = 0; i < NBLOCK + 1; ++i)
   {
      bc->gb[i]->block_number = block_corners.block[i].block_number;
      bc->gb[i]->ulc->gc = block_corners.block[i].ulc;
      bc->gb[i]->urc->gc = block_corners.block[i].urc;
      bc->gb[i]->ctr->gc = block_corners.block[i].ctr;
      bc->gb[i]->lrc->gc = block_corners.block[i].lrc;
      bc->gb[i]->llc->gc = block_corners.block[i].llc;
   }

   return (PyObject*)bc;
}

PyMethodDef coordquery_methods[] = {
   {"bls_to_latlon", (PyCFunction)BlsToLatLon, METH_VARARGS,
    "Convert from Block, Line, Sample, to Latitude and Longitude in Decimal Degrees."},
   {"bls_to_somxy", (PyCFunction)BlsToSomXY, METH_VARARGS,
    "Convert from Block, Line, Sample, to SOM Coordinates."},
   {"latlon_to_bls", (PyCFunction)LatLonToBls, METH_VARARGS,
    "Convert decimal degrees latitude and longitude to block, line, sample."},

   {"latlon_to_somxy", (PyCFunction)LatLonToSomXY, METH_VARARGS,
    "Convert decimal degrees latitude and longitude to SOM X, SOM Y."},
   {"path_to_projparam", (PyCFunction)PathToProjParam, METH_VARARGS,
    "Get projection parameters."},
   {"somxy_to_bls", (PyCFunction)SomXYToBls, METH_VARARGS,
    "Convert SOM X, SOM Y to block, line, sample."},
   {"somxy_to_latlon", (PyCFunction)SomXYToLatLon, METH_VARARGS,
    "Convert SOM X, SOM Y to decimal degrees latitude and longitude."},
   {"path_block_range_to_block_corners", (PyCFunction)PathBlockRangeToBlockCorners, METH_VARARGS,
    "Compute block corner coordinates in decimal degrees of latitude and longitude for a given path and block range."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

