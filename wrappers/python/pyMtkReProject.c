/*===========================================================================
=                                                                           =
=                                 MtkReProject                              =
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
#include <stdlib.h>
#include "pyMtk.h"
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
#include "pyhelpers.h"

extern PyTypeObject DataPlaneType;
extern PyTypeObject MtkMapInfoType;

static void
MtkReProject_dealloc(MtkReProject* self)
{
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkReProject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkReProject *self;

   self = (MtkReProject *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
       //empty body
   }
   return (PyObject*)self;
}


static int
MtkReProject_init(MtkReProject *self, PyObject *args, PyObject *kwds)
{
   return 0;
}

static PyObject *
MtkReProject_CreateGeoGrid(MtkReProject *self, PyObject *args)
{
    PyObject *result;
    MTKt_status status;

    /* Inputs */
    double ulc_lat_dd;
    double ulc_lon_dd;
    double lrc_lat_dd;
    double lrc_lon_dd;
    double lat_cellsize_dd;
    double lon_cellsize_dd;

    /* Outputs */
    MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
    PyObject *lat_dd_arr = NULL;
    MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;
    PyObject *lon_dd_arr = NULL;

    /* Inputs */
    switch (PyTuple_Size(args))
    {
        case 6:
            if (!PyArg_ParseTuple(args,"dddddd",&ulc_lat_dd,&ulc_lon_dd,&lrc_lat_dd,&lrc_lon_dd,&lat_cellsize_dd,&lon_cellsize_dd)) {
                PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
                return NULL;
            }
            break;
        default:
            PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
            return NULL;   
    }
    
    /* Misr Toolkit Call */
    status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd, lat_cellsize_dd, lon_cellsize_dd, &latbuf, &lonbuf);
    if (status != MTK_SUCCESS)
    {
      PyErr_SetString(PyExc_StandardError, "MtkCreateGeoGrid Failed");
      return NULL;
    }
    
    /* Outputs */
    status = Mtk_DataBufferToPyArray(&latbuf, &lat_dd_arr, NPY_DOUBLE);
    MTK_ERR_PY_COND_JUMP(status);
    status = Mtk_DataBufferToPyArray(&lonbuf, &lon_dd_arr, NPY_DOUBLE);
    MTK_ERR_PY_COND_JUMP(status);        

    result = Py_BuildValue("NN",PyArray_Return((PyArrayObject *)lat_dd_arr), PyArray_Return((PyArrayObject *)lon_dd_arr));
    return result;

ERROR_HANDLE:
    Py_XDECREF(lat_dd_arr);
    Py_XDECREF(lon_dd_arr);
    MtkDataBufferFree(&latbuf);
    MtkDataBufferFree(&lonbuf);
    return NULL;
}	

static PyObject *
MtkReProject_ResampleCubicConvolution(MtkReProject *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    PyObject *arg5 = NULL;
    MTKt_DataBuffer srcdata = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer srcmask = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer lines = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer samples = MTKT_DATABUFFER_INIT;
    float a = 0.0;

    /* Outputs */
    MTKt_DataBuffer rsmpdata = MTKT_DATABUFFER_INIT;
    PyObject *rsmpdata_arr = NULL;    
    MTKt_DataBuffer rsmpmask = MTKT_DATABUFFER_INIT;
    PyObject *rsmpmask_arr = NULL;

    if (PyTuple_Size(args) == 5) 
    {
        if (!PyArg_ParseTuple(args,"O|O|O|O|O",&arg1,&arg2,&arg3,&arg4,&arg5)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( (PyArray_TYPE((PyArrayObject *)arg1) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg1) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg1, &srcdata);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_UBYTE) && (PyArray_NDIM((PyArrayObject *)arg2) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg2, &srcmask);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 2D uint8 numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg3) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg3) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg3, &lines);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 3 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg4) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg4) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg4, &samples);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 4 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if (PyFloat_Check(arg5)) {
            a = (float) PyFloat_AsDouble(arg5);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 5 is not a float type.");
            return NULL;            
        }

        /* Misr Toolkit Call */
        status =  MtkResampleCubicConvolution(&srcdata, &srcmask, &lines, &samples, a, &rsmpdata, &rsmpmask);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkResampleCubicConvolution Failed");
            return NULL;
        }

        /* Outputs */
        status = Mtk_DataBufferToPyArray(&rsmpdata, &rsmpdata_arr, NPY_FLOAT);
        MTK_ERR_PY_COND_JUMP(status);
        status = Mtk_DataBufferToPyArray(&rsmpmask, &rsmpmask_arr, NPY_UBYTE);
        MTK_ERR_PY_COND_JUMP(status);

        result = Py_BuildValue("NN",PyArray_Return((PyArrayObject *)rsmpdata_arr), PyArray_Return((PyArrayObject *)rsmpmask_arr));
        MtkDataBufferFree(&srcdata);
        MtkDataBufferFree(&srcmask);
        MtkDataBufferFree(&lines);
        MtkDataBufferFree(&samples);
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&srcdata);
    MtkDataBufferFree(&srcmask);
    MtkDataBufferFree(&lines);
    MtkDataBufferFree(&samples);
    MtkDataBufferFree(&rsmpdata);                
    MtkDataBufferFree(&rsmpmask);                    
    Py_XDECREF(rsmpdata_arr);
    Py_XDECREF(rsmpmask_arr);    
    return NULL;
}

static PyObject *
MtkReProject_ResampleNearestNeighbor(MtkReProject *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    MTKt_DataBuffer srcdata = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer lines = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer samples = MTKT_DATABUFFER_INIT;

    /* Outputs */
    MTKt_DataBuffer rsmpdata = MTKT_DATABUFFER_INIT;
    PyObject *rsmpdata_arr = NULL;    

    if (PyTuple_Size(args) == 3) 
    {
        if (!PyArg_ParseTuple(args,"OOO",&arg1,&arg2,&arg3)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( ((PyArray_TYPE((PyArrayObject *)arg1) == NPY_FLOAT32) || (PyArray_TYPE((PyArrayObject *)arg1) == NPY_UBYTE)) && 
            (PyArray_NDIM((PyArrayObject *)arg1) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg1, &srcdata);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 2D float (32-bit) or uint8 numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg2) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg2, &lines);
            MTK_ERR_PY_COND_JUMP(status);
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg3) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg3) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg3, &samples);
            MTK_ERR_PY_COND_JUMP(status);
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 3 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }  

        /* Misr Toolkit Call */
        status =  MtkResampleNearestNeighbor(srcdata, lines, samples, &rsmpdata);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkResampleNearestNeighbor Failed");
            return NULL;
        }

        /* Outputs */
        if (PyArray_ISFLOAT((PyArrayObject *)arg1)) {
            status = Mtk_DataBufferToPyArray(&rsmpdata, &rsmpdata_arr, NPY_FLOAT);
            MTK_ERR_PY_COND_JUMP(status);
        }  else if (PyArray_ISINTEGER((PyArrayObject *)arg1)) {
            status = Mtk_DataBufferToPyArray(&rsmpdata, &rsmpdata_arr, NPY_UBYTE);
            MTK_ERR_PY_COND_JUMP(status);
        }        
        
        result = Py_BuildValue("N",PyArray_Return((PyArrayObject *)rsmpdata_arr));
        MtkDataBufferFree(&srcdata);
        MtkDataBufferFree(&lines);
        MtkDataBufferFree(&samples);
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&srcdata);
    MtkDataBufferFree(&lines);
    MtkDataBufferFree(&samples);      
    MtkDataBufferFree(&rsmpdata);    
    Py_XDECREF(rsmpdata_arr);
    return NULL;
}

static PyObject *
MtkReProject_TransformCoordinates(MtkReProject *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;    
    MTKt_MapInfo *mapinfo = NULL;
    MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;

    /* Outputs */
    MTKt_DataBuffer lines = MTKT_DATABUFFER_INIT;
    PyObject *lines_arr = NULL;    
    MTKt_DataBuffer samples = MTKT_DATABUFFER_INIT;
    PyObject *samples_arr = NULL;  

    if (PyTuple_Size(args) == 3) 
    {
        if (!PyArg_ParseTuple(args,"OOO",&arg1,&arg2,&arg3)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }
                        
        /* Inputs */
        if (PyObject_TypeCheck(arg1,&MtkMapInfoType)) {
            mapinfo = &((MtkMapInfo*)arg1)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 1 must be a mapinfo");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_FLOAT64) && (PyArray_NDIM((PyArrayObject *)arg2) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg2, &latbuf);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 2D double numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg3) == NPY_FLOAT64) && (PyArray_NDIM((PyArrayObject *)arg3) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg3, &lonbuf);    
            MTK_ERR_PY_COND_JUMP(status);
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 3 must be a 2D double numpy array.");
            return NULL;           
        }
              
        /* Misr Toolkit Call */
        status =  MtkTransformCoordinates(*mapinfo, latbuf, lonbuf, &lines, &samples);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkTransformCoordinates Failed");
            return NULL;
        }
        
        /* Outputs */
        status = Mtk_DataBufferToPyArray(&lines, &lines_arr, NPY_FLOAT);
        MTK_ERR_PY_COND_JUMP(status);        
        status = Mtk_DataBufferToPyArray(&samples, &samples_arr, NPY_FLOAT);
        MTK_ERR_PY_COND_JUMP(status);
        
        result = Py_BuildValue("NN",PyArray_Return((PyArrayObject *)lines_arr), PyArray_Return((PyArrayObject *)samples_arr));           
        MtkDataBufferFree(&latbuf);
        MtkDataBufferFree(&lonbuf); 
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&latbuf);
    MtkDataBufferFree(&lonbuf); 
    MtkDataBufferFree(&lines);
    MtkDataBufferFree(&samples);    
    Py_XDECREF(lines_arr);
    Py_XDECREF(samples_arr);    
    return NULL;
}		

static PyGetSetDef MtkReProject_getseters[] = {
   {NULL}  /* Sentinel */
};

PyMethodDef mtkreproject_methods[] = {
    {"create_geogrid", (PyCFunction)MtkReProject_CreateGeoGrid, METH_VARARGS,
     "Creates a regularly spaced geographic 2-D grid."},    
    {"resample_cubic_convolution", (PyCFunction)MtkReProject_ResampleCubicConvolution, METH_VARARGS,
     "Resample source data at the given coordinates using interpolation by cubic convolution."},
    {"resample_nearest_neighbor", (PyCFunction)MtkReProject_ResampleNearestNeighbor, METH_VARARGS,
     "Perform nearest neighbor resampling."},
     {"transform_coordinates", (PyCFunction)MtkReProject_TransformCoordinates, METH_VARARGS,
      "Transforms latitude/longitude coordinates into line/sample coordinates."},                      
    {NULL}  /* Sentinel */
};

PyTypeObject MtkReProjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkReProject",      /*tp_name*/
    sizeof(MtkReProject),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkReProject_dealloc,/*tp_dealloc*/
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
    "MtkReProject objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    mtkreproject_methods,           /* tp_methods */
    0,                         /* tp_members */
    MtkReProject_getseters,         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkReProject_init,    /* tp_init */
    0,                         /* tp_alloc */
    MtkReProject_new,  /*PyType_GenericNew()*/              /* tp_new */
};

