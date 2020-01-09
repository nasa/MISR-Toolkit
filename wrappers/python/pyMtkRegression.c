/*===========================================================================
=                                                                           =
=                                 MtkRegression                              =
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
extern int MtkMapInfo_init(MtkMapInfo *self, PyObject *args, PyObject *kwds);
extern int MtkMapInfo_copy(MtkMapInfo *self, MTKt_MapInfo mapinfo);
extern PyTypeObject RegCoeffType;
int RegCoeff_init(RegCoeff *self, PyObject *args, PyObject *kwds);

static void
MtkRegression_dealloc(MtkRegression* self)
{
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkRegression_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkRegression *self;

   self = (MtkRegression *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
       //empty body
   }
   return (PyObject*)self;
}


static int
MtkRegression_init(MtkRegression *self, PyObject *args, PyObject *kwds)
{
   return 0;
}

static PyObject *
MtkRegression_Downsample(MtkRegression *self, PyObject *args)
{
    PyObject *result;
    MTKt_status status;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    MTKt_DataBuffer srcdata = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer srcmask = MTKT_DATABUFFER_INIT;
    int sizefactor = 0;

    /* Outputs */
    MTKt_DataBuffer rsmpdata = MTKT_DATABUFFER_INIT;
    PyObject *rsmpdata_arr = NULL;    
    MTKt_DataBuffer rsmpmask = MTKT_DATABUFFER_INIT;
    PyObject *rsmpmask_arr = NULL;


    /* Inputs */
    if (PyTuple_Size(args) == 3) 
    {
        if (!PyArg_ParseTuple(args,"O|O|O",&arg1,&arg2,&arg3)) {
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
        if (PyInt_Check(arg3)) {
            sizefactor = (int) PyInt_AsLong(arg3);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 3 is not an integer type.");
            return NULL;            
        }

        /* Misr Toolkit Call */
        status =  MtkDownsample(&srcdata, &srcmask, sizefactor, &rsmpdata, &rsmpmask);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkDownsample Failed");
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
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&srcdata);
    MtkDataBufferFree(&srcmask);
    MtkDataBufferFree(&rsmpdata);                
    MtkDataBufferFree(&rsmpmask);                    
    Py_XDECREF(rsmpdata_arr);
    Py_XDECREF(rsmpmask_arr);    
    return NULL;
}

static PyObject *
MtkRegression_LinearRegressionCalc(MtkRegression *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;
    npy_intp dim_size[1];

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    PyArrayObject* x_arr = NULL;
    PyArrayObject* y_arr = NULL;
    PyArrayObject* ysig_arr = NULL;
    int numElements;

    /* Outputs */
    PyArrayObject* a_arr = NULL;
    PyArrayObject* b_arr = NULL;
    PyArrayObject* corr_arr = NULL;

    /* Inputs */
    if (PyTuple_Size(args) == 3) 
    {
        if (!PyArg_ParseTuple(args,"OOO",&arg1,&arg2,&arg3)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( (PyArray_TYPE((PyArrayObject *)arg1) == NPY_FLOAT64) && (PyArray_NDIM((PyArrayObject *)arg1) == 1) )  {
            x_arr = (PyArrayObject*)PyArray_FROMANY(arg1, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
            if (!x_arr) {
                PyErr_SetString(PyExc_StandardError, "Problem converting argument 1 to PyArray.");
                return NULL;            
            }
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 1D double (64-bit) numpy array.");
            return NULL;    
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_FLOAT64) && (PyArray_NDIM((PyArrayObject *)arg2) == 1) )  {
            y_arr = (PyArrayObject*)PyArray_FROMANY(arg2, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
            if (!y_arr) {
                PyErr_SetString(PyExc_StandardError, "Problem converting argument 2 to PyArray.");
                return NULL;            
            }
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 1D double (64-bit) numpy array.");
            return NULL;    
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg3) == NPY_FLOAT64) && (PyArray_NDIM((PyArrayObject *)arg3) == 1) )  {
            ysig_arr = (PyArrayObject*)PyArray_FROMANY(arg3, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
            if (!ysig_arr) {
                PyErr_SetString(PyExc_StandardError, "Problem converting argument 3 to PyArray.");
                return NULL;            
            }
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 3 must be a 1D double (64-bit) numpy array.");
            return NULL;    
        }
        
        /* Outputs */
        numElements = (int) PyArray_DIM(x_arr,0);
        dim_size[0] = numElements;
        a_arr = (PyArrayObject*)PyArray_SimpleNew(1,dim_size,NPY_DOUBLE);
        b_arr = (PyArrayObject*)PyArray_SimpleNew(1,dim_size,NPY_DOUBLE);
        corr_arr = (PyArrayObject*)PyArray_SimpleNew(1,dim_size,NPY_DOUBLE);    

        /* Misr Toolkit Call */
        status =  MtkLinearRegressionCalc(numElements, (double*)PyArray_DATA(x_arr), (double*)PyArray_DATA(y_arr), (double*)PyArray_DATA(ysig_arr), 
                                        (double*)PyArray_DATA(a_arr), (double*)PyArray_DATA(b_arr), (double*)PyArray_DATA(corr_arr) );
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkLinearRegressionCalc Failed");
            goto ERROR_HANDLE;
        }
        
        result = Py_BuildValue("NNN",PyArray_Return(a_arr), PyArray_Return(b_arr), PyArray_Return(corr_arr) );
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:             
    Py_XDECREF(a_arr);
    Py_XDECREF(b_arr);   
    Py_XDECREF(corr_arr);   
    return NULL;
}

static PyObject *
MtkRegression_SmoothData(MtkRegression *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    MTKt_DataBuffer srcdata = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer srcmask = MTKT_DATABUFFER_INIT;
    int line_width = 0;
    int sample_width = 0;

    /* Outputs */
    MTKt_DataBuffer rsmpdata = MTKT_DATABUFFER_INIT;
    PyObject *rsmpdata_arr = NULL;    

    if (PyTuple_Size(args) == 4) 
    {
        if (!PyArg_ParseTuple(args,"OOOO",&arg1,&arg2,&arg3,&arg4)) {
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
        if (PyInt_Check(arg3)) {
            line_width = (int) PyInt_AsLong(arg3);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 3 is not an integer type.");
            return NULL;            
        }
        if (PyInt_Check(arg4)) {
            sample_width = (int) PyInt_AsLong(arg4);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 4 is not an integer type.");
            return NULL;            
        } 

        /* Misr Toolkit Call */
        status =  MtkSmoothData(&srcdata, &srcmask, line_width, sample_width, &rsmpdata);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkResampleCubicConvolution Failed");
            return NULL;
        }

        /* Outputs */
        status = Mtk_DataBufferToPyArray(&rsmpdata, &rsmpdata_arr, NPY_FLOAT);
        MTK_ERR_PY_COND_JUMP(status);


        result = Py_BuildValue("N",PyArray_Return((PyArrayObject *)rsmpdata_arr));
        MtkDataBufferFree(&srcdata);
        MtkDataBufferFree(&srcmask);
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&srcdata);
    MtkDataBufferFree(&srcmask);
    MtkDataBufferFree(&rsmpdata); 
    Py_XDECREF(rsmpdata_arr);
    return NULL;
}			


static PyObject *
MtkRegression_UpsampleMask(MtkRegression *self, PyObject *args)
{
    PyObject *result;
    MTKt_status status;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    MTKt_DataBuffer srcmask = MTKT_DATABUFFER_INIT;
    int sizefactor = 0;

    /* Outputs */
    MTKt_DataBuffer rsmpmask = MTKT_DATABUFFER_INIT;
    PyObject *rsmpmask_arr = NULL;


    /* Inputs */
    if (PyTuple_Size(args) == 2) 
    {
        if (!PyArg_ParseTuple(args,"OO",&arg1,&arg2)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( (PyArray_TYPE((PyArrayObject *)arg1) == NPY_UBYTE) && (PyArray_NDIM((PyArrayObject *)arg1) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg1, &srcmask);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 2D uint8 numpy array.");
            return NULL;            
        }
        if (PyInt_Check(arg2)) {
            sizefactor = (int) PyInt_AsLong(arg2);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 2 is not an integer type.");
            return NULL;            
        }

        /* Misr Toolkit Call */
        status =  MtkUpsampleMask(&srcmask, sizefactor, &rsmpmask);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkDownsample Failed");
            return NULL;
        }

        /* Outputs */
        status = Mtk_DataBufferToPyArray(&rsmpmask, &rsmpmask_arr, NPY_UBYTE);
        MTK_ERR_PY_COND_JUMP(status);

        result = Py_BuildValue("N", PyArray_Return((PyArrayObject *)rsmpmask_arr));
        MtkDataBufferFree(&srcmask);
        return result;
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&srcmask);
    MtkDataBufferFree(&rsmpmask);                    
    Py_XDECREF(rsmpmask_arr);    
    return NULL;
}

static PyObject *
MtkRegression_CoeffCalc(MtkRegression *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    PyObject *arg5 = NULL;
    PyObject *arg6 = NULL;
    PyObject *arg7 = NULL;                
    MTKt_DataBuffer data1 = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer mask1 = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer data2 = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer sigma2 = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer mask2 = MTKT_DATABUFFER_INIT;
    MTKt_MapInfo *mapinfo = NULL;    
    int size_factor = 0;

    /* Outputs */
	MTKt_RegressionCoeff regr_coeff = MTKT_REGRESSION_COEFF_INIT;
    RegCoeff *py_regr_coeff = NULL;
    MTKt_MapInfo regr_map_info = MTKT_MAPINFO_INIT;
    MtkMapInfo *py_regr_map_info = NULL;         

    if (PyTuple_Size(args) == 7) 
    {
        if (!PyArg_ParseTuple(args,"OOOOOOO",&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( (PyArray_TYPE((PyArrayObject *)arg1) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg1) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg1, &data1);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_UBYTE) && (PyArray_NDIM((PyArrayObject *)arg2) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg2, &mask1);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 2D uint8 numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg3) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg3) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg3, &data2);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 3 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg4) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg4) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg4, &sigma2);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 4 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }        
        if ( (PyArray_TYPE((PyArrayObject *)arg5) == NPY_UBYTE) && (PyArray_NDIM((PyArrayObject *)arg5) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg5, &mask2);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 5 must be a 2D uint8 numpy array.");
            return NULL;            
        }
        if (PyInt_Check(arg6)) {
             size_factor = (int) PyInt_AsLong(arg6);
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 6 is not an integer type.");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg7,&MtkMapInfoType)) {
            mapinfo = &((MtkMapInfo*)arg7)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 7 must be a mapinfo");
            return NULL;            
        }
        
        /* Misr Toolkit Call */
        py_regr_coeff = PyObject_New(RegCoeff, &RegCoeffType);
        RegCoeff_init(py_regr_coeff,NULL,NULL);
        
        status =  MtkRegressionCoeffCalc(&data1, &mask1, &data2, &sigma2, &mask2, mapinfo, size_factor, &regr_coeff, &regr_map_info);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkRegressionCoeffCalc Failed");
            return NULL;
        }        

        /* Outputs */
        status = Mtk_MtkRegCoeffToPy(&regr_coeff, &py_regr_coeff);
        MTK_ERR_PY_COND_JUMP(status);
        py_regr_map_info = (MtkMapInfo*)PyObject_New(MtkMapInfo, &MtkMapInfoType);
        MtkMapInfo_init(py_regr_map_info,NULL,NULL);
        MtkMapInfo_copy(py_regr_map_info,regr_map_info);
        result = Py_BuildValue("NN",py_regr_coeff, py_regr_map_info);
        
        MtkDataBufferFree(&data1);
        MtkDataBufferFree(&mask1);
        MtkDataBufferFree(&data2);
        MtkDataBufferFree(&sigma2);        
        MtkDataBufferFree(&mask2);        
        return result;        
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&data1);
    MtkDataBufferFree(&mask1);
    MtkDataBufferFree(&data2);
    MtkDataBufferFree(&sigma2);        
    MtkDataBufferFree(&mask2);
    Py_XDECREF(py_regr_coeff);
    Py_XDECREF(py_regr_map_info);    
    return NULL;
}	

static PyObject *
MtkRegression_ApplyRegression(MtkRegression *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
    PyObject *arg4 = NULL;
    PyObject *arg5 = NULL;
    MTKt_DataBuffer data1 = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer mask1 = MTKT_DATABUFFER_INIT;
//    MTKt_MapInfo map_info = MTKT_MAPINFO_INIT;    
    MTKt_MapInfo *map_info = NULL;    
	MTKt_RegressionCoeff regr_coeff = MTKT_REGRESSION_COEFF_INIT;
    RegCoeff *py_regr_coeff = NULL;
    MTKt_MapInfo *regr_map_info = NULL;

    /* Outputs */
    MTKt_DataBuffer regrdata = MTKT_DATABUFFER_INIT;
    PyObject *regrdata_arr = NULL;    
    MTKt_DataBuffer regrmask = MTKT_DATABUFFER_INIT;
    PyObject *regrmask_arr = NULL;


    if (PyTuple_Size(args) == 5) 
    {
        if (!PyArg_ParseTuple(args,"OOOOO",&arg1,&arg2,&arg3,&arg4,&arg5)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if ( (PyArray_TYPE((PyArrayObject *)arg1) == NPY_FLOAT32) && (PyArray_NDIM((PyArrayObject *)arg1) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg1, &data1);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 1 must be a 2D float (32-bit) numpy array.");
            return NULL;            
        }
        if ( (PyArray_TYPE((PyArrayObject *)arg2) == NPY_UBYTE) && (PyArray_NDIM((PyArrayObject *)arg2) == 2) )  {
            status = Mtk_PyArrayToDataBuffer(&arg2, &mask1);
            MTK_ERR_PY_COND_JUMP(status);  
        } else {
            PyErr_SetString(PyExc_StandardError, "Argument 2 must be a 2D uint8 numpy array.");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg3,&MtkMapInfoType)) {
            map_info = &((MtkMapInfo*)arg3)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 3 must be a mapinfo");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg4,&RegCoeffType)) {
            py_regr_coeff = (RegCoeff*)arg4;
            status = Mtk_PyRegCoeffToMtk(&py_regr_coeff, &regr_coeff);
            MTK_ERR_PY_COND_JUMP(status);                      
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 4 must be a Reg Coeff type");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg5,&MtkMapInfoType)) {
            regr_map_info = &((MtkMapInfo*)arg5)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 5 must be a mapinfo");
            return NULL;            
        }
        
        /* Misr Toolkit Call */
        
        status =  MtkApplyRegression(&data1, &mask1, map_info, &regr_coeff, regr_map_info, &regrdata, &regrmask);
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkApplyRegression Failed");
            return NULL;
        }        

        /* Outputs */
        status = Mtk_DataBufferToPyArray(&regrdata, &regrdata_arr, NPY_FLOAT);
        MTK_ERR_PY_COND_JUMP(status);
        status = Mtk_DataBufferToPyArray(&regrmask, &regrmask_arr, NPY_UBYTE);
        MTK_ERR_PY_COND_JUMP(status);

        result = Py_BuildValue("NN",PyArray_Return((PyArrayObject *)regrdata_arr), PyArray_Return((PyArrayObject *)regrmask_arr));
        MtkDataBufferFree(&data1);
        MtkDataBufferFree(&mask1);
        return result;        
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    MtkDataBufferFree(&data1);
    MtkDataBufferFree(&mask1);
    MtkDataBufferFree(&regrdata);
    MtkDataBufferFree(&regrmask);
    Py_XDECREF(regrdata_arr);
    Py_XDECREF(regrmask_arr);    
    return NULL;
}

static PyObject *
MtkRegression_ResampleRegCoeff(MtkRegression *self, PyObject *args)
{
    PyObject *result = NULL;
    MTKt_status status = MTK_FAILURE;

    /* Inputs */
    PyObject *arg1 = NULL; 
    PyObject *arg2 = NULL; 
    PyObject *arg3 = NULL;
	MTKt_RegressionCoeff regr_coeff = MTKT_REGRESSION_COEFF_INIT;
    RegCoeff *py_regr_coeff = NULL;
    MTKt_MapInfo *regr_map_info = NULL;
    MTKt_MapInfo *target_map_info = NULL;

    /* Outputs */
	MTKt_RegressionCoeff regr_coeff_out = MTKT_REGRESSION_COEFF_INIT;
    RegCoeff *py_regr_coeff_out = NULL;


    if (PyTuple_Size(args) == 3) 
    {
        if (!PyArg_ParseTuple(args,"OOO",&arg1,&arg2,&arg3)) {
            PyErr_SetString(PyExc_StandardError, "Problem parsing arguments.");
            return NULL;
        }

        /* Inputs */
        if (PyObject_TypeCheck(arg1,&RegCoeffType)) {
            py_regr_coeff = (RegCoeff*)arg1;
            status = Mtk_PyRegCoeffToMtk(&py_regr_coeff, &regr_coeff);
            MTK_ERR_PY_COND_JUMP(status);                      
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 1 must be a Reg Coeff type");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg2,&MtkMapInfoType)) {
            regr_map_info = &((MtkMapInfo*)arg2)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 2 must be a mapinfo");
            return NULL;            
        }
        if (PyObject_TypeCheck(arg3,&MtkMapInfoType)) {
            target_map_info = &((MtkMapInfo*)arg3)->mapinfo;            
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument 3 must be a mapinfo");
            return NULL;            
        }
        
        /* Misr Toolkit Call */
        py_regr_coeff_out = PyObject_New(RegCoeff, &RegCoeffType);
        RegCoeff_init(py_regr_coeff_out,NULL,NULL);
        
        status =  MtkResampleRegressionCoeff(&regr_coeff, regr_map_info, target_map_info, &regr_coeff_out );
        if (status != MTK_SUCCESS)
        {
            PyErr_SetString(PyExc_StandardError, "MtkResampleRegressionCoeff Failed");
            return NULL;
        }        

        /* Outputs */
        status = Mtk_MtkRegCoeffToPy(&regr_coeff_out, &py_regr_coeff_out);
        MTK_ERR_PY_COND_JUMP(status);
        result = Py_BuildValue("N",py_regr_coeff_out);
        return result;        
    } else {
        PyErr_SetString(PyExc_StandardError, "Wrong number of arguments.");
        return NULL;
    }

ERROR_HANDLE:
    Py_XDECREF(py_regr_coeff_out);
    return NULL;
}

static PyGetSetDef MtkRegression_getseters[] = {
   {NULL}  /* Sentinel */
};

PyMethodDef mtkregression_methods[] = {
     {"downsample", (PyCFunction)MtkRegression_Downsample, METH_VARARGS,
      "Downsamples data by averaging pixels."},
     {"linear_regression_calc", (PyCFunction)MtkRegression_LinearRegressionCalc, METH_VARARGS,
      "Uses linear regression to fit data."},
     {"smooth_data", (PyCFunction)MtkRegression_SmoothData, METH_VARARGS,
      "Smooths the given array with a boxcar average."},
     {"upsample_mask", (PyCFunction)MtkRegression_UpsampleMask, METH_VARARGS,
      "Upsamples a mask by nearest neighbor sampling."},
     {"coeff_calc", (PyCFunction)MtkRegression_CoeffCalc, METH_VARARGS,
      "Calculates linear regression coefficients for translating values."}, 
     {"apply_regression", (PyCFunction)MtkRegression_ApplyRegression, METH_VARARGS,
      "Applies regression to given data."},
     {"resample_reg_coeff", (PyCFunction)MtkRegression_ResampleRegCoeff, METH_VARARGS,
      "Resamples regression coefficients at each pixel."},            
    {NULL}  /* Sentinel */
};

PyTypeObject MtkRegressionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkRegression",      /*tp_name*/
    sizeof(MtkRegression),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkRegression_dealloc,/*tp_dealloc*/
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
    "MtkRegression objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    mtkregression_methods,           /* tp_methods */
    0,                         /* tp_members */
    MtkRegression_getseters,         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkRegression_init,    /* tp_init */
    0,                         /* tp_alloc */
    MtkRegression_new,  /*PyType_GenericNew()*/              /* tp_new */
};

