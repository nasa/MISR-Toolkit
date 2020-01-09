#include "pyhelpers.h"
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

char *mtk_errdesc[] = MTK_ERR_DESC;

/* --------------------------------------------------------------- */
/* Mtk_MtkToPyArrDatatype					                       */
/* --------------------------------------------------------------- */
MTKt_status Mtk_MtkToPyArrDatatype( MTKt_DataType mtk_datatype, int *py_datatype) {
    MTKt_status status = MTK_FAILURE;
    
    switch (mtk_datatype) {
        case MTKe_char8:        
        case MTKe_int8:
            *py_datatype = NPY_BYTE;
            break;
        case MTKe_uchar8:        
        case MTKe_uint8:	
            *py_datatype = NPY_UBYTE;
            break;
        case MTKe_int16:
        	*py_datatype = NPY_SHORT;
            break;
        case MTKe_uint16:
        	*py_datatype = NPY_USHORT;
            break;
        case MTKe_int32:
        	*py_datatype = NPY_INT32;
            break;
        case MTKe_uint32:
        	*py_datatype = NPY_UINT32;
            break;
        case MTKe_int64:
        	*py_datatype = NPY_INT64;
            break;
        case MTKe_uint64:
        	*py_datatype = NPY_UINT64;
            break;
        case MTKe_float:
        	*py_datatype = NPY_FLOAT32;
            break;
        case MTKe_double:
        	*py_datatype = NPY_FLOAT64;
            break;
        default:
            MTK_ERR_PY_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
            break;
    }

    return(MTK_SUCCESS);
    ERROR_HANDLE:
    return (status);
}

/* --------------------------------------------------------------- */
/* Mtk_PyToMTkDatatype					                           */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyArrToMtkDatatype(PyObject* pyobj, MTKt_DataType *mtk_datatype ) {
    MTKt_status status = MTK_FAILURE;
    //check pyarray
    int type_num = PyArray_TYPE((PyArrayObject *)pyobj);
    
    switch (type_num) {
        case NPY_INT8:
//      case NPY_BYTE:
            *mtk_datatype = MTKe_int8;
            break;        
        case NPY_UINT8:	
//      case NPY_UBYTE:	        
            *mtk_datatype = MTKe_uint8;
            break;
        case NPY_INT16:	
            *mtk_datatype = MTKe_int16;
            break;
        case NPY_UINT16:
        	*mtk_datatype = MTKe_uint16;
            break;
        case NPY_INT32:	
            *mtk_datatype = MTKe_int32;
            break;
        case NPY_UINT32:	
            *mtk_datatype = MTKe_uint32;
            break;
        case NPY_INT64:	
            *mtk_datatype = MTKe_int64;
            break;
        case NPY_UINT64:	
            *mtk_datatype = MTKe_uint64;
            break;
        case NPY_FLOAT32:	
            *mtk_datatype = MTKe_float;
            break;
        case NPY_FLOAT64:	
            *mtk_datatype = MTKe_double;
            break;
        default:
            MTK_ERR_PY_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
            break;
    }
    return(MTK_SUCCESS);
ERROR_HANDLE:
    return (status);
}

/* --------------------------------------------------------------- */
/* Mtk_DataBufferToPyArray					                       */
/* --------------------------------------------------------------- */
MTKt_status Mtk_DataBufferToPyArray(MTKt_DataBuffer *databuf, PyObject **pyobj, int py_datatype) {
    MTKt_status status = MTK_FAILURE;
    npy_intp dims[2];    
    dims[0] = databuf->nline;
    dims[1] = databuf->nsample;
    *pyobj = PyArray_SimpleNewFromData(2, dims, py_datatype, databuf->vdata[0]);                
    if ( !(*pyobj) ) {
        PyErr_SetString(PyExc_TypeError, "Problem Converting DataBuffer to PyArray.");
        return status;
    }
    status = MTK_SUCCESS;
    return status;
}

/* --------------------------------------------------------------- */
/* Mtk_PyArrayToDataBuffer					                       */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyArrayToDataBuffer(PyObject **pyobj, MTKt_DataBuffer *databuf) {
    MTKt_status status = MTK_FAILURE;    
    PyObject *databuf_arr = NULL;   
    int databuf_nlines = 0;
    int databuf_nsamples = 0;
    int *py_datatype = malloc(sizeof(*py_datatype));
    MTKt_DataType mtk_datatype=0;
    *py_datatype = 0;
    
    status = Mtk_PyArrToMtkDatatype(*pyobj, &mtk_datatype);   
    if (status != MTK_SUCCESS) { 
       PyErr_SetString(PyExc_StandardError, "Problem converting numpy dtype to Mtk dtype");
       return status;
    }    

    databuf_arr = PyArray_FROMANY(*pyobj, PyArray_TYPE((PyArrayObject *)*pyobj), 2, 2, NPY_ARRAY_IN_ARRAY);
    if (!databuf_arr) {
        
        PyErr_SetString(PyExc_TypeError, "Type Problem converting to PyArray.");
        goto ERROR_HANDLE;
    }
    
    databuf_nlines = (int) PyArray_DIM((PyArrayObject *)*pyobj, 0);
    databuf_nsamples = (int) PyArray_DIM((PyArrayObject *)*pyobj, 1);
    status = MtkDataBufferAllocate(databuf_nlines, databuf_nsamples, mtk_datatype, databuf);
    if (status != MTK_SUCCESS)
    {
        PyErr_SetString(PyExc_StandardError, "MtkDataBufferAllocate Failed");
        goto ERROR_HANDLE;
    }
    
    status = Mtk_MtkToPyArrDatatype((*databuf).datatype,py_datatype);
    if (status != MTK_SUCCESS)
    {
        PyErr_SetString(PyExc_StandardError, "Problem converting Mtk dtype to numpy dtype");
        goto ERROR_HANDLE;
    }
    
    (*databuf).vdata[0] = PyArray_DATA((PyArrayObject *)*pyobj);

    /* Hook the data pointer to the data */
    (*databuf).dataptr = (*databuf).vdata[0];


    Py_DECREF(databuf_arr);
    return status;
    
ERROR_HANDLE:
    Py_XDECREF(databuf_arr);
    return status;    
}

/* --------------------------------------------------------------- */
/* Mtk_MtkRegCoeffToPy                                             */
/* --------------------------------------------------------------- */
MTKt_status Mtk_MtkRegCoeffToPy(MTKt_RegressionCoeff *regr_coeff, RegCoeff **py_regr_coeff) {
    MTKt_status status = MTK_FAILURE;
    (*py_regr_coeff)->valid_mask = (*regr_coeff).valid_mask;
    (*py_regr_coeff)->slope = (*regr_coeff).slope;
    (*py_regr_coeff)->intercept = (*regr_coeff).intercept;
    (*py_regr_coeff)->correlation = (*regr_coeff).correlation;            
    status = MTK_SUCCESS;
    return status;
}

/* --------------------------------------------------------------- */
/* Mtk_PyRegCoeffToMtk                                             */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyRegCoeffToMtk(RegCoeff **py_regr_coeff, MTKt_RegressionCoeff *regr_coeff) {
    MTKt_status status = MTK_FAILURE;
    (*regr_coeff).valid_mask = (*py_regr_coeff)->valid_mask;
    (*regr_coeff).slope = (*py_regr_coeff)->slope;
    (*regr_coeff).intercept = (*py_regr_coeff)->intercept;
    (*regr_coeff).correlation = (*py_regr_coeff)->correlation;
/*    (*py_regr_coeff)->valid_mask = (*regr_coeff).valid_mask;
    
    (*py_regr_coeff)->slope = (*regr_coeff).slope;
    (*py_regr_coeff)->intercept = (*regr_coeff).intercept;
    (*py_regr_coeff)->correlation = (*regr_coeff).correlation;  */          
    status = MTK_SUCCESS;
    return status;
}
