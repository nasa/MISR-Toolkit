#ifndef PYHELPER_H
#define PYHELPER_H

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

/* --------------------------------------------------------------- */
/* Mtk_MtkToPyArrDatatype                                          */
/* --------------------------------------------------------------- */
MTKt_status Mtk_MtkToPyArrDatatype( MTKt_DataType mtk_datatype, int *py_datatype);

/* --------------------------------------------------------------- */
/* Mtk_PyToMTkDatatype                                             */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyArrToMtkDatatype(PyObject* pyobj, MTKt_DataType *mtk_datatype );

/* --------------------------------------------------------------- */
/* Mtk_DataBufferToPyArray                                         */
/* --------------------------------------------------------------- */
MTKt_status Mtk_DataBufferToPyArray(MTKt_DataBuffer *databuf, PyObject **pyobj, int py_datatype);

/* --------------------------------------------------------------- */
/* Mtk_PyArrayToDataBuffer                                         */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyArrayToDataBuffer(PyObject **pyobj, MTKt_DataBuffer *databuf);

/* --------------------------------------------------------------- */
/* Mtk_MtkRegCoeffToPy                                             */
/* --------------------------------------------------------------- */
MTKt_status Mtk_MtkRegCoeffToPy(MTKt_RegressionCoeff *regr_coeff, RegCoeff **py_regr_coeff);

/* --------------------------------------------------------------- */
/* Mtk_PyRegCoeffToMtk                                             */
/* --------------------------------------------------------------- */
MTKt_status Mtk_PyRegCoeffToMtk(RegCoeff **py_regr_coeff, MTKt_RegressionCoeff *regr_coeff);


#define MTK_ERR_PY_COND_JUMP(status) \
    if (status != MTK_SUCCESS) { \
        if(PyErr_Occurred()) { \
            PyErr_SetString(PyExc_StandardError, "Unknown error occured."); \
        }\
        goto ERROR_HANDLE; \
    }
  
#define MTK_ERR_PY_JUMP(status) \
    { \
        PyErr_SetString(PyExc_TypeError, mtk_errdesc[status]); \
        goto ERROR_HANDLE; \
    }

#endif /* PYHELPER_H */
