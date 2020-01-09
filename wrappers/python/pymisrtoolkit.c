/*===========================================================================
=                                                                           =
=                              PyMisrToolkit                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/
#define PY_SSIZE_T_CLEAN
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
#include "numpy/arrayobject.h"
#include "MisrToolkit.h"

extern PyMethodDef coordquery_methods[];
extern PyMethodDef filequery_methods[];
extern PyMethodDef orbitpath_methods[];
extern PyMethodDef unitconv_methods[];
extern PyMethodDef util_methods[];

/* MtkProjParam */
extern PyTypeObject MtkProjParamType;
extern PyMethodDef mtkprojparam_methods[];

/* MtkGeoCoord */
extern PyTypeObject MtkGeoCoordType;
extern PyMethodDef mtkgeocoord_methods[];

/* MtkGeoBlock */
extern PyTypeObject MtkGeoBlockType;
extern PyMethodDef mtkgeoblock_methods[];

/* MtkBlockCorners */
extern PyTypeObject MtkBlockCornersType;
extern PyMethodDef mtkblockcorners_methods[];

/* MtkRegion Type */
extern PyTypeObject RegionType;
extern PyMethodDef region_methods[];

/* MtkDataPlane Type */
extern PyTypeObject DataPlaneType;
extern PyMethodDef  dataplane_methods[];

/* MtkSomCoord */
extern PyTypeObject MtkSomCoordType;
extern PyMethodDef mtksomcoord_methods[];

/* MtkSomRegion */
extern PyTypeObject MtkSomRegionType;
extern PyMethodDef mtksomregion_methods[];

/* MtkGeoRegion */
extern PyTypeObject MtkGeoRegionType;
extern PyMethodDef mtkgeoregion_methods[];

/* MtkMapInfo Type */
extern PyTypeObject MtkMapInfoType;
extern PyMethodDef  mtkmapinfo_methods[];

/* MtkTimeMetaData Type */
extern PyTypeObject MtkTimeMetaDataType;
extern PyMethodDef  mtktimemetadata_methods[];

/* MtkFile Type */
extern PyTypeObject pyMtkFileType;
extern PyMethodDef mtkfile_methods[];

/* MtkFileId Type */
extern PyTypeObject MtkFileIdType;
extern PyMethodDef mtkfileid_methods[];

/* MtkGrid Type */
extern PyTypeObject MtkGridType;
extern PyMethodDef mtkgrid_methods[];

/* MtkField Type */
extern PyTypeObject MtkFieldType;
extern PyMethodDef mtkfield_methods[];

/* MtkReProject Type */
extern PyTypeObject MtkReProjectType;
extern PyMethodDef mtkreproject_methods[];

/* MtkRegression Type */
extern PyTypeObject MtkRegressionType;
extern PyMethodDef mtkregression_methods[];

/* MtkRegCoeff Type */
extern PyTypeObject RegCoeffType;
extern PyMethodDef  regcoeff_methods[];

/* size PyMethodDef array */
#define MTK_METHODDEF_SIZE 100

/* All of the root Mtk methods in a single PyMethodDef */
struct MtkMethods { int index; PyMethodDef methods[MTK_METHODDEF_SIZE]; };
typedef struct MtkMethods MtkMethods;
MtkMethods mtk_methods;


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
	#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION >= 3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
                PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif

void fill_methods(MtkMethods* mtk_methods, PyMethodDef added_methods[]) {
    int added_index = 0;
    while (mtk_methods->index < MTK_METHODDEF_SIZE - 2 && added_methods[added_index].ml_name != NULL) {
        mtk_methods->methods[mtk_methods->index] = added_methods[added_index];
        added_index++;
        mtk_methods->index++;
    }
    /* Add sentinel */
    mtk_methods->methods[mtk_methods->index].ml_name = NULL;
    mtk_methods->methods[mtk_methods->index].ml_meth = NULL;
    mtk_methods->methods[mtk_methods->index].ml_flags = 0;
    mtk_methods->methods[mtk_methods->index].ml_doc = NULL;

}

MOD_INIT(MisrToolkit)
{
   PyObject *m;
   
   if (strcmp(MTK_VERSION,MtkVersion()) != 0)
   {
        PyErr_Format(PyExc_ImportError, "Python MISR Toolkit V%s does not"
        " match MISR Toolkit Library V%s.", MTK_VERSION, MtkVersion());
        return MOD_ERROR_VAL;
   }
   
   fill_methods(&mtk_methods, coordquery_methods);
   fill_methods(&mtk_methods, filequery_methods);
   fill_methods(&mtk_methods, orbitpath_methods);
   fill_methods(&mtk_methods, unitconv_methods);
   fill_methods(&mtk_methods, util_methods);
   fill_methods(&mtk_methods, mtkprojparam_methods);
   fill_methods(&mtk_methods, mtkgeocoord_methods);
   fill_methods(&mtk_methods, mtkgeoblock_methods);
   fill_methods(&mtk_methods, mtkblockcorners_methods);
   fill_methods(&mtk_methods, region_methods);
   fill_methods(&mtk_methods, dataplane_methods);
   fill_methods(&mtk_methods, mtksomcoord_methods);
   fill_methods(&mtk_methods, mtksomregion_methods);
   fill_methods(&mtk_methods, mtkgeoregion_methods);
   fill_methods(&mtk_methods, mtkmapinfo_methods);
   fill_methods(&mtk_methods, mtktimemetadata_methods);
   fill_methods(&mtk_methods, mtkfile_methods);
   fill_methods(&mtk_methods, mtkfileid_methods);
   fill_methods(&mtk_methods, mtkgrid_methods);
   fill_methods(&mtk_methods, mtkfield_methods);
   fill_methods(&mtk_methods, mtkreproject_methods);
   fill_methods(&mtk_methods, mtkregression_methods);
   fill_methods(&mtk_methods, regcoeff_methods);

   MOD_DEF(m, "MisrToolkit", NULL, mtk_methods.methods)

   if (m == NULL) {
       return MOD_ERROR_VAL;
   }

   /* Add MtkProjParam Type */
   MtkProjParamType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkProjParamType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkProjParamType);
   PyModule_AddObject(m, "MtkProjParam", (PyObject *)&MtkProjParamType);

   /* Add MtkGeoCoord Type */
   MtkGeoCoordType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoCoordType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkGeoCoordType);
   PyModule_AddObject(m, "MtkGeoCoord", (PyObject *)&MtkGeoCoordType);

   /* Add MtkGeoBlock Type */
   MtkGeoBlockType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoBlockType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkGeoBlockType);
   PyModule_AddObject(m, "MtkGeoBlock", (PyObject *)&MtkGeoBlockType);

   /* Add MtkBlockCorners Type */
   MtkBlockCornersType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkBlockCornersType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkBlockCornersType);
   PyModule_AddObject(m, "MtkBlockCorners", (PyObject *)&MtkBlockCornersType);

   /* Add MtkRegion Type */
   RegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&RegionType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&RegionType);
   PyModule_AddObject(m, "MtkRegion", (PyObject *)&RegionType);

   /* Add MtkDataPlane Type */
   DataPlaneType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&DataPlaneType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&DataPlaneType);
   PyModule_AddObject(m, "MtkDataPlane", (PyObject *)&DataPlaneType);

   /* Add MtkSomCoord Type */
   MtkSomCoordType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkSomCoordType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkSomCoordType);
   PyModule_AddObject(m, "MtkSomCoord", (PyObject *)&MtkSomCoordType);

   /* Add MtkSomRegion Type */
   MtkSomRegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkSomRegionType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkSomRegionType);
   PyModule_AddObject(m, "MtkSomRegion", (PyObject *)&MtkSomRegionType);

  /* Add MtkGeoRegion Type */
   MtkGeoRegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoRegionType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkGeoRegionType);
   PyModule_AddObject(m, "MtkGeoRegion", (PyObject *)&MtkGeoRegionType);

   /* Add MtkMapInfo Type */
   MtkMapInfoType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkMapInfoType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkMapInfoType);
   PyModule_AddObject(m, "MtkMapInfo", (PyObject *)&MtkMapInfoType);

   /* Add MtkTimeMetaData Type */
   MtkTimeMetaDataType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkTimeMetaDataType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkTimeMetaDataType);
   PyModule_AddObject(m, "MtkTimeMetaData", (PyObject *)&MtkTimeMetaDataType);

   /* Add MtkFile Type */
   pyMtkFileType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&pyMtkFileType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&pyMtkFileType);
   PyModule_AddObject(m, "MtkFile", (PyObject *)&pyMtkFileType);

   /* Add MtkFileId Type */
   MtkFileIdType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkFileIdType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkFileIdType);
   PyModule_AddObject(m, "MtkFileId", (PyObject *)&MtkFileIdType);

   /* Add MtkGrid Type */
   MtkGridType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGridType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkGridType);
   PyModule_AddObject(m, "MtkGrid", (PyObject *)&MtkGridType);

   /* Add MtkField Type */
   MtkFieldType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkFieldType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkFieldType);
   PyModule_AddObject(m, "MtkField", (PyObject *)&MtkFieldType);

   /* Add MtkReProject Type */
   MtkReProjectType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkReProjectType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkReProjectType);
   PyModule_AddObject(m, "MtkReProject", (PyObject *)&MtkReProjectType);

   /* Add MtkRegression Type */
   MtkRegressionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkRegressionType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&MtkRegressionType);
   PyModule_AddObject(m, "MtkRegression", (PyObject *)&MtkRegressionType);

   /* Add MtkRegCoeff Type */
   RegCoeffType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&RegCoeffType) < 0) {
      return MOD_ERROR_VAL;
   }
   Py_INCREF(&RegCoeffType);
   PyModule_AddObject(m, "MtkRegCoeff", (PyObject *)&RegCoeffType);

   /* import numpy API */
   import_array();

   return MOD_SUCCESS_VAL(m);
}
