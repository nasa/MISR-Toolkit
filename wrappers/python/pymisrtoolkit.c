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



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initMisrToolkit(void)
{
   PyObject *m;
   
   if (strcmp(MTK_VERSION,MtkVersion()) != 0)
   {
   	  PyErr_Format(PyExc_ImportError, "Python MISR Toolkit V%s does not"
   	  " match MISR Toolkit Library V%s.", MTK_VERSION, MtkVersion());
   	  return;
   }
   
   m = Py_InitModule("MisrToolkit", coordquery_methods);
   m = Py_InitModule("MisrToolkit", filequery_methods);
   m = Py_InitModule("MisrToolkit", orbitpath_methods);
   m = Py_InitModule("MisrToolkit", unitconv_methods);
   m = Py_InitModule("MisrToolkit", util_methods);

   /* Add MtkProjParam Type */
   MtkProjParamType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkProjParamType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkprojparam_methods, NULL);

   Py_INCREF(&MtkProjParamType);
   PyModule_AddObject(m, "MtkProjParam", (PyObject *)&MtkProjParamType);

   /* Add MtkGeoCoord Type */
   MtkGeoCoordType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoCoordType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkgeocoord_methods, NULL);

   Py_INCREF(&MtkGeoCoordType);
   PyModule_AddObject(m, "MtkGeoCoord", (PyObject *)&MtkGeoCoordType);

   /* Add MtkGeoBlock Type */
   MtkGeoBlockType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoBlockType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkgeoblock_methods, NULL);

   Py_INCREF(&MtkGeoBlockType);
   PyModule_AddObject(m, "MtkGeoBlock", (PyObject *)&MtkGeoBlockType);

   /* Add MtkBlockCorners Type */
   MtkBlockCornersType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkBlockCornersType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkblockcorners_methods, NULL);

   Py_INCREF(&MtkBlockCornersType);
   PyModule_AddObject(m, "MtkBlockCorners", (PyObject *)&MtkBlockCornersType);

   /* Add MtkRegion Type */
   RegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&RegionType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", region_methods, NULL);

   Py_INCREF(&RegionType);
   PyModule_AddObject(m, "MtkRegion", (PyObject *)&RegionType);

   /* Add MtkDataPlane Type */
   DataPlaneType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&DataPlaneType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", dataplane_methods, NULL);

   Py_INCREF(&DataPlaneType);
   PyModule_AddObject(m, "MtkDataPlane", (PyObject *)&DataPlaneType);

   /* Add MtkSomCoord Type */
   MtkSomCoordType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkSomCoordType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtksomcoord_methods, NULL);

   Py_INCREF(&MtkSomCoordType);
   PyModule_AddObject(m, "MtkSomCoord", (PyObject *)&MtkSomCoordType);

   /* Add MtkSomRegion Type */
   MtkSomRegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkSomRegionType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtksomregion_methods, NULL);

   Py_INCREF(&MtkSomRegionType);
   PyModule_AddObject(m, "MtkSomRegion", (PyObject *)&MtkSomRegionType);

  /* Add MtkGeoRegion Type */
   MtkGeoRegionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGeoRegionType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkgeoregion_methods, NULL);

   Py_INCREF(&MtkGeoRegionType);
   PyModule_AddObject(m, "MtkGeoRegion", (PyObject *)&MtkGeoRegionType);
   
   /* Add MtkMapInfo Type */
   MtkMapInfoType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkMapInfoType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkmapinfo_methods, NULL);

   Py_INCREF(&MtkMapInfoType);
   PyModule_AddObject(m, "MtkMapInfo", (PyObject *)&MtkMapInfoType);

   /* Add MtkTimeMetaData Type */
   MtkTimeMetaDataType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkTimeMetaDataType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkmapinfo_methods, NULL);

   Py_INCREF(&MtkTimeMetaDataType);
   PyModule_AddObject(m, "MtkTimeMetaData", (PyObject *)&MtkTimeMetaDataType);

   /* Add MtkFile Type */
   pyMtkFileType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&pyMtkFileType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkfile_methods, NULL);

   Py_INCREF(&pyMtkFileType);
   PyModule_AddObject(m, "MtkFile", (PyObject *)&pyMtkFileType);

   /* Add MtkFileId Type */
   MtkFileIdType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkFileIdType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkfileid_methods, NULL);

   Py_INCREF(&MtkFileIdType);
   PyModule_AddObject(m, "MtkFileId", (PyObject *)&MtkFileIdType);

   /* Add MtkGrid Type */
   MtkGridType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkGridType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkgrid_methods, NULL);

   Py_INCREF(&MtkGridType);
   PyModule_AddObject(m, "MtkGrid", (PyObject *)&MtkGridType);

   /* Add MtkField Type */
   MtkFieldType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkFieldType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkfield_methods, NULL);

   Py_INCREF(&MtkFieldType);
   PyModule_AddObject(m, "MtkField", (PyObject *)&MtkFieldType);
   
   /* Add MtkReProject Type */
   MtkReProjectType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkReProjectType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkreproject_methods, NULL);

   Py_INCREF(&MtkReProjectType);
   PyModule_AddObject(m, "MtkReProject", (PyObject *)&MtkReProjectType);
   
   /* Add MtkRegression Type */
   MtkRegressionType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&MtkRegressionType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", mtkregression_methods, NULL);

   Py_INCREF(&MtkRegressionType);
   PyModule_AddObject(m, "MtkRegression", (PyObject *)&MtkRegressionType);
   
   /* Add MtkRegCoeff Type */
   RegCoeffType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&RegCoeffType) < 0)
      return;

   m = Py_InitModule3("MisrToolkit", regcoeff_methods, NULL);

   Py_INCREF(&RegCoeffType);
   PyModule_AddObject(m, "MtkRegCoeff", (PyObject *)&RegCoeffType);

   import_array();
}
