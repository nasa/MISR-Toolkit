/*===========================================================================
=                                                                           =
=                                  PyMtk                                    =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef PYMTK_H
#define PYMTK_H

#include <Python.h>
#include "MisrToolkit.h"
#include <hdf.h>

typedef struct {
    PyObject_HEAD
    MTKt_MisrProjParam pp;
} MtkProjParam;

typedef struct {
   PyObject_HEAD
   MTKt_GeoCoord gc;
} MtkGeoCoord;

typedef struct {
   PyObject_HEAD
   int block_number;
   MtkGeoCoord *ulc;
   MtkGeoCoord *urc;
   MtkGeoCoord *ctr;
   MtkGeoCoord *lrc;
   MtkGeoCoord *llc;
} MtkGeoBlock;

typedef struct {
   PyObject_HEAD
   int path;
   int start_block;
   int end_block;
   MtkGeoBlock *gb[NBLOCK + 1];
} MtkBlockCorners;

typedef struct
{
   PyObject_HEAD
   MTKt_Region region;
} Region;

typedef struct
{
   PyObject_HEAD
   MTKt_DataBuffer databuf;
   MTKt_MapInfo mapinfo;
} DataPlane;

typedef struct
{
   PyObject_HEAD
   MTKt_DataBuffer valid_mask;
   MTKt_DataBuffer slope;
   MTKt_DataBuffer intercept;
   MTKt_DataBuffer correlation;      
} RegCoeff;

typedef struct
{
   PyObject_HEAD
   MTKt_SomCoord som_coord;
} MtkSomCoord;

typedef struct
{
   PyObject_HEAD
   MtkGeoCoord *ulc;
   MtkGeoCoord *urc;
   MtkGeoCoord *ctr;
   MtkGeoCoord *lrc;
   MtkGeoCoord *llc;
} MtkGeoRegion;

typedef struct
{
   PyObject_HEAD
   int path;
   MtkSomCoord *ulc;
   MtkSomCoord *ctr;
   MtkSomCoord *lrc;
} MtkSomRegion;

typedef struct
{
   PyObject_HEAD
   PyObject *pixelcenter; /* PyBool Type*/
   MtkSomRegion *som;
   MtkGeoRegion *geo;
   MtkProjParam *pp;
   MTKt_MapInfo mapinfo;
} MtkMapInfo;

typedef struct
{
   PyObject_HEAD
   MTKt_TimeMetaData time_metadata;
} MtkTimeMetaData;

typedef struct
{
   PyObject_HEAD
   int32 fid;
   int32 sid;
   int32 hdf_fid;
   int ncid;

} MtkFileId;

typedef struct
{
   PyObject_HEAD
   PyObject *filename;
   PyObject *gridname;
   PyObject *fieldname;
   MtkFileId *file_id;

} MtkField;

typedef struct
{
   PyObject_HEAD
   PyObject *filename;
   PyObject *gridname;
   MtkField **fields;
   int num_fields;
   int max_fields;
   MtkFileId *file_id;

} MtkGrid;

typedef struct
{
   PyObject_HEAD
   PyObject *filename;
   MtkFileId *file_id;
   MtkGrid **grids;
   int num_grids;

} MtkFile;

typedef struct
{
   PyObject_HEAD
} MtkReProject;

typedef struct
{
   PyObject_HEAD
} MtkRegression;


#define MTK_READ_ONLY_ATTR(PYMTKTYPE, ATTR_NAME) \
static int \
PYMTKTYPE ## _set ## ATTR_NAME (PYMTKTYPE *self, PyObject *value, void *closure) \
{ \
   if (value == NULL) \
   { \
      PyErr_SetString(PyExc_TypeError, "Cannot delete the " # ATTR_NAME " attribute."); \
      return -1; \
   } \
\
   return 0; \
}

#if PY_MAJOR_VERSION >= 3
    #define PyInt_FromLong PyLong_FromLong
    #define PyInt_AsLong PyLong_AsLong
    #define PyInt_Check PyLong_Check
    #define PyExc_StandardError PyExc_Exception
    #define PyString PyBytes
    #define PyString_FromString PyUnicode_FromString
    #define PyString_AsString PyUnicode_AsUTF8
    #define PyString_Format PyUnicode_Format
#else

#endif

#endif /* PYMTK_H */
