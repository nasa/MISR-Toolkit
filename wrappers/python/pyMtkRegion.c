/*===========================================================================
=                                                                           =
=                                MtkRegion                                  =
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
#include "pyMtk.h"

extern PyTypeObject MtkMapInfoType;
extern int MtkMapInfo_init(MtkMapInfo *self, PyObject *args, PyObject *kwds);
extern int MtkMapInfo_copy(MtkMapInfo *self, MTKt_MapInfo mapinfo);

static void
Region_dealloc(Region* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Region_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   Region *self;
   MTKt_Region init = MTKT_REGION_INIT;

   self = (Region *)type->tp_alloc(type,0);
   if (self != NULL)
      self->region = init;

   return (PyObject *)self;
}

static int
Region_init(Region *self, PyObject *args, PyObject *kwds)
{
   MTKt_status status;
   int path;
   int start_block;
   int end_block;

   double ulc_lat;
   double ulc_lon;
   double lrc_lat;
   double lrc_lon;

   double ctr_lat;
   double ctr_lon;
   double lat_extent;
   double lon_extent;
   char *extent_units;

   MTKt_Region region = MTKT_REGION_INIT;

   switch (PyTuple_Size(args))
   {
      case 3: if (!PyArg_ParseTuple(args,"iii",&path,&start_block,&end_block))
                 return -1;
         status = MtkSetRegionByPathBlockRange(path,start_block,end_block,&region);
         if (status != MTK_SUCCESS)
         {
            PyErr_SetString(PyExc_StandardError, "MtkSetRegionByPathBlockRange Failed");
            return -1;
         }
         break;
      case 4: if (!PyArg_ParseTuple(args,"dddd",&ulc_lat,&ulc_lon,&lrc_lat,&lrc_lon))
	         return -1;
 	 status = MtkSetRegionByUlcLrc(ulc_lat,ulc_lon,lrc_lat,lrc_lon,&region);
         if (status != MTK_SUCCESS)
         {
            PyErr_SetString(PyExc_StandardError, "MtkSetRegionByUlcLrc Failed");
            return -1;
         }
         break;
      case 5:
         if (!PyArg_ParseTuple(args,"idddd",&path,&ulc_lat,&ulc_lon,&lrc_lat,&lrc_lon)) {
             PyErr_Clear(); //we want to try to parse arguments as extent instead then
             // printf("Didn't compute with int, got extent as expected.\n");
             if (!PyArg_ParseTuple(args,"dddds",&ctr_lat,&ctr_lon,&lat_extent,&lon_extent,&extent_units))
        	         return -1;
        	 status = MtkSetRegionByLatLonExtent(ctr_lat,ctr_lon,lat_extent,lon_extent,extent_units,&region);
             if (status != MTK_SUCCESS)
             {
                PyErr_SetString(PyExc_StandardError, "MtkSetRegionByLatLonExtent Failed");
                return -1;
             }
         } else { 
        	 status =  MtkSetRegionByPathSomUlcLrc(path,ulc_lat,ulc_lon,lrc_lat,lrc_lon,&region);
             if (status != MTK_SUCCESS)
             {
                PyErr_SetString(PyExc_StandardError, "MtkSetRegionByPathSomUlcLrc Failed");
                return -1;
             }
         }
         break;
      default:
         break;
   }

   self->region = region;

   return 0;
}

static PyObject *
Region_name(Region* self)
{
   static PyObject *format = NULL;
   PyObject *args, *result;

   if (format == NULL)
   {
      format = PyString_FromString("Ctr Lat: %s Ctr Lon: %s "
                                   "Ext xlat: %s Ext ylon: %s");
      if (format == NULL)
	 return NULL;
   }

   args = Py_BuildValue("dddd",self->region.geo.ctr.lat,self->region.geo.ctr.lon,
                        2.0*self->region.hextent.xlat,2.0*self->region.hextent.ylon);
   if (args == NULL)
      return NULL;

   result = PyString_Format(format,args);
   Py_DECREF(args);

   return result;
}

static PyObject *
Region_BlockRange(Region* self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   int path;
   int start_block;
   int end_block;

   if (!PyArg_ParseTuple(args,"i",&path))
      return NULL;

   status = MtkRegionPathToBlockRange(self->region, path, &start_block,
				      &end_block);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkRegionPathToBlockRange Failed");
      return NULL;
   }

   result = Py_BuildValue("ii",start_block,end_block);
   return result;
}

static PyObject *
Region_getcenter(Region *self, void *closure)
{
   PyObject *result;

   result = Py_BuildValue("dd",self->region.geo.ctr.lat,self->region.geo.ctr.lon);
   return result;
}

MTK_READ_ONLY_ATTR(Region, center)

static PyObject *
Region_getextent(Region *self, void *closure)
{
   PyObject *result;

  result = Py_BuildValue("dd",2.0*self->region.hextent.xlat,2.0*self->region.hextent.ylon);
   return result;
}

MTK_READ_ONLY_ATTR(Region, extent)

static PyObject *
Region_getpath_list(Region *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   int pathcnt;
   int *pathlist;
   int i;

   status = MtkRegionToPathList(self->region, &pathcnt, &pathlist);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkRegionToPathList Failed");
      return NULL;
   }

   result = PyList_New(pathcnt);
   for (i = 0; i < pathcnt; ++i)
     PyList_SetItem(result, i, Py_BuildValue("i",pathlist[i]));

   free(pathlist);

   return result;
}

MTK_READ_ONLY_ATTR(Region, path_list)

static PyObject *
Region_SnapToGrid(Region* self, PyObject *args)
{
  /*PyObject *result;*/
   MTKt_status status;
   int path;
   int resolution;
   MtkMapInfo *mapinfo;
   MTKt_MapInfo mp;

   if (!PyArg_ParseTuple(args,"ii",&path,&resolution))
      return NULL;

   mapinfo = (MtkMapInfo*)PyObject_New(MtkMapInfo, &MtkMapInfoType);
   MtkMapInfo_init(mapinfo,NULL,NULL);
   
   if (mapinfo == NULL)
   {
      PyErr_Format(PyExc_StandardError, "Problem initializing MtkMapInfo.");
      return NULL;
   }

   status = MtkSnapToGrid(path, resolution, self->region, &mp);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkSnapToGrid Failed");
      Py_XDECREF(mapinfo);
      return NULL;
   }

   /* Copy data into MtkMapInfo */
   MtkMapInfo_copy(mapinfo,mp);

   return (PyObject*)mapinfo;
}

static PyGetSetDef Region_getseters[] = {
   {"center", (getter)Region_getcenter, (setter)Region_setcenter,
    "Center coordinate of the region in degrees.", NULL},
   {"extent", (getter)Region_getextent, (setter)Region_setextent,
    "Extent of the region in meters.", NULL},
   {"path_list", (getter)Region_getpath_list, (setter)Region_setpath_list,
    "List of paths that cover the region.", NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef Region_methods[] = {
   {"name", (PyCFunction)Region_name, METH_NOARGS,
    "Return the Ctr Lat, Ctr Lon, Ext xlat, Ext ylon"},
   {"block_range",
    (PyCFunction)Region_BlockRange, METH_VARARGS,
    "Return block range that covers the region for the given path."},
   {"snap_to_grid",
    (PyCFunction)Region_SnapToGrid, METH_VARARGS,
    "Snap a region to a MISR grid based on path number and resolution."},

   {NULL}  /* Sentinel */
};

PyTypeObject RegionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkRegion",   /*tp_name*/
    sizeof(Region),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Region_dealloc,/*tp_dealloc*/
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
    "MtkRegion objects",       /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Region_methods,            /* tp_methods */
    0,                         /* tp_members */
    Region_getseters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Region_init,     /* tp_init */
    0,                         /* tp_alloc */
    Region_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef region_methods[] = {
    {NULL}  /* Sentinel */
};
