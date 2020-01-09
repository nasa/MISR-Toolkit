/*===========================================================================
=                                                                           =
=                                 MtkFile                                   =
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

extern PyTypeObject MtkGridType;
extern PyTypeObject MtkFileIdType;
MtkGrid* grid_init(MtkGrid *self, const char *filename, const char *gridname, MtkFileId *file_id);
int file_id_init(MtkFileId *self, const char *filename);

extern PyTypeObject MtkTimeMetaDataType;

static void
MtkFile_dealloc(MtkFile* self)
{
   int i;

   Py_CLEAR(self->filename);
   for (i = 0; i < self->num_grids; ++i) {
     Py_CLEAR(self->grids[i]);
   }
   Py_CLEAR(self->file_id);

   if (self->grids != NULL) {
     PyMem_Free(self->grids);
   }
   Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
MtkFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    MtkFile *self;
    static char *kwlist[] = {"filename", NULL};

    self = (MtkFile *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        #if PY_MAJOR_VERSION >= 3
            if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist,
                                             &self->filename)) {
        #else
            if (!PyArg_ParseTupleAndKeywords(args, kwds, "S", kwlist,
                                             &self->filename)) {
        #endif
                return NULL;
            }
      Py_INCREF(self->filename);
    }

    return (PyObject*)self;
}


static int
MtkFile_init(MtkFile *self, PyObject *args, PyObject *kwds)
{
   PyObject *filename = NULL, *tmp;
   MTKt_status status;
   char *file_name;
   int num_grids;
   char **gridlist;
   int i;

   static char *kwlist[] = {"filename", NULL};
   #if PY_MAJOR_VERSION >= 3
       if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", kwlist,
                                        &filename)) {
   #else
       if (!PyArg_ParseTupleAndKeywords(args, kwds, "S", kwlist,
                                        &filename)) {
   #endif
           return -1;
       }

   if (filename) {
      tmp = self->filename;
      Py_INCREF(filename);
      self->filename = filename;
      Py_XDECREF(tmp);
   }

   file_name = PyString_AsString(filename);

   self->file_id = PyObject_New(MtkFileId, &MtkFileIdType);

   status = file_id_init(self->file_id, file_name);
   if (status)  {
     PyErr_Format(PyExc_IOError, "Trouble opening file: %s", file_name);

     Py_CLEAR(self->filename);
     Py_CLEAR(self->file_id);
     return -1;
   }

   if (self->file_id->ncid > 0) {
     status = MtkFileToGridListNcid(self->file_id->ncid, &num_grids, &gridlist);
   } else {
     status = MtkFileToGridListFid(self->file_id->fid, &num_grids, &gridlist);
   }
   if (status != MTK_SUCCESS)
   {
       PyErr_Format(PyExc_IOError, "Trouble reading grid list: %s", file_name);
       Py_XDECREF(self->filename);
       self->filename = NULL;
       return -1;
   }

   /* Create MtkGrid Objects */
   self->grids = (MtkGrid**)PyMem_New(PyObject*, num_grids);
   self->num_grids = num_grids;

   for (i = 0; i < num_grids; ++i)
   {
      self->grids[i] = PyObject_New(MtkGrid, &MtkGridType);
      self->grids[i] = grid_init(self->grids[i], file_name, gridlist[i], self->file_id);
      if (self->grids[i] == NULL)
      {
         PyErr_Format(PyExc_StandardError, "Problem initializing Grid: %s", gridlist[i]);
         Py_CLEAR(self->filename);
         Py_CLEAR(self->file_id);
         MtkStringListFree(num_grids, &gridlist);
         return -1;
      }
   }

   MtkStringListFree(num_grids, &gridlist);

   return 0;
}

static PyObject *
MtkFile_getlocal_granule_id(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;    /* File name */
   char *lgid;        /* Local Granual ID */

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileLGIDNcid(self->file_id->ncid,&lgid);
   } else {
     status = MtkFileLGIDFid(self->file_id->sid,&lgid);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileLGID Failed");
      return NULL;
   }

   result = Py_BuildValue("s",lgid);
   free(lgid);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, local_granule_id)

static PyObject *
MtkFile_getblock(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;  /* File Name */
   int start_block; /* Start Block */
   int end_block;   /* End Block */

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileToBlockRangeNcid(self->file_id->ncid,&start_block,&end_block);
   } else {
     status = MtkFileToBlockRangeFid(self->file_id->sid,&start_block,&end_block);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileToBlockRange Failed");
      return NULL;
   }

   result = Py_BuildValue("ii",start_block,end_block);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, block)

static PyObject *
Grid(MtkFile *self, PyObject *args)
{
   char *gridname;
   int i;

   if (!PyArg_ParseTuple(args,"s",&gridname))
	return NULL;

   for (i = 0; i < self->num_grids; ++i)
     if (strcmp(gridname,PyString_AsString(self->grids[i]->gridname)) == 0)
     {
        Py_INCREF(self->grids[i]);
        return (PyObject*)self->grids[i];
     }

   PyErr_Format(PyExc_NameError, "Grid: %s Not Found.", gridname);
   return NULL;
}

static PyObject *
MtkFile_getgrid_list(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;  /* File name */
   int ngrids;      /* Number of grids */
   char **gridlist; /* Grid list */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileToGridListNcid(self->file_id->ncid,&ngrids,&gridlist);
   } else {
     status = MtkFileToGridListFid(self->file_id->fid,&ngrids,&gridlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileToGridList Failed");
      return NULL;
   }

   result = PyList_New(ngrids);
   for (i = 0; i < ngrids; ++i)
     PyList_SetItem(result, i, PyString_FromString(gridlist[i]));


   MtkStringListFree(ngrids, &gridlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, grid_list)

static PyObject *
MtkFile_getpath(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename; /* File name */
   int path;       /* Path number */

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileToPathNcid(self->file_id->ncid,&path);
   } else {
     status = MtkFileToPathFid(self->file_id->sid,&path);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileToPath Failed");
      return NULL;
   }

   result = Py_BuildValue("i",path);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, path)

static PyObject *
MtkFile_getorbit(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename; /* File name */
   int orbit;      /* Orbit number */

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileToOrbitNcid(self->file_id->ncid,&orbit);
   } else {
     status = MtkFileToOrbitFid(self->file_id->sid,&orbit);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileToOrbit Failed");
      return NULL;
   }

   result = Py_BuildValue("i",orbit);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, orbit)

static PyObject *
MtkFile_getfile_type(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename = NULL;
   MTKt_FileType filetype;
   char *types[] = MTKT_FILE_TYPE_DESC;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileTypeNcid(self->file_id->ncid, &filetype);
   } else {
     status = MtkFileTypeFid(self->file_id->fid, &filetype);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileType Failed");
      return NULL;
   }

   result = PyString_FromString(types[filetype]);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, file_type)

static PyObject *
MtkFile_getfile_name(MtkFile *self, void *closure)
{
  Py_INCREF(self->filename);
  return self->filename;
}

MTK_READ_ONLY_ATTR(MtkFile, file_name)

static PyObject *
MtkFile_getversion(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   char fileversion[10]; /* File version */

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileVersionNcid(self->file_id->ncid,fileversion);
   } else {
     status = MtkFileVersionFid(self->file_id->sid,fileversion);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileVersion Failed");
      return NULL;
   }

   result = Py_BuildValue("s",fileversion);
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, version)

static PyObject *
CoreMetaDataGet(MtkFile *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   char *paramname;      /* Parameter name */
   MtkCoreMetaData metadata;
   int i;

   if (!PyArg_ParseTuple(args,"s",&paramname))
     return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileCoreMetaDataGetNcid(self->file_id->ncid,paramname,&metadata);
   } else {
     status = MtkFileCoreMetaDataGetFid(self->file_id->sid,paramname,&metadata);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileCoreMetaDataGet Failed");
      return NULL;
   }

   switch (metadata.datatype)
   {
      case MTKMETA_CHAR :
         if (metadata.num_values > 1)
	 {
	    result = PyList_New(metadata.num_values);
	    for (i = 0; i < metadata.num_values; ++i)
	       PyList_SetItem(result, i, PyString_FromString(metadata.data.s[i]));
         }
         else
            result = Py_BuildValue("s",metadata.data.s[0]);
         break;
      case MTKMETA_INT :
         if (metadata.num_values > 1)
	 {
	    result = PyList_New(metadata.num_values);
	    for (i = 0; i < metadata.num_values; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(metadata.data.i[i]));
	 }
         else
            result = Py_BuildValue("i",metadata.data.i[0]);
         break;

      case MTKMETA_DOUBLE :
         if (metadata.num_values > 1)
	 {
	    result = PyList_New(metadata.num_values);
	    for (i = 0; i < metadata.num_values; ++i)
	       PyList_SetItem(result, i, PyFloat_FromDouble(metadata.data.d[i]));
	 }
         else
	   result = Py_BuildValue("f",metadata.data.d[0]);
         break;
      default : result = NULL;
         break;
  }

   MtkCoreMetaDataFree(&metadata);

   return result;
}

static PyObject *
MtkFile_getcore_metadata_list(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   int nparam;
   char **paramlist;
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileCoreMetaDataQueryNcid(self->file_id->ncid,&nparam,&paramlist);
   } else {
     status = MtkFileCoreMetaDataQueryFid(self->file_id->sid,&nparam,&paramlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileCoreMetaDataQuery Failed");
      return NULL;
   }

   result = PyList_New(nparam);
   for (i = 0; i < nparam; ++i)
      PyList_SetItem(result, i, PyString_FromString(paramlist[i]));

   MtkStringListFree(nparam,&paramlist);
        
   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, core_metadata_list)

static PyObject *
AttrGet(MtkFile *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   char *attrname;       /* Attribute name */
   MTKt_DataBuffer attrbuf;
   int i;

   if (!PyArg_ParseTuple(args,"s",&attrname))
     return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileAttrGetNcid(self->file_id->ncid,attrname,&attrbuf);
   } else {
     status = MtkFileAttrGetFid(self->file_id->sid,attrname,&attrbuf);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileAttrGet Failed");
      return NULL;
   }

   switch (attrbuf.datatype)
   {
      case MTKe_void : result = NULL;
	break;
      case  MTKe_char8 : result = Py_BuildValue("s",attrbuf.data.c8[0]);
	break;
      case  MTKe_uchar8 : result = Py_BuildValue("s",attrbuf.data.uc8[0]);
	break;
      case  MTKe_int8 :
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.i8[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.i8[0][0]);
	 break;
      case  MTKe_uint8 :
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.u8[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.u8[0][0]);
	 break;
      case  MTKe_int16 :
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.i16[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.i16[0][0]);
	break;
      case  MTKe_uint16 : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.u16[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.u16[0][0]);
	break;
      case  MTKe_int32 : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.i32[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.i32[0][0]);
	break;
      case  MTKe_uint32 : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong(attrbuf.data.u32[0][i]));
	 }
	 else
	    result = Py_BuildValue("i",attrbuf.data.u32[0][0]);
	break;
      case  MTKe_int64 : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong((long)attrbuf.data.i64[0][i]));
	 }
	 else
	    result = Py_BuildValue("l",attrbuf.data.i64[0][0]);
	break;
      case  MTKe_uint64 : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyInt_FromLong((long)attrbuf.data.u64[0][i]));
	 }
	 else
	    result = Py_BuildValue("l",attrbuf.data.u64[0][0]);
	break;
      case  MTKe_float : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyFloat_FromDouble(attrbuf.data.f[0][i]));
	 }
	 else
	    result = Py_BuildValue("f",attrbuf.data.f[0][0]);
	break;
      case  MTKe_double : 
	 if (attrbuf.nsample > 1)
	 {
	    result = PyList_New(attrbuf.nsample);
	    for (i = 0; i < attrbuf.nsample; ++i)
	       PyList_SetItem(result, i, PyFloat_FromDouble(attrbuf.data.d[0][i]));
	 }
	 else
	    result = Py_BuildValue("f",attrbuf.data.d[0][0]);
	break;
      default : result = NULL;
        break;
   }

   MtkDataBufferFree(&attrbuf);

   return result;
}

static PyObject *
MtkFile_getattr_list(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   int num_attrs;        /* Number of attributes */
   char **attrlist;      /* Attribute list */
   int i;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     status = MtkFileAttrListNcid(self->file_id->ncid,&num_attrs,&attrlist);
   } else {
     status = MtkFileAttrListFid(self->file_id->sid,&num_attrs,&attrlist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileAttrList Failed");
      return NULL;
   }

   result = PyList_New(num_attrs);
   for (i = 0; i < num_attrs; ++i)
     PyList_SetItem(result, i, PyString_FromString(attrlist[i]));

   MtkStringListFree(num_attrs, &attrlist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, attr_list)

static PyObject *
MtkFile_getblock_metadata_list(MtkFile *self, void *closure)
{
   PyObject *result;
   MTKt_status status;
   char *filename;        /* File name */
   int nblockmeta;        /* Number of block metadata structures */
   char **blockmetalist;  /* Block metadata list */
   int i;
   
   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     return NULL;  /* No block metadata in current netCDF products */
   } else {
     status = MtkFileBlockMetaListFid(self->file_id->hdf_fid, &nblockmeta, &blockmetalist);
   }
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFileBlockMetaList Failed");
      return NULL;
   }

   result = PyList_New(nblockmeta);
   for (i = 0; i < nblockmeta; ++i)
     PyList_SetItem(result, i, PyString_FromString(blockmetalist[i]));

   MtkStringListFree(nblockmeta, &blockmetalist);

   return result;
}

MTK_READ_ONLY_ATTR(MtkFile, block_metadata_list)

static PyObject *
BlockMetaFieldList(MtkFile *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   char *blockmetaname;  /* Block metadata structure name */
   int nfields;          /* Number of Fields */
   char **fieldlist;     /* List of Fields */
   int i;

   if (!PyArg_ParseTuple(args,"s",&blockmetaname))
      return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     return NULL;  /* No block metadata in current netCDF products */
   } else {
     status = MtkFileBlockMetaFieldListFid(self->file_id->hdf_fid, blockmetaname, &nfields,
                                           &fieldlist);
   }
   switch (status)
   {
      case MTK_SUCCESS :
         break;
      case MTK_HDF_VSFIND_FAILED : PyErr_Format(PyExc_NameError, "Structure: %s Not Found.", blockmetaname);
   	     return NULL;
         break;
      default : PyErr_SetString(PyExc_StandardError, "MtkFileBlockMetaFieldList Failed");
         return NULL;
         break;
   }

   result = PyList_New(nfields);
   for (i = 0; i < nfields; ++i)
     PyList_SetItem(result, i, PyString_FromString(fieldlist[i]));

   MtkStringListFree(nfields, &fieldlist);

   return result;
}

static PyObject *
BlockMetaFieldRead(MtkFile *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *filename;       /* File name */
   char *blockmetaname; /* Block metadata structure name */
   char *fieldname; /* Field name */
   int i;
   int j;
   PyObject *temp_list;
   MTKt_DataBuffer blockmetabuf;
   char datetime[MTKd_DATETIME_LEN];

   if (!PyArg_ParseTuple(args,"ss",&blockmetaname,&fieldname))
     return NULL;

   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     return NULL;  /* No block metadata in current netCDF products */
   } else {
     status = MtkFileBlockMetaFieldReadFid(self->file_id->hdf_fid, blockmetaname, fieldname, 
                                           &blockmetabuf);
   }
   switch (status)
   {
      case MTK_SUCCESS :
         break;
      case MTK_HDF_VSFIND_FAILED : PyErr_Format(PyExc_NameError, "Structure: %s Not Found.", blockmetaname);
   	     return NULL;
         break;
      case MTK_HDF_VSFINDEX_FAILED : PyErr_Format(PyExc_NameError, "Field: %s Not Found.", fieldname);
   	     return NULL;
         break;
   	  default : PyErr_SetString(PyExc_StandardError, "MtkFileBlockMetaFieldRead Failed");
         return NULL;
   	     break;
   }

   switch (blockmetabuf.datatype)
   {
      case MTKe_void : result = NULL;
	     break;
      case  MTKe_char8 :
         if (strcmp(blockmetaname, "PerBlockMetadataRad") == 0 &&
             strcmp(fieldname, "transform.ref_time") == 0)
         {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(2);

	           strncpy(datetime,blockmetabuf.data.c8[i],MTKd_DATETIME_LEN - 1);
	           datetime[MTKd_DATETIME_LEN - 1] = '\0';
	           PyList_SetItem(temp_list, 0, PyString_FromString(datetime));
	           strncpy(datetime,blockmetabuf.data.c8[i] + MTKd_DATETIME_LEN - 1,MTKd_DATETIME_LEN - 1);
	           datetime[MTKd_DATETIME_LEN - 1] = '\0';
	           PyList_SetItem(temp_list, 1, PyString_FromString(datetime));

	           PyList_SetItem(result, i, temp_list);
	        }
         }
         else
         {
            result = PyList_New(blockmetabuf.nline);
            for (i = 0; i < blockmetabuf.nline; ++i)
               PyList_SetItem(result, i, PyString_FromString(blockmetabuf.data.c8[i]));
         }
	     break;
      case  MTKe_uchar8 :
         if (strcmp(blockmetaname, "PerBlockMetadataRad") == 0 &&
             strcmp(fieldname, "transform.ref_time") == 0)
         {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(2);

	           strncpy(datetime,blockmetabuf.data.c8[i],MTKd_DATETIME_LEN - 1);
	           datetime[MTKd_DATETIME_LEN - 1] = '\0';
	           PyList_SetItem(temp_list, 0, PyString_FromString(datetime));
	           strncpy(datetime,blockmetabuf.data.c8[i] + MTKd_DATETIME_LEN - 1,MTKd_DATETIME_LEN - 1);
	           datetime[MTKd_DATETIME_LEN - 1] = '\0';
	           PyList_SetItem(temp_list, 1, PyString_FromString(datetime));

	           PyList_SetItem(result, i, temp_list);
	        }
         }
         else
         {
            result = PyList_New(blockmetabuf.nline);
            for (i = 0; i < blockmetabuf.nline; ++i)
               PyList_SetItem(result, i, PyString_FromString((char*)blockmetabuf.data.uc8[i]));
         }
	     break;
      case  MTKe_int8 :
	     if (blockmetabuf.nsample > 1)
	     {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.i8[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
	     }
	     else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.i8[i][0]));
	     }
	     break;
      case  MTKe_uint8 :
	     if (blockmetabuf.nsample > 1)
	     {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.u8[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
	     }
	     else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.u8[i][0]));
	     }
	     break;
      case  MTKe_int16 :
    	 if (blockmetabuf.nsample > 1)
    	 {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.i16[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
    	 {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.i16[i][0]));
    	 }
    	break;
      case  MTKe_uint16 : 
    	 if (blockmetabuf.nsample > 1)
    	 {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.u16[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
    	 {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.u16[i][0]));
    	 }
    	break;
      case  MTKe_int32 : 
    	 if (blockmetabuf.nsample > 1)
    	 {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.i32[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
    	 {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.i32[i][0]));
    	 }
    	break;
      case  MTKe_uint32 : 
	     if (blockmetabuf.nsample > 1)
	     {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong(blockmetabuf.data.u32[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
	     }
	     else
    	 {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong(blockmetabuf.data.u32[i][0]));
    	 }
	    break;
      case  MTKe_int64 : 
	     if (blockmetabuf.nsample > 1)
	     {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong((long)blockmetabuf.data.i64[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
	     }
	     else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong((long)blockmetabuf.data.i64[i][0]));
	     }
	     break;
      case  MTKe_uint64 : 
    	 if (blockmetabuf.nsample > 1)
    	 {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyInt_FromLong((unsigned long)blockmetabuf.data.u64[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyInt_FromLong((unsigned long)blockmetabuf.data.u64[i][0]));
	     }
     	 break;
      case  MTKe_float : 
    	 if (blockmetabuf.nsample > 1)
    	 {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyFloat_FromDouble(blockmetabuf.data.f[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyFloat_FromDouble(blockmetabuf.data.f[i][0]));
	     }
      	 break;
      case  MTKe_double : 
	     if (blockmetabuf.nsample > 1)
	     {
	     	result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	        {
	           temp_list = PyList_New(blockmetabuf.nsample);
	           for (j = 0; j < blockmetabuf.nsample; ++j)
	              PyList_SetItem(temp_list, j, PyFloat_FromDouble(blockmetabuf.data.d[i][j]));
	           PyList_SetItem(result, i, temp_list);
	        }
    	 }
    	 else
	     {
	        result = PyList_New(blockmetabuf.nline);
	        for (i = 0; i < blockmetabuf.nline; ++i)
	           PyList_SetItem(result, i, PyFloat_FromDouble(blockmetabuf.data.d[i][0]));
	     }
    	 break;
      default : result = NULL;
         break;
   }

   MtkDataBufferFree(&blockmetabuf);

   return result;
}

static PyObject *
TimeMetaRead(MtkFile *self, PyObject *args)
{
   MTKt_status status;
   char *filename;
   MtkTimeMetaData *tmd;
   
   tmd = PyObject_New(MtkTimeMetaData, &MtkTimeMetaDataType);
   
   filename = PyString_AsString(self->filename);
   if (filename == NULL)
     return NULL;

   if (self->file_id->ncid > 0) {
     return NULL;  /* Not available for netCDF */
   } else {
     status = MtkTimeMetaReadFid(self->file_id->hdf_fid, self->file_id->sid,  &tmd->time_metadata);
   }
   switch (status)
   {
   	  case MTK_SUCCESS :
   	     break;
      case MTK_HDF_VSFIND_FAILED :
         PyErr_SetString(PyExc_StandardError, "File doesn't contain time metadata.");
         return NULL;
   		 break;
      default :
         PyErr_SetString(PyExc_StandardError, "MtkTimeMetaRead Failed");
         return NULL;
         break;
   }
   
   return (PyObject*)tmd;
}

static PyGetSetDef MtkFile_getseters[] = {
   {"local_granule_id", (getter)MtkFile_getlocal_granule_id,
   	(setter)MtkFile_setlocal_granule_id,
    "Local granual ID of MISR product file.", NULL},
   {"block", (getter)MtkFile_getblock, (setter)MtkFile_setblock,
    "Start and end block numbers.", NULL},
   {"grid_list", (getter)MtkFile_getgrid_list, (setter)MtkFile_setgrid_list,
    "List of grid names.", NULL},
   {"path", (getter)MtkFile_getpath, (setter)MtkFile_setpath,
    "Path number.", NULL},
   {"orbit", (getter)MtkFile_getorbit, (setter)MtkFile_setorbit,
    "Orbit number.", NULL},
   {"file_type", (getter)MtkFile_getfile_type, (setter)MtkFile_setfile_type,
    "MISR product file type.", NULL},
   {"file_name", (getter)MtkFile_getfile_name, (setter)MtkFile_setfile_name,
    "File name of file.", NULL},
   {"version", (getter)MtkFile_getversion, (setter)MtkFile_setversion,
    "MISR product file version.", NULL},
   {"core_metadata_list", (getter)MtkFile_getcore_metadata_list,
   	(setter)MtkFile_setcore_metadata_list,
    "List of core metadata parameter names.", NULL},
   {"attr_list", (getter)MtkFile_getattr_list, (setter)MtkFile_setattr_list,
    "List of file attributes names.", NULL},
   {"block_metadata_list", (getter)MtkFile_getblock_metadata_list,
   	(setter)MtkFile_setblock_metadata_list,
    "List of block metadata structure names.", NULL},
   {NULL}  /* Sentinel */
};

static PyMethodDef MtkFile_methods[] = {
   {"grid", (PyCFunction)Grid, METH_VARARGS,
    "Return MtkGrid."},
   {"core_metadata_get", (PyCFunction)CoreMetaDataGet, METH_VARARGS,
    "Get core metadata parameter."},
   {"attr_get", (PyCFunction)AttrGet, METH_VARARGS,
    "Get a file attribute."},
   {"block_metadata_field_list", (PyCFunction)BlockMetaFieldList, METH_VARARGS,
    "List fields in a block metadata structure."},
   {"block_metadata_field_read", (PyCFunction)BlockMetaFieldRead, METH_VARARGS,
    "Read a block metadata field."},
   {"time_metadata_read", (PyCFunction)TimeMetaRead, METH_NOARGS,
    "Read time metadata in a Level 1B2 file."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyTypeObject pyMtkFileType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkFile",      /*tp_name*/
    sizeof(MtkFile),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkFile_dealloc,/*tp_dealloc*/
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
    "MtkFile objects",          /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MtkFile_methods,           /* tp_methods */
    0,                         /* tp_members */
    MtkFile_getseters,         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MtkFile_init,    /* tp_init */
    0,                         /* tp_alloc */
    MtkFile_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkfile_methods[] = {
    {NULL}  /* Sentinel */
};
