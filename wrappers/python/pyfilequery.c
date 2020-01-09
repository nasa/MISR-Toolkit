/*===========================================================================
=                                                                           =
=                               PyFileQuery                                 =
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

PyObject* FindFileList(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *searchdir;  /* Search Directory */
   char *product;    /* Product */
   PyObject *camera; /* Camera */
   char *path;       /* Path */
   char *orbit;      /* Orbit */
   char *version;    /* Version */
   int filecnt;      /* File count */
   char **filenames; /* Filenames */
   int i;

   if (!PyArg_ParseTuple(args,"ssOsss",&searchdir,&product,&camera,
                         &path,&orbit,&version))
      return NULL;

   if (camera == Py_None)
     status = MtkFindFileList(searchdir,product,NULL,path,orbit,
                              version,&filecnt,&filenames);
   else
     status = MtkFindFileList(searchdir,product,PyString_AsString(camera),
                              path,orbit,version,&filecnt,&filenames);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkFindFileList Failed");
      return NULL;
   }

   result = PyList_New(filecnt);
   for (i = 0; i < filecnt; ++i)
     PyList_SetItem(result, i, PyString_FromString(filenames[i]));

   MtkStringListFree(filecnt, &filenames);

   return result;
}

PyObject* MakeFilename(PyObject *self, PyObject *args)
{
   PyObject *result;
   MTKt_status status;
   char *basedir;  /* Base Directory */
   char *product;  /* Product */
   PyObject *camera;   /* Camera */
   int path;       /* Path */
   int orbit;      /* Orbit */
   char *version;  /* Version */
   char *filename; /* Filename */

   if (!PyArg_ParseTuple(args,"ssOiis",&basedir,&product,&camera,&path,
                         &orbit,&version))
      return NULL;

   if (camera == Py_None)
     status = MtkMakeFilename(basedir,product,NULL,path,orbit,
                              version,&filename);
   else
     status = MtkMakeFilename(basedir,product,PyString_AsString(camera),
                              path,orbit,version,&filename);
   if (status != MTK_SUCCESS)
   {
      PyErr_SetString(PyExc_StandardError, "MtkMakeFilename Failed");
      return NULL;
   }

   result = Py_BuildValue("s",filename);
   free(filename);
   return result;
}

PyMethodDef filequery_methods[] = {
   {"find_file_list", (PyCFunction)FindFileList, METH_VARARGS,
    "Find files in directory tree, using regular expressions."},
   {"make_filename", (PyCFunction)MakeFilename, METH_VARARGS,
    "Given a base directory, product, camera, path, orbit, version make file name."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};
