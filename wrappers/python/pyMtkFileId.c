/*===========================================================================
=                                                                           =
=                               MtkFileId                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <Python.h>
#include "structmember.h"
#include "pyMtk.h"
#include <HdfEosDef.h>
#include <netcdf.h>

static void
MtkFileId_dealloc(MtkFileId* self)
{

  if (self->ncid > 0) {
    nc_close(self->ncid);
    self->ncid = 0;
  }

  /* Close file. */
  if (self->fid != FAIL) {
    GDclose(self->fid);
  }

  Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject *
MtkFileId_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
   MtkFileId *self;

   self = (MtkFileId *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
     self->fid = FAIL; 
     self->ncid = 0;
   }

   return (PyObject *)self;
}

int
file_id_init(MtkFileId *self, const char *filename)
{
  int32 fid;
  intn hdfstatus;		/* HDF return status */

  /* Try netCDF4 format */
  {
    int ncid;
    int status = nc_open(filename, NC_NOWRITE, &ncid);
    if (status == NC_NOERR) {  // success
      self->ncid = ncid;
      self->fid = FAIL;
      return 0; // success
    }
    self->ncid = 0;
  }

  /* Otherwise, try HDF-EOS2 format */
  fid = GDopen((char*)filename, DFACC_READ);
  if (fid == FAIL) {
    self->fid = FAIL;
    self->hdf_fid = FAIL;
    self->sid = FAIL;
    return -1;
  }
  self->fid = fid;

  hdfstatus = EHidinfo(fid, &(self->hdf_fid), &(self->sid));
  if (hdfstatus == FAIL) {
    GDclose(self->fid);
    self->fid = FAIL;
    self->hdf_fid = FAIL;
    self->sid = FAIL;
    return -1;
  }

  return 0;
}

PyTypeObject MtkFileIdType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "MisrToolkit.MtkFileId", /*tp_name*/
    sizeof(MtkFileId),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MtkFileId_dealloc,/*tp_dealloc*/
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
    "MtkFileId objects",     /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    MtkFileId_new,  /*PyType_GenericNew()*/              /* tp_new */
};

PyMethodDef mtkfileid_methods[] = {
    {NULL}  /* Sentinel */
};
