/*===========================================================================
=                                                                           =
=                                 idl_mtk                                   =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include "idl_export.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------- */
/* Region IDL structure definitions */
/* -------------------------------- */

static IDL_STRUCT_TAG_DEF region_s_tags[] = {
  { "GEO_CTR_LAT",  0, (void *) IDL_TYP_DOUBLE },
  { "GEO_CTR_LON",  0, (void *) IDL_TYP_DOUBLE },
  { "HEXTENT_XLAT", 0, (void *) IDL_TYP_DOUBLE },
  { "HEXTENT_YLON", 0, (void *) IDL_TYP_DOUBLE },
  { 0 }
};

static IDL_MEMINT pp_dims[] = { 1, 15 };
static IDL_MEMINT ulc_dims[] = { 1, 2 };
static IDL_MEMINT lrc_dims[] = { 1, 2 };
static IDL_MEMINT roff_dims[] = { 1, NBLOCK-1 };
static IDL_MEMINT blk1_dims[] = { 1, NBLOCK+1 };
static IDL_MEMINT substruct_dims[] = { 1, 1 };
static IDL_MEMINT camname_dims[] = { 1, 3 };
static IDL_MEMINT blk1_ngridcell_datetime_dims[] = { 3, NBLOCK+1, NGRIDCELL, MTKd_DATETIME_LEN };
static IDL_MEMINT blk1_ngridcell_dims[] = { 2, NBLOCK+1, NGRIDCELL };

static IDL_STRUCT_TAG_DEF mapinfo_s_tags[] = {
  { "PATH",                   0, (void *) IDL_TYP_LONG   },
  { "START_BLOCK",            0, (void *) IDL_TYP_LONG   },
  { "END_BLOCK",              0, (void *) IDL_TYP_LONG   },
  { "RESOLUTION",             0, (void *) IDL_TYP_LONG   },
  { "RESFACTOR",              0, (void *) IDL_TYP_LONG   },
  { "NLINE",                  0, (void *) IDL_TYP_LONG   },
  { "NSAMPLE",                0, (void *) IDL_TYP_LONG   },
  { "PIXEL_CENTER",           0, (void *) IDL_TYP_LONG   },
  { "SOM_PATH",               0, (void *) IDL_TYP_LONG   },
  { "SOM_ULC_X",              0, (void *) IDL_TYP_DOUBLE },
  { "SOM_ULC_Y",              0, (void *) IDL_TYP_DOUBLE },
  { "SOM_CTR_X",              0, (void *) IDL_TYP_DOUBLE },
  { "SOM_CTR_Y",              0, (void *) IDL_TYP_DOUBLE },
  { "SOM_LRC_X",              0, (void *) IDL_TYP_DOUBLE },
  { "SOM_LRC_Y",              0, (void *) IDL_TYP_DOUBLE },
  { "GEO_ULC_LAT",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_ULC_LON",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_URC_LAT",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_URC_LON",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_CTR_LAT",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_CTR_LON",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_LRC_LAT",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_LRC_LON",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_LLC_LAT",            0, (void *) IDL_TYP_DOUBLE },
  { "GEO_LLC_LON",            0, (void *) IDL_TYP_DOUBLE },
  { "PP_PATH",                0, (void *) IDL_TYP_LONG   },
  { "PP_PROJCODE",            0, (void *) IDL_TYP_LONG   },
  { "PP_ZONECODE",            0, (void *) IDL_TYP_LONG   },
  { "PP_SPHERECODE",          0, (void *) IDL_TYP_LONG   },
  { "PP_PROJPARAM",     pp_dims, (void *) IDL_TYP_DOUBLE },
  { "PP_ULC_BLOCK1",   ulc_dims, (void *) IDL_TYP_DOUBLE },
  { "PP_LRC_BLOCK1",   lrc_dims, (void *) IDL_TYP_DOUBLE },
  { "PP_NBLOCK",              0, (void *) IDL_TYP_LONG   },
  { "PP_NLINE",               0, (void *) IDL_TYP_LONG   },
  { "PP_NSAMPLE",             0, (void *) IDL_TYP_LONG   },
  { "PP_RELOFFSET",   roff_dims, (void *) IDL_TYP_FLOAT  },
  { "PP_RESOLUTION",          0, (void *) IDL_TYP_LONG   },
  { 0 }
};

static IDL_STRUCT_TAG_DEF projparam_s_tags[] = {
  { "PATH",                   0, (void *) IDL_TYP_LONG   },
  { "PROJCODE",               0, (void *) IDL_TYP_LONG64   },
  { "ZONECODE",               0, (void *) IDL_TYP_LONG64   },
  { "SPHERECODE",             0, (void *) IDL_TYP_LONG64   },
  { "PROJPARAM",        pp_dims, (void *) IDL_TYP_DOUBLE },
  { "ULC_BLOCK1",      ulc_dims, (void *) IDL_TYP_DOUBLE },
  { "LRC_BLOCK1",      lrc_dims, (void *) IDL_TYP_DOUBLE },
  { "NBLOCK",                 0, (void *) IDL_TYP_LONG   },
  { "NLINE",                  0, (void *) IDL_TYP_LONG   },
  { "NSAMPLE",                0, (void *) IDL_TYP_LONG   },
  { "RELOFFSET",      roff_dims, (void *) IDL_TYP_FLOAT  },
  { "RESOLUTION",             0, (void *) IDL_TYP_LONG   },
  { 0 }
};

static IDL_STRUCT_TAG_DEF geocoord_s_tags[] = {
  { "LAT",                    0, (void *) IDL_TYP_DOUBLE },
  { "LON",                    0, (void *) IDL_TYP_DOUBLE },
  { 0 }
};

static IDL_STRUCT_TAG_DEF geoblock_s_tags[] = {
  { "BLOCK_NUMBER",           0, (void *) IDL_TYP_LONG },
  { "ULC",       substruct_dims, (void *) IDL_TYP_STRUCT },
  { "URC",       substruct_dims, (void *) IDL_TYP_STRUCT },
  { "CTR",       substruct_dims, (void *) IDL_TYP_STRUCT },
  { "LRC",       substruct_dims, (void *) IDL_TYP_STRUCT },
  { "LLC",       substruct_dims, (void *) IDL_TYP_STRUCT },
  { 0 }
};

static IDL_STRUCT_TAG_DEF blockcorners_s_tags[] = {
  { "PATH",                   0, (void *) IDL_TYP_LONG   },
  { "START_BLOCK",            0, (void *) IDL_TYP_LONG   },
  { "END_BLOCK",              0, (void *) IDL_TYP_LONG   },
  { "BLOCK",          blk1_dims, (void *) IDL_TYP_STRUCT },
  { 0 }
};

static IDL_STRUCT_TAG_DEF time_metadata_s_tags[] = {
  { "PATH",                                    0, (void *) IDL_TYP_LONG   },
  { "START_BLOCK",                             0, (void *) IDL_TYP_LONG },
  { "END_BLOCK",                               0, (void *) IDL_TYP_LONG },
  { "CAMERA",                       camname_dims, (void *) IDL_TYP_BYTE },
  { "NUMBER_TRANSFORM",                blk1_dims, (void *) IDL_TYP_LONG },
  { "REF_TIME",     blk1_ngridcell_datetime_dims, (void *) IDL_TYP_BYTE },
  { "START_LINE",            blk1_ngridcell_dims, (void *) IDL_TYP_LONG },
  { "NUMBER_LINE",           blk1_ngridcell_dims, (void *) IDL_TYP_LONG },
  { "COEFF_LINE",            blk1_ngridcell_dims, (void *) IDL_TYP_DOUBLE },
  { "SOM_CTR_X",             blk1_ngridcell_dims, (void *) IDL_TYP_DOUBLE },
  { "SOM_CTR_Y",             blk1_ngridcell_dims, (void *) IDL_TYP_DOUBLE },
  { 0 }
};


/* ---------------------------------------------- */
/* IDL MisrToolkit Error Message Block and Macros */
/* ---------------------------------------------- */

static IDL_MSG_DEF msg_arr[] = {
  #define M_MTK_FUNC 0
  { "M_MTK_FUNC", "%N%s" }
};

static IDL_MSG_BLOCK msg_block;

char *mtk_errdesc[] = MTK_ERR_DESC;

#define MTK_ERR_IDL_COND_JUMP(status) \
  if (status != MTK_SUCCESS) { \
    IDL_MessageFromBlock(msg_block, M_MTK_FUNC, IDL_MSG_INFO, \
			 mtk_errdesc[status]); \
    goto ERROR_HANDLE; \
  }

#define MTK_ERR_IDL_LOG(status) \
  { \
    IDL_MessageFromBlock(msg_block, M_MTK_FUNC, IDL_MSG_INFO, \
			 mtk_errdesc[status]); \
  }

#define MTK_ERR_IDL_JUMP(status) \
  { \
    IDL_MessageFromBlock(msg_block, M_MTK_FUNC, IDL_MSG_INFO, \
			 mtk_errdesc[status]); \
    goto ERROR_HANDLE; \
  }

/* ------------------ */
/* IDL_Load Prototype */
/* ------------------ */

int IDL_Load( void );

/* --------------------------------------------------------------- */
/* C malloc free callback function                                 */
/* --------------------------------------------------------------- */

static void free_cb( UCHAR *p ) {
  free( (void *)p );
}

/* --------------------------------------------------------------- */
/* Mtk_MtkToIdlDatatype				                     		   */
/* --------------------------------------------------------------- */

MTKt_status Mtk_MtkToIdlDatatype( MTKt_DataType mtk_datatype, int *idl_datatype ) {

  MTKt_status status = MTK_FAILURE;

  switch (mtk_datatype) {
  case MTKe_char8:	*idl_datatype = IDL_TYP_BYTE;
    break;
  case MTKe_uchar8:	*idl_datatype = IDL_TYP_BYTE;
    break;
  case MTKe_int8:	*idl_datatype = IDL_TYP_BYTE;
    break;
  case MTKe_uint8:	*idl_datatype = IDL_TYP_BYTE;
    break;
  case MTKe_int16:	*idl_datatype = IDL_TYP_INT;
    break;
  case MTKe_uint16:	*idl_datatype = IDL_TYP_UINT;
    break;
  case MTKe_int32:	*idl_datatype = IDL_TYP_LONG;
    break;
  case MTKe_uint32:	*idl_datatype = IDL_TYP_ULONG;
    break;
  case MTKe_int64:	*idl_datatype = IDL_TYP_LONG64;
    break;
  case MTKe_uint64:	*idl_datatype = IDL_TYP_ULONG64;
    break;
  case MTKe_float:	*idl_datatype = IDL_TYP_FLOAT;
    break;
  case MTKe_double:	*idl_datatype = IDL_TYP_DOUBLE;
    break;
  default:
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
    break;
  }

  return(MTK_SUCCESS);
 ERROR_HANDLE:
  return (status);
}

/* --------------------------------------------------------------- */
/* Mtk_IdlToMTkDatatype					                           */
/* --------------------------------------------------------------- */

MTKt_status Mtk_IdlToMtkDatatype( int idl_datatype, MTKt_DataType *mtk_datatype ) {

  MTKt_status status = MTK_FAILURE;

  switch (idl_datatype) {
  case IDL_TYP_BYTE:	*mtk_datatype = MTKe_uint8;
    break;
  case IDL_TYP_INT:	*mtk_datatype = MTKe_int16;
    break;
  case IDL_TYP_UINT:	*mtk_datatype = MTKe_uint16;
    break;
  case IDL_TYP_LONG:	*mtk_datatype = MTKe_int32;
    break;
  case IDL_TYP_ULONG:	*mtk_datatype = MTKe_uint32;
    break;
  case IDL_TYP_LONG64:	*mtk_datatype = MTKe_int64;
    break;
  case IDL_TYP_ULONG64:	*mtk_datatype = MTKe_uint64;
    break;
  case IDL_TYP_FLOAT:	*mtk_datatype = MTKe_float;
    break;
  case IDL_TYP_DOUBLE:	*mtk_datatype = MTKe_double;
    break;
  default:
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
    break;
  }

  return(MTK_SUCCESS);
 ERROR_HANDLE:
  return (status);
}

/* --------------------------------------------------------------- */
/* Function to translate a MTKDataBuffer into an IDL data buffer   */
/* --------------------------------------------------------------- */

static MTKt_status Mtk_toIDLDataBuffer( MTKt_DataBuffer* srcbuf, IDL_VPTR *vsrcbuf) {
  MTKt_status status = MTK_FAILURE;
  int idl_datatype;
  IDL_MEMINT dim[2];
  /* Set idl dimenson */
  dim[0] = (*srcbuf).nsample;
  dim[1] = (*srcbuf).nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype((*srcbuf).datatype, &idl_datatype);
  MTK_ERR_IDL_COND_JUMP(status);

	/* Import array from C to IDL (does not copy the data) */
  *vsrcbuf = IDL_ImportArray(2, dim, idl_datatype, (UCHAR *)(*srcbuf).vdata[0],
                            free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free((*srcbuf).vdata);

  return status;

 ERROR_HANDLE:
  return status;
}

/* --------------------------------------------------------------- */
/* Function to translate a IDL data buffer into a MtkDataBuffer    */
/* --------------------------------------------------------------- */

static MTKt_status IDL_toMtkDataBuffer(IDL_VPTR *vsrcbuf, MTKt_DataBuffer *srcbuf) {
  int srcbuf_nline;
  int srcbuf_nsample;
  MTKt_DataType mtk_datatype;
  MTKt_status status = MTK_FAILURE;
  /* Check if the IDL src data buffer vsrcbuf is an array.
  Extract dimensions, datatype and pointer to data to create
  the srcbuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(*vsrcbuf);
  IDL_ENSURE_ARRAY(*vsrcbuf);
  if ((*vsrcbuf)->value.arr->n_dim != 2) {
    MTK_ERR_IDL_COND_JUMP(status);	
    status = MTK_FAILURE;
    return status;
  }
  srcbuf_nline = (*vsrcbuf)->value.arr->dim[1];
  srcbuf_nsample = (*vsrcbuf)->value.arr->dim[0];
  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype((*vsrcbuf)->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Allocate a data buffer */
  status = MtkDataBufferImport(srcbuf_nline, srcbuf_nsample, mtk_datatype,
                              (*vsrcbuf)->value.arr->data, srcbuf);
  return status;

 ERROR_HANDLE:
  return status;
}

/* --------------------------------------------------------------- */
/* Function to translate a C Reg Coeff Struct to an IDL Struct     */
/* --------------------------------------------------------------- */

static MTKt_status Mtk_toIDLRegCoeffStruct(MTKt_RegressionCoeff* reg_coeffp, IDL_VPTR *idl_out_struct ){
  MTKt_status status = MTK_FAILURE;
  void*         s;
  char*         s_data;
  IDL_MEMINT    tmp_dims[IDL_MAX_ARRAY_DIM];
  IDL_MEMINT    offset;
  int*          reg_valid_mask_ptr;
  float*       reg_slope_ptr;
  float*       reg_intercept_ptr;
  float*       reg_correlation_ptr;
  int           i;
  int           j;
  unsigned int  num_lines = reg_coeffp->valid_mask.nline;
  unsigned int  num_samples = reg_coeffp->valid_mask.nsample;
  static IDL_MEMINT regression_coeff_dims[3];
  static IDL_STRUCT_TAG_DEF regression_coeff_s_tags[] = {
    { "VALID_MASK",  regression_coeff_dims, (void *) IDL_TYP_LONG },
    { "SLOPE",       regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { "INTERCEPT",   regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { "CORRELATION", regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { 0 }
  };
  regression_coeff_dims[0] = 2;
  regression_coeff_dims[1] = num_samples;
  regression_coeff_dims[2] = num_lines;
  regression_coeff_s_tags[0].dims = regression_coeff_dims;
  regression_coeff_s_tags[1].dims = regression_coeff_dims;
  regression_coeff_s_tags[2].dims = regression_coeff_dims;
  regression_coeff_s_tags[3].dims = regression_coeff_dims;


  s = IDL_MakeStruct("regression_coeff", regression_coeff_s_tags);

  tmp_dims[0] = 1;
  tmp_dims[1] = num_samples;
  s_data = (char *)IDL_MakeTempStruct(s, 1, tmp_dims, idl_out_struct, 0);

  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "VALID_MASK", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_valid_mask_ptr = (int *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_valid_mask_ptr++) = reg_coeffp->valid_mask.data.u8[i][j];
		   
  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "SLOPE", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_slope_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_slope_ptr++) = reg_coeffp->slope.data.f[i][j];

  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "INTERCEPT", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_intercept_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_intercept_ptr++) = reg_coeffp->intercept.data.f[i][j];

  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "CORRELATION", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_correlation_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_correlation_ptr++) = reg_coeffp->correlation.data.f[i][j];
    
  status = MTK_SUCCESS;
  return status;

}

/* --------------------------------------------------------------------- */
/* Function to translate a C Reg Coeff Struct to a Second IDL Struct     */
/* --------------------------------------------------------------------- */

static MTKt_status Mtk_toIDLRegCoeffOutStruct(MTKt_RegressionCoeff* reg_coeffp, IDL_VPTR *idl_out_struct ){
  MTKt_status status = MTK_FAILURE;
  void*         s;
  char*         s_data;
  IDL_MEMINT    tmp_dims[IDL_MAX_ARRAY_DIM];
  IDL_MEMINT    offset;
  int*          reg_valid_mask_ptr;
  float*       reg_slope_ptr;
  float*       reg_intercept_ptr;
  float*       reg_correlation_ptr;
  int           i;
  int           j;
  unsigned int  num_lines = reg_coeffp->valid_mask.nline;
  unsigned int  num_samples = reg_coeffp->valid_mask.nsample;
  static IDL_MEMINT regression_coeff_dims[3];
  static IDL_STRUCT_TAG_DEF regression_coeff_out_s_tags[] = {
    { "VALID_MASK", regression_coeff_dims, (void *) IDL_TYP_LONG },
    { "SLOPE", regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { "INTERCEPT", regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { "CORRELATION", regression_coeff_dims, (void *) IDL_TYP_FLOAT },
    { 0 }
  };
  regression_coeff_dims[0] = 2;
  regression_coeff_dims[1] = num_samples;
  regression_coeff_dims[2] = num_lines;
  regression_coeff_out_s_tags[0].dims = regression_coeff_dims;
  regression_coeff_out_s_tags[1].dims = regression_coeff_dims;
  regression_coeff_out_s_tags[2].dims = regression_coeff_dims;
  regression_coeff_out_s_tags[3].dims = regression_coeff_dims;

  s = IDL_MakeStruct("regression_coeff_out", regression_coeff_out_s_tags);

  tmp_dims[0] = 1;
  tmp_dims[1] = num_samples;
  s_data = (char *)IDL_MakeTempStruct(s, 1, tmp_dims, idl_out_struct, 0);

  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "VALID_MASK", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_valid_mask_ptr = (int *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_valid_mask_ptr++) = reg_coeffp->valid_mask.data.u8[i][j];
		   
  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "SLOPE", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_slope_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_slope_ptr++) = reg_coeffp->slope.data.f[i][j];
	
  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "INTERCEPT", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_intercept_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_intercept_ptr++) = reg_coeffp->intercept.data.f[i][j];
	
  // Get the field of the structure:
  offset = IDL_StructTagInfoByName(s, "CORRELATION", IDL_MSG_LONGJMP, NULL);
  // Get a pointer to that location:
  reg_correlation_ptr = (float *)(s_data + offset);
  // Store values into array:
  for ( i = 0; i < num_lines; i++)
     for (j = 0; j< num_samples; j++)
       *(reg_correlation_ptr++) = reg_coeffp->correlation.data.f[i][j];
    
  status = MTK_SUCCESS;
  return status;

}

/* --------------------------------------------------------------- */
/* Function to translate a IDL Reg Coeff Struct to C Struct        */
/* --------------------------------------------------------------- */

static MTKt_status IDL_toMtkRegCoeffStruct(IDL_VPTR *vstructp, MTKt_RegressionCoeff* reg_coeffp) {
  MTKt_status status = MTK_FAILURE;
  MTKt_DataType mtk_datatype;
  int reg_coeff_nline;
  int reg_coeff_nsample;

  MTKt_DataBuffer reg_coeff_valid_mask = MTKT_DATABUFFER_INIT;
  IDL_VPTR vreg_coeff_valid_mask;

  MTKt_DataBuffer reg_coeff_slope = MTKT_DATABUFFER_INIT;
  IDL_VPTR vreg_coeff_slope;

  MTKt_DataBuffer reg_coeff_intercept = MTKT_DATABUFFER_INIT;
  IDL_VPTR vreg_coeff_intercept;

  MTKt_DataBuffer reg_coeff_correlation = MTKT_DATABUFFER_INIT;
  IDL_VPTR vreg_coeff_correlation;

  IDL_SREF sref;            /* IDL structure reference */
  unsigned char *sdata;     /* pointer to structure data */
  IDL_StructDefPtr sdef;    /* pointer to the structure definition */
  IDL_VPTR datav;           /* pointer to tag data variable */
  IDL_LONG toffset;         /* offset to tag data */
  int * mask_ptr;           /* pointer to structure tag data buffer (int) */
  float * float_mask_ptr; /* pointer to structure tag data buffer (float) */

  /* Get pointer to the structure data and get dimensions */
  sref = (*vstructp)[0].value.s;
  sdata = sref.arr->data;
  sdef = sref.sdef;
  vreg_coeff_valid_mask = &(sdef[0].tags[0].var);
  reg_coeff_nline = vreg_coeff_valid_mask->value.arr->dim[1];
  reg_coeff_nsample = vreg_coeff_valid_mask->value.arr->dim[0];
  status = MtkRegressionCoeffAllocate(reg_coeff_nline, reg_coeff_nsample, reg_coeffp);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Setup RegressionCoeff valid_mask */
  /* Check if the IDL data buffer is an array.
     Extract dimensions, datatype and pointer to data to create
     the MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vreg_coeff_valid_mask);
  IDL_ENSURE_ARRAY(vreg_coeff_valid_mask);
  if (vreg_coeff_valid_mask->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vreg_coeff_valid_mask->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Get the IDL structure tag info */
  toffset=IDL_StructTagInfoByIndex( sdef, 0, IDL_MSG_RET, &datav );
  mask_ptr = (int *)(sdata + toffset);
  /* Allocate a data buffer */
  status = MtkDataBufferImport(reg_coeff_nline, reg_coeff_nsample, mtk_datatype,
                               mask_ptr, &reg_coeff_valid_mask);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Set C structure data buffer to temp databuffer */
  reg_coeffp->valid_mask.data = reg_coeff_valid_mask.data;
  /* End Setup RegressionCoeff valid_mask */

  /* Setup RegressionCoeff slope */
  vreg_coeff_slope = &(sdef[0].tags[1].var);
  /* Check if the IDL data buffer is an array.
     Extract dimensions, datatype and pointer to data to create
     the MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vreg_coeff_slope);
  IDL_ENSURE_ARRAY(vreg_coeff_slope);
  if (vreg_coeff_slope->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vreg_coeff_slope->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Get the IDL structure tag info */
  toffset=IDL_StructTagInfoByIndex( sdef, 1, IDL_MSG_RET, &datav );
  float_mask_ptr = (float *)(sdata + toffset);
  /* Allocate a data buffer */
  status = MtkDataBufferImport(reg_coeff_nline, reg_coeff_nsample, mtk_datatype,
                               float_mask_ptr, &reg_coeff_slope);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Set C structure data buffer to temp databuffer */
  reg_coeffp->slope.data = reg_coeff_slope.data;
  /* End Setup RegressionCoeff slope */

  /* Setup RegressionCoeff intercept */
  vreg_coeff_intercept = &(sdef[0].tags[2].var);
  /* Check if the IDL data buffer is an array.
     Extract dimensions, datatype and pointer to data to create
     the MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vreg_coeff_intercept);
  IDL_ENSURE_ARRAY(vreg_coeff_intercept);
  if (vreg_coeff_intercept->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vreg_coeff_intercept->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Get the IDL structure tag info */
  toffset=IDL_StructTagInfoByIndex( sdef, 2, IDL_MSG_RET, &datav );
  float_mask_ptr = (float *)(sdata + toffset);
  /* Allocate a data buffer */
  status = MtkDataBufferImport(reg_coeff_nline, reg_coeff_nsample, mtk_datatype,
                               float_mask_ptr, &reg_coeff_intercept);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Set C structure data buffer to temp databuffer */
  reg_coeffp->intercept.data = reg_coeff_intercept.data;
  /* End Setup RegressionCoeff intercept */

  /* Setup RegressionCoeff correlation */
  vreg_coeff_correlation = &(sdef[0].tags[3].var);
  /* Check if the IDL data buffer is an array.
     Extract dimensions, datatype and pointer to data to create
     the MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vreg_coeff_correlation);
  IDL_ENSURE_ARRAY(vreg_coeff_correlation);
  if (vreg_coeff_correlation->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vreg_coeff_correlation->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Get the IDL structure tag info */
  toffset=IDL_StructTagInfoByIndex( sdef, 3, IDL_MSG_RET, &datav );
  float_mask_ptr = (float *)(sdata + toffset);
  /* Allocate a data buffer */
  status = MtkDataBufferImport(reg_coeff_nline, reg_coeff_nsample, mtk_datatype,
                               float_mask_ptr, &reg_coeff_correlation);
  MTK_ERR_IDL_COND_JUMP(status);
  /* Set C structure data buffer to temp databuffer */
  reg_coeffp->correlation.data = reg_coeff_correlation.data;
  /* End Setup RegressionCoeff correlation */
  return status;

 ERROR_HANDLE:
  return status;
}

/* --------------------------------------------------------------- */
/* Mtk_Version		                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Version( int argc, IDL_VPTR *argv ) {

  /* MISR Toolkit call */
  return IDL_StrToSTRING(MtkVersion());
}

/* --------------------------------------------------------------- */
/* Mtk_Path_To_ProjParam                                           */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Path_To_ProjParam( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[1] */
  IDL_VPTR vpath, vres;
  int *path, *res;
  IDL_MEMINT npath, nres;

  /* Output argv[2] */
  IDL_VPTR vprojparam;
  void *sprojparam;
  MTKt_MisrProjParam *projparam;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  /* Output */
  projparam = (MTKt_MisrProjParam *)malloc(sizeof(MTKt_MisrProjParam));
  sprojparam = IDL_MakeStruct(NULL, projparam_s_tags);
  vprojparam = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			       (UCHAR *)projparam, free_cb, sprojparam);

  /* MISR Toolkit call */
  status = MtkPathToProjParam( *path, *res, projparam );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_VarCopy(vprojparam, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vprojparam);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Path_BlockRange_To_BlockCorners                             */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Path_BlockRange_To_BlockCorners( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath, vsb, veb;
  int *path, *sb, *eb;
  IDL_MEMINT npath, nsb, neb;

  /* Output argv[3] */
  IDL_VPTR vbc;
  void *sbc;
  MTKt_BlockCorners *bc;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vsb = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vsb, &nsb, (char **)&sb, IDL_TRUE);

  veb = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(veb, &neb, (char **)&eb, IDL_TRUE);

  /* Output */
  sbc = IDL_MakeStruct(NULL, geocoord_s_tags);
  geoblock_s_tags[1].type = sbc;
  geoblock_s_tags[2].type = sbc;
  geoblock_s_tags[3].type = sbc;
  geoblock_s_tags[4].type = sbc;
  geoblock_s_tags[5].type = sbc;
  sbc = IDL_MakeStruct(NULL, geoblock_s_tags);
  blockcorners_s_tags[3].type = sbc;
  sbc = IDL_MakeStruct(NULL, blockcorners_s_tags);
  bc = (MTKt_BlockCorners *)malloc(sizeof(MTKt_BlockCorners));
  vbc = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			(UCHAR *)bc, free_cb, sbc);

  /* MISR Toolkit call */
  status = MtkPathBlockRangeToBlockCorners( *path, *sb, *eb, bc );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_DELTMP(vsb);
  IDL_DELTMP(veb);
  IDL_VarCopy(vbc, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath); 
  IDL_DELTMP(vsb);
  IDL_DELTMP(veb);
  IDL_DELTMP(vbc);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_To_Path                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_To_Path( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vpath = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileToPath( filename, path );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vpath, argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_To_Orbit                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_To_Orbit( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  IDL_VPTR vorbit;
  int *orbit;
  IDL_MEMINT norbit;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vorbit = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vorbit, &norbit, (char **)&orbit, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileToOrbit( filename, orbit );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vorbit, argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vorbit);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_To_BlockRange                                          */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_To_BlockRange( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] to argv[2] */
  IDL_VPTR vsb, veb;
  int *sb, *eb;
  IDL_MEMINT nsb, neb;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vsb = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vsb, &nsb, (char **)&sb, IDL_TRUE);

  veb = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(veb, &neb, (char **)&eb, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileToBlockRange( filename, sb, eb );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vsb, argv[1]);
  IDL_VarCopy(veb, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vsb);
  IDL_DELTMP(veb);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_To_GridList                                            */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_To_GridList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] to argv[2] */
  IDL_VPTR vgridcnt;
  int *gridcnt;
  IDL_MEMINT ngridcnt;

  IDL_VPTR vgridlist = NULL;
  IDL_STRING *data = NULL;
  char **gridlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vgridcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vgridcnt, &ngridcnt, (char **)&gridcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileToGridList( filename, gridcnt, &gridlist_tmp );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*gridcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*gridcnt,
					    IDL_ARR_INI_ZERO, &vgridlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *gridcnt; i++)
    IDL_StrStore(&data[i],gridlist_tmp[i]);

  /* Free temporary gridlist */
  MtkStringListFree(*gridcnt, &gridlist_tmp);

  IDL_VarCopy(vgridcnt, argv[1]);
  if (*gridcnt > 0) IDL_VarCopy(vgridlist, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*gridcnt > 0) IDL_DELTMP(vgridlist);
  IDL_DELTMP(vgridcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Grid_To_FieldList                                      */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_To_FieldList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *gridname;

  /* Output argv[2] to argv[3] */
  IDL_VPTR vfieldcnt;
  int *fieldcnt;
  IDL_MEMINT nfieldcnt;

  IDL_VPTR vfieldlist = NULL;
  IDL_STRING *data = NULL;
  char **fieldlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);

  /* Output */
  vfieldcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vfieldcnt, &nfieldcnt, (char **)&fieldcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileGridToFieldList( filename, gridname,
				   fieldcnt, &fieldlist_tmp );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*fieldcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*fieldcnt,
					    IDL_ARR_INI_ZERO, &vfieldlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *fieldcnt; i++)
    IDL_StrStore(&data[i],fieldlist_tmp[i]);

  /* Free temporary fieldlist */
  MtkStringListFree(*fieldcnt, &fieldlist_tmp);

  IDL_VarCopy(vfieldcnt, argv[2]);
  if (*fieldcnt > 0) IDL_VarCopy(vfieldlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*fieldcnt > 0) IDL_DELTMP(vfieldlist);
  IDL_DELTMP(vfieldcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Grid_To_Native_FieldList                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_To_Native_FieldList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *gridname;

  /* Output argv[2] to argv[3] */
  IDL_VPTR vfieldcnt;
  int *fieldcnt;
  IDL_MEMINT nfieldcnt;

  IDL_VPTR vfieldlist = NULL;
  IDL_STRING *data = NULL;
  char **fieldlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);

  /* Output */
  vfieldcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vfieldcnt, &nfieldcnt, (char **)&fieldcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileGridToNativeFieldList( filename, gridname,
					 fieldcnt, &fieldlist_tmp );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*fieldcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*fieldcnt,
					    IDL_ARR_INI_ZERO, &vfieldlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *fieldcnt; i++)
    IDL_StrStore(&data[i],fieldlist_tmp[i]);

  /* Free temporary fieldlist */
  MtkStringListFree(*fieldcnt, &fieldlist_tmp);

  IDL_VarCopy(vfieldcnt, argv[2]);
  if (*fieldcnt > 0) IDL_VarCopy(vfieldlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*fieldcnt > 0) IDL_DELTMP(vfieldlist);
  IDL_DELTMP(vfieldcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Grid_To_Resolution                                     */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_To_Resolution( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *gridname;

  /* Output argv[2] */
  IDL_VPTR vres;
  int *res;
  IDL_MEMINT nres;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);

  /* Output */
  vres = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileGridToResolution( filename, gridname, res );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vres, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vres);
  return IDL_GettmpLong(status);
}

/* ------------------------------------------------------------------- */
/* Mtk_File_Grid_Field_To_DimList                                      */
/* ------------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_Field_To_DimList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];
  int i;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *gridname;
  char *fieldname;

  /* Output argv[3] to argv[5] */
  IDL_VPTR vdimcnt, vdimsizelist = NULL;
  int *dimcnt, *dimsizelist;
  IDL_MEMINT ndimcnt;

  IDL_VPTR vdimlist = NULL;
  IDL_STRING *data = NULL;
  char **dimlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  /* Output */
  vdimcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vdimcnt, &ndimcnt, (char **)&dimcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileGridFieldToDimList( filename, gridname, fieldname,
				      dimcnt, &dimlist_tmp, &dimsizelist );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*dimcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*dimcnt,
					    IDL_ARR_INI_ZERO, &vdimlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *dimcnt; i++)
    IDL_StrStore(&data[i],dimlist_tmp[i]);

  /* Free temporary dimlist */
  MtkStringListFree(*dimcnt, &dimlist_tmp);

  /* Import array from C to IDL (does not copy the data) */
  if (*dimcnt > 0) {
    dim[0] = *dimcnt;
    vdimsizelist = IDL_ImportArray(1, dim, IDL_TYP_LONG,
				   (UCHAR *)dimsizelist, free_cb, 0);
  }

  IDL_VarCopy(vdimcnt, argv[3]);
  if (*dimcnt > 0) IDL_VarCopy(vdimlist, argv[4]);
  if (*dimcnt > 0) IDL_VarCopy(vdimsizelist, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*dimcnt > 0) IDL_DELTMP(vdimlist);
  if (*dimcnt > 0) IDL_DELTMP(vdimsizelist);
  IDL_DELTMP(vdimcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Grid_Field_To_Datatype                                 */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_Field_To_Datatype( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int datatype;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *gridname;
  char *fieldname;

  /* Output argv[3] */
  IDL_VPTR vdtype;
  MTKt_DataType dtype;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkFileGridFieldToDataType(filename, gridname, fieldname, &dtype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(dtype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Output */
  vdtype = IDL_GettmpLong((IDL_LONG)datatype);

  IDL_VarCopy(vdtype, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}
/* --------------------------------------------------------------- */
/* Mtk_File_Grid_Field_Check                                       */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Grid_Field_Check( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *gridname;
  char *fieldname;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkFileGridFieldCheck(filename, gridname, fieldname);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_LGID                                                   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_LGID( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  char *lgid;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* MISR Toolkit call */
  status = MtkFileLGID( filename, &lgid );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(IDL_StrToSTRING(lgid), argv[1]);
  free(lgid);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Version                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Version( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  char version[10];

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* MISR Toolkit call */
  status = MtkFileVersion( filename, version );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(IDL_StrToSTRING(version), argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Type                                                   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Type( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  MTKt_FileType filetype;
  char *filetype_str[] = MTKT_FILE_TYPE_DESC;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* MISR Toolkit call */
  status = MtkFileType( filename, &filetype );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(IDL_StrToSTRING(filetype_str[filetype]), argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Make_Filename                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Make_Filename( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[5] */
  char *basedir;
  char *product;
  char *camera;
  IDL_VPTR vpath, vorbit;
  int *path, *orbit;
  IDL_MEMINT npath, norbit;
  char *version;

  /* Output argv[6] */
  char *filename;

  /* Inputs */
  basedir = IDL_VarGetString(argv[0]);
  product = IDL_VarGetString(argv[1]);
  camera =  IDL_VarGetString(argv[2]);

  vpath = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vorbit = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_LONG);
  IDL_VarGetData(vorbit, &norbit, (char **)&orbit, IDL_TRUE);

  version = IDL_VarGetString(argv[5]);

  /* MISR Toolkit call */
  status = MtkMakeFilename( basedir, product, camera, *path, *orbit,
			    version, &filename );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_DELTMP(vorbit);
  IDL_VarCopy(IDL_StrToSTRING(filename), argv[6]);
  free(filename);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vorbit);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Find_FileList                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Find_FileList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] to argv[5] */
  char *searchdir;
  char *product;
  char *camera;
  char *path;
  char *orbit;
  char *version;

  /* Output argv[6] to argv[7] */
  IDL_VPTR vfilecnt;
  int *filecnt;
  IDL_MEMINT nfilecnt;

  IDL_VPTR vfilelist = NULL;
  IDL_STRING *data = NULL;
  char **filelist_tmp;

  /* Inputs */
  searchdir = IDL_VarGetString(argv[0]);
  product = IDL_VarGetString(argv[1]);
  camera =  IDL_VarGetString(argv[2]);
  path =    IDL_VarGetString(argv[3]);
  orbit =   IDL_VarGetString(argv[4]);
  version = IDL_VarGetString(argv[5]);

  /* Output */
  vfilecnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vfilecnt, &nfilecnt, (char **)&filecnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFindFileList( searchdir, product, camera, path, orbit,
			    version, filecnt, &filelist_tmp );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*filecnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*filecnt,
					    IDL_ARR_INI_ZERO, &vfilelist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *filecnt; i++)
    IDL_StrStore(&data[i],filelist_tmp[i]);

  /* Free temporary filelist */
  MtkStringListFree(*filecnt, &filelist_tmp);

  IDL_VarCopy(vfilecnt, argv[6]);
  if (*filecnt > 0) IDL_VarCopy(vfilelist, argv[7]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*filecnt > 0) IDL_DELTMP(vfilelist);
  IDL_DELTMP(vfilecnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LatLon_To_PathList                                          */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LatLon_To_PathList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];

  /* Input argv[0] to argv[1] */
  IDL_VPTR vlat, vlon;
  double *lat, *lon;
  IDL_MEMINT nlat, nlon;

  /* Output argv[2] to argv[3] */
  IDL_VPTR vpathcnt, vpathlist = NULL;
  int *pathcnt, *pathlist;
  IDL_MEMINT npathcnt;

  /* Inputs */
  vlat = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat, &nlat, (char **)&lat, IDL_TRUE);

  vlon = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon, &nlon, (char **)&lon, IDL_TRUE);

  /* Output */
  vpathcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vpathcnt, &npathcnt, (char **)&pathcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLatLonToPathList( *lat, *lon, pathcnt, &pathlist );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  if (*pathcnt != 0) {
    dim[0] = *pathcnt;
    vpathlist = IDL_ImportArray(1, dim, IDL_TYP_LONG,
				(UCHAR *)pathlist, free_cb, 0);
  }

  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_VarCopy(vpathcnt, argv[2]);
  if (*pathcnt != 0) IDL_VarCopy(vpathlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_DELTMP(vpathcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Region_To_PathList                                          */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Region_To_PathList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];

  /* Input argv[0] */
  MTKt_Region *region;
  IDL_MEMINT nregion;

  /* Output argv[1] to argv[2] */
  IDL_VPTR vpathcnt, vpathlist = NULL;
  int *pathcnt, *pathlist;
  IDL_MEMINT npathcnt;

  /* Inputs */
  IDL_VarGetData(argv[0], &nregion, (char **)&region, IDL_FALSE);

  /* Output */
  vpathcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vpathcnt, &npathcnt, (char **)&pathcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkRegionToPathList( *region, pathcnt, &pathlist );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  if (*pathcnt != 0) {
    dim[0] = *pathcnt;
    vpathlist = IDL_ImportArray(1, dim, IDL_TYP_LONG,
				(UCHAR *)pathlist, free_cb, 0);
  }

  IDL_VarCopy(vpathcnt, argv[1]);
  if (*pathcnt !=0) IDL_VarCopy(vpathlist, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpathcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Region_Path_To_BlockRange                                   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Region_Path_To_BlockRange( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[1] */
  MTKt_Region *region;
  IDL_MEMINT nregion;

  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  /* Output argv[2] to argv[3] */
  IDL_VPTR vsb, veb;
  int *sb, *eb;
  IDL_MEMINT nsb, neb;

  /* Inputs */
  IDL_VarGetData(argv[0], &nregion, (char **)&region, IDL_FALSE);

  vpath = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  /* Output */
  vsb = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vsb, &nsb, (char **)&sb, IDL_TRUE);

  veb = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(veb, &neb, (char **)&eb, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkRegionPathToBlockRange( *region, *path, sb, eb );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_VarCopy(vsb, argv[2]);
  IDL_VarCopy(veb, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vsb);
  IDL_DELTMP(veb);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Orbit_To_Path						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Orbit_To_Path( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vorbit;
  int *orbit;
  IDL_MEMINT norbit;

  /* Output argv[1] */
  IDL_VPTR vpath;
  int *path;

  /* Inputs */
  vorbit = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vorbit, &norbit, (char **)&orbit, IDL_TRUE);

  /* Output */
  path = (int *)IDL_VarMakeTempFromTemplate(vorbit, IDL_TYP_LONG, NULL,
					    &vpath, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < norbit; i++) {
    result = MtkOrbitToPath( orbit[i], &path[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vorbit);
  IDL_VarCopy(vpath, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Time_To_Orbit_Path					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Time_To_Orbit_Path( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *time;

  /* Output argv[1] to argv[2] */
  IDL_VPTR vorbit, vpath;
  int *orbit, *path;
  IDL_MEMINT norbit, npath;

  /* Inputs */
  time = IDL_VarGetString(argv[0]);

  /* Output */
  vorbit = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vorbit, &norbit, (char **)&orbit, IDL_TRUE);

  vpath = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkTimeToOrbitPath( time, orbit, path );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vorbit, argv[1]);
  IDL_VarCopy(vpath, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vorbit);
  IDL_DELTMP(vpath);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Path_TimeRange_To_OrbitList		      		   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Path_TimeRange_To_OrbitList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  char *start_time;
  char *end_time;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vorbitcnt, vorbitlist = NULL;
  int *orbitcnt, *orbitlist;
  IDL_MEMINT norbitcnt;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  start_time = IDL_VarGetString(argv[1]);
  end_time = IDL_VarGetString(argv[2]);

  /* Output */
  vorbitcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vorbitcnt, &norbitcnt, (char **)&orbitcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkPathTimeRangeToOrbitList( *path, start_time, end_time,
					orbitcnt, &orbitlist );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  if (*orbitcnt != 0) { 
    dim[0] = *orbitcnt;
    vorbitlist = IDL_ImportArray(1, dim, IDL_TYP_LONG,
				 (UCHAR *)orbitlist, free_cb, 0);
  }

  IDL_DELTMP(vpath);
  IDL_VarCopy(vorbitcnt, argv[3]);
  if (*orbitcnt != 0) IDL_VarCopy(vorbitlist, argv[4]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vorbitcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Orbit_To_TimeRange					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Orbit_To_TimeRange( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  /* Output argv[1] to argv[2] */
  char start_time[MTKd_DATETIME_LEN];
  char end_time[MTKd_DATETIME_LEN];

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkOrbitToTimeRange( *path, start_time, end_time );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_VarCopy(IDL_StrToSTRING(start_time), argv[1]);
  IDL_VarCopy(IDL_StrToSTRING(end_time), argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* MtkTimeRange_To_OrbitList					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_TimeRange_To_OrbitList( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];

  /* Input argv[0] to argv[1] */
  char *start_time;
  char *end_time;

  /* Output argv[2] to argv[3] */
  IDL_VPTR vorbitcnt, vorbitlist = NULL;
  int *orbitcnt, *orbitlist;
  IDL_MEMINT norbitcnt;

  /* Inputs */
  start_time = IDL_VarGetString(argv[0]);
  end_time = IDL_VarGetString(argv[1]);

  /* Output */
  vorbitcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vorbitcnt, &norbitcnt, (char **)&orbitcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkTimeRangeToOrbitList( start_time, end_time,
				    orbitcnt, &orbitlist );
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  if (*orbitcnt != 0) {
    dim[0] = *orbitcnt;
    vorbitlist = IDL_ImportArray(1, dim, IDL_TYP_LONG,
				 (UCHAR *)orbitlist, free_cb, 0);
  }

  IDL_VarCopy(vorbitcnt, argv[2]);
  if (*orbitcnt != 0) IDL_VarCopy(vorbitlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vorbitcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LatLon_To_Bls                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LatLon_To_Bls( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[3] */
  IDL_VPTR vpath, vres;
  int *path, *res;
  IDL_MEMINT npath, nres;

  IDL_VPTR vlat, vlon;
  double *lat, *lon;
  IDL_MEMINT nlat, nlon;

  /* Output argv[4] to argv[6] */
  IDL_VPTR vblock;
  int *block;

  IDL_VPTR vline, vsample;
  float *line, *sample;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  vlat = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat, &nlat, (char **)&lat, IDL_TRUE);

  vlon = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon, &nlon, (char **)&lon, IDL_TRUE);

  /* Output */
  block = (int *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_LONG, NULL,
					     &vblock, IDL_TRUE);
  line = (float *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_FLOAT, NULL,
					     &vline, IDL_TRUE);
  sample = (float *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_FLOAT, NULL,
					     &vsample, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLatLonToBlsAry( *path, *res, nlat, lat, lon,
			      block, line, sample );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_VarCopy(vblock, argv[4]);
  IDL_VarCopy(vline, argv[5]);
  IDL_VarCopy(vsample, argv[6]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LatLon_To_SomXY                                             */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LatLon_To_SomXY( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  IDL_VPTR vlat, vlon;
  double *lat, *lon;
  IDL_MEMINT nlat, nlon;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vlat = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat, &nlat, (char **)&lat, IDL_TRUE);

  vlon = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon, &nlon, (char **)&lon, IDL_TRUE);

  /* Output */
  som_x = (double *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_DOUBLE, NULL,
						&vsom_x, IDL_TRUE);
  som_y = (double *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_DOUBLE, NULL,
						&vsom_y, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLatLonToSomXYAry( *path, nlat, lat, lon, som_x, som_y );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_VarCopy(vsom_x, argv[3]);
  IDL_VarCopy(vsom_y, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SomXY_To_Bls                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SomXY_To_Bls( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[3] */
  IDL_VPTR vpath, vres;
  int *path, *res;
  IDL_MEMINT npath, nres;

  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;
  IDL_MEMINT nsom_x, nsom_y;

  /* Output argv[4] to argv[6] */
  IDL_VPTR vblock;
  int *block;

  IDL_VPTR vline, vsample;
  float *line, *sample;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  vsom_x = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_x, &nsom_x, (char **)&som_x, IDL_TRUE);

  vsom_y = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_y, &nsom_y, (char **)&som_y, IDL_TRUE);

  /* Output */
  block = (int *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_LONG, NULL,
					     &vblock, IDL_TRUE);
  line = (float *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_FLOAT, NULL,
					      &vline, IDL_TRUE);
  sample = (float *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_FLOAT, NULL,
						&vsample, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkSomXYToBlsAry( *path, *res, nsom_x, som_x, som_y,
			     block, line, sample );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vsom_x);
  IDL_DELTMP(vsom_y);
  IDL_VarCopy(vblock, argv[4]);
  IDL_VarCopy(vline, argv[5]);
  IDL_VarCopy(vsample, argv[6]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Bls_To_LatLon                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Bls_To_LatLon( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[4] */
  IDL_VPTR vpath, vres, vblock;
  int *path, *res, *block;
  IDL_MEMINT npath, nres, nblock;

  IDL_VPTR vline, vsample;
  float *line, *sample;
  IDL_MEMINT nline, nsample;

  /* Output argv[5] to argv[6] */
  IDL_VPTR vlat, vlon;
  double *lat, *lon;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  vblock = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(vblock, &nblock, (char **)&block, IDL_TRUE);

  vline = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_FLOAT);
  IDL_VarGetData(vline, &nline, (char **)&line, IDL_TRUE);

  vsample = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_FLOAT);
  IDL_VarGetData(vsample, &nsample, (char **)&sample, IDL_TRUE);

  /* Output */
  lat = (double *)IDL_VarMakeTempFromTemplate(vblock, IDL_TYP_DOUBLE, NULL,
					      &vlat, IDL_TRUE);
  lon = (double *)IDL_VarMakeTempFromTemplate(vblock, IDL_TYP_DOUBLE, NULL,
					      &vlon, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkBlsToLatLonAry( *path, *res, nblock, block, line, sample,
			      lat, lon );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vblock);
  IDL_DELTMP(vline);
  IDL_DELTMP(vsample);
  IDL_VarCopy(vlat, argv[5]);
  IDL_VarCopy(vlon, argv[6]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Bls_To_SomXY                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Bls_To_SomXY( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[4] */
  IDL_VPTR vpath, vres, vblock;
  int *path, *res, *block;
  IDL_MEMINT npath, nres, nblock;

  IDL_VPTR vline, vsample;
  float *line, *sample;
  IDL_MEMINT nline, nsample;

  /* Output argv[5] to argv[6] */
  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  vblock = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(vblock, &nblock, (char **)&block, IDL_TRUE);

  vline = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_FLOAT);
  IDL_VarGetData(vline, &nline, (char **)&line, IDL_TRUE);

  vsample = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_FLOAT);
  IDL_VarGetData(vsample, &nsample, (char **)&sample, IDL_TRUE);

  /* Output */
  som_x = (double *)IDL_VarMakeTempFromTemplate(vblock, IDL_TYP_DOUBLE, NULL,
					      &vsom_x, IDL_TRUE);
  som_y = (double *)IDL_VarMakeTempFromTemplate(vblock, IDL_TYP_DOUBLE, NULL,
					      &vsom_y, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkBlsToSomXYAry( *path, *res, nblock, block, line, sample,
			     som_x, som_y );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vblock);
  IDL_DELTMP(vline);
  IDL_DELTMP(vsample);
  IDL_VarCopy(vsom_x, argv[5]);
  IDL_VarCopy(vsom_y, argv[6]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SomXY_To_LatLon                                             */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SomXY_To_LatLon( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;

  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;
  IDL_MEMINT nsom_x, nsom_y;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vlat, vlon;
  double *lat, *lon;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vsom_x = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_x, &nsom_x, (char **)&som_x, IDL_TRUE);

  vsom_y = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_y, &nsom_y, (char **)&som_y, IDL_TRUE);

  /* Output */
  lat = (double *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_DOUBLE, NULL,
					      &vlat, IDL_TRUE);
  lon = (double *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_DOUBLE, NULL,
					      &vlon, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkSomXYToLatLonAry( *path, nsom_x, som_x, som_y, lat, lon );

  IDL_DELTMP(vpath);
  IDL_DELTMP(vsom_x);
  IDL_DELTMP(vsom_y);
  IDL_VarCopy(vlat, argv[3]);
  IDL_VarCopy(vlon, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SetRegion_By_Ulc_Lrc                                        */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SetRegion_By_Ulc_Lrc( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[3] */
  IDL_VPTR vulc_lat, vulc_lon;
  double *ulc_lat, *ulc_lon;
  IDL_MEMINT nulc_lat, nulc_lon;

  IDL_VPTR vlrc_lat, vlrc_lon;
  double *lrc_lat, *lrc_lon;
  IDL_MEMINT nlrc_lat, nlrc_lon;

  /* Output argv[4] */
  IDL_VPTR vregion;
  void *sregion;
  MTKt_Region *region;

  /* Inputs */
  vulc_lat = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lat, &nulc_lat, (char **)&ulc_lat, IDL_TRUE);

  vulc_lon = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lon, &nulc_lon, (char **)&ulc_lon, IDL_TRUE);

  vlrc_lat = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lat, &nlrc_lat, (char **)&lrc_lat, IDL_TRUE);

  vlrc_lon = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lon, &nlrc_lon, (char **)&lrc_lon, IDL_TRUE);

  /* Output */
  region = (MTKt_Region *)malloc(sizeof(MTKt_Region));
  sregion = IDL_MakeStruct(NULL, region_s_tags);
  vregion = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			    (UCHAR *)region, free_cb, sregion);

  /* MISR Toolkit call */
  status = MtkSetRegionByUlcLrc( *ulc_lat, *ulc_lon,
				 *lrc_lat, *lrc_lon, region );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vulc_lat);
  IDL_DELTMP(vulc_lon);
  IDL_DELTMP(vlrc_lat);
  IDL_DELTMP(vlrc_lon);
  IDL_VarCopy(vregion, argv[4]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vulc_lat);
  IDL_DELTMP(vulc_lon);
  IDL_DELTMP(vlrc_lat);
  IDL_DELTMP(vlrc_lon);
  IDL_DELTMP(vregion);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SetRegion_By_Path_BlockRange                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SetRegion_By_Path_BlockRange( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath, vsblk, veblk;
  int *path, *sblk, *eblk;
  IDL_MEMINT npath, nsblk, neblk;

  /* Output argv[3] */
  IDL_VPTR vregion;
  void *sregion;
  MTKt_Region *region;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vsblk = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vsblk, &nsblk, (char **)&sblk, IDL_TRUE);

  veblk = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(veblk, &neblk, (char **)&eblk, IDL_TRUE);

  /* Output */
  region = (MTKt_Region *)malloc(sizeof(MTKt_Region));
  sregion = IDL_MakeStruct(NULL, region_s_tags);
  vregion = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			    (UCHAR *)region, free_cb, sregion);

  /* MISR Toolkit call */
  status = MtkSetRegionByPathBlockRange( *path, *sblk, *eblk, region );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_DELTMP(vsblk);
  IDL_DELTMP(veblk);
  IDL_VarCopy(vregion, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vsblk);
  IDL_DELTMP(veblk);
  IDL_DELTMP(vregion);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SetRegion_By_LatLon_Extent                                  */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SetRegion_By_LatLon_Extent( int argc,
						IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[4] */
  IDL_VPTR vlat, vlon;
  double *lat, *lon;
  IDL_MEMINT nlat, nlon;

  IDL_VPTR vlat_extent, vlon_extent;
  double *lat_extent, *lon_extent;
  IDL_MEMINT nlat_extent, nlon_extent;

  char *extent_units;

  /* Output argv[5] */
  IDL_VPTR vregion;
  void *sregion;
  MTKt_Region *region;

  /* Inputs */
  vlat = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat, &nlat, (char **)&lat, IDL_TRUE);

  vlon = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon, &nlon, (char **)&lon, IDL_TRUE);

  vlat_extent = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat_extent, &nlat_extent, (char **)&lat_extent, IDL_TRUE);

  vlon_extent = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon_extent, &nlon_extent, (char **)&lon_extent, IDL_TRUE);

  extent_units = IDL_VarGetString(argv[4]);

  /* Output */
  region = (MTKt_Region *)malloc(sizeof(MTKt_Region));
  sregion = IDL_MakeStruct(NULL, region_s_tags);
  vregion = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			    (UCHAR *)region, free_cb, sregion);

  /* MISR Toolkit call */
  status = MtkSetRegionByLatLonExtent( *lat, *lon,
				       *lat_extent, *lon_extent,
				       (const char *)extent_units,
				       region );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_DELTMP(vlat_extent);
  IDL_DELTMP(vlon_extent);
  IDL_VarCopy(vregion, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_DELTMP(vlat_extent);
  IDL_DELTMP(vlon_extent);
  IDL_DELTMP(vregion);
  return IDL_GettmpLong(status);
}


/* --------------------------------------------------------------- */
/* Mtk_SetRegion_By_Path_Som_Ulc_Lrc                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SetRegion_By_Path_Som_Ulc_Lrc( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[4] */
  IDL_VPTR vpath;
  int *path;
  IDL_MEMINT npath;  
  IDL_VPTR vulc_lat, vulc_lon;
  double *ulc_lat, *ulc_lon;
  IDL_MEMINT nulc_lat, nulc_lon;

  IDL_VPTR vlrc_lat, vlrc_lon;
  double *lrc_lat, *lrc_lon;
  IDL_MEMINT nlrc_lat, nlrc_lon;

  /* Output argv[4] */
  IDL_VPTR vregion;
  void *sregion;
  MTKt_Region *region;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);
  
  vulc_lat = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lat, &nulc_lat, (char **)&ulc_lat, IDL_TRUE);

  vulc_lon = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lon, &nulc_lon, (char **)&ulc_lon, IDL_TRUE);

  vlrc_lat = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lat, &nlrc_lat, (char **)&lrc_lat, IDL_TRUE);

  vlrc_lon = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lon, &nlrc_lon, (char **)&lrc_lon, IDL_TRUE);

  /* Output */
  region = (MTKt_Region *)malloc(sizeof(MTKt_Region));
  sregion = IDL_MakeStruct(NULL, region_s_tags);
  vregion = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			    (UCHAR *)region, free_cb, sregion);

  /* MISR Toolkit call */
  status = MtkSetRegionByPathSomUlcLrc( *path, *ulc_lat, *ulc_lon,
				 *lrc_lat, *lrc_lon, region );
  MTK_ERR_IDL_COND_JUMP(status);
  
  IDL_DELTMP(vpath);
  IDL_DELTMP(vulc_lat);
  IDL_DELTMP(vulc_lon);
  IDL_DELTMP(vlrc_lat);
  IDL_DELTMP(vlrc_lon);
  IDL_VarCopy(vregion, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vulc_lat);
  IDL_DELTMP(vulc_lon);
  IDL_DELTMP(vlrc_lat);
  IDL_DELTMP(vlrc_lon);
  IDL_DELTMP(vregion);
  return IDL_GettmpLong(status);
}



/* --------------------------------------------------------------- */
/* Mtk_Snap_To_Grid						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Snap_To_Grid( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] to argv[2] */
  IDL_VPTR vpath, vres;
  int *path, *res;
  IDL_MEMINT npath, nres;

  MTKt_Region *region;
  IDL_MEMINT nregion;

  /* Output argv[3] */
  IDL_VPTR vmapinfo;
  void *smapinfo;
  MTKt_MapInfo *mapinfo;

  /* Inputs */
  vpath = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vpath, &npath, (char **)&path, IDL_TRUE);

  vres = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vres, &nres, (char **)&res, IDL_TRUE);

  IDL_VarGetData(argv[2], &nregion, (char **)&region, IDL_FALSE);

  /* Outputs */
  mapinfo = (MTKt_MapInfo *)malloc(sizeof(MTKt_MapInfo));
  smapinfo = IDL_MakeStruct(NULL, mapinfo_s_tags);
  vmapinfo = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			     (UCHAR *)mapinfo, free_cb, smapinfo);

  /* MISR Toolkit call */
  status = MtkSnapToGrid(*path, *res, *region, mapinfo);
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_VarCopy(vmapinfo, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vpath);
  IDL_DELTMP(vres);
  IDL_DELTMP(vmapinfo);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ReadBlock                                                   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ReadBlock( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[3] */
  char *filename, *gridname, *fieldname;
  IDL_VPTR vblock;
  int *block;
  IDL_MEMINT nblock;

  /* Output argv[4] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  vblock = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_LONG);
  IDL_VarGetData(vblock, &nblock, (char **)&block, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkReadBlock(filename, gridname, fieldname, *block, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  IDL_DELTMP(vblock);
  IDL_VarCopy(vdatabuf, argv[4]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vblock)
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ReadBlockRange                                              */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ReadBlockRange( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[3];
  int datatype;

  /* Input argv[0] to argv[4] */
  char *filename, *gridname, *fieldname;
  IDL_VPTR vstartblock, vendblock;
  int *startblock, *endblock;
  IDL_MEMINT nstartblock, nendblock;

  /* Output argv[5] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer3D databuf = MTKT_DATABUFFER3D_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  vstartblock = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_LONG);
  IDL_VarGetData(vstartblock, &nstartblock, (char **)&startblock, IDL_TRUE);

  vendblock = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_LONG);
  IDL_VarGetData(vendblock, &nendblock, (char **)&endblock, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkReadBlockRange(filename, gridname, fieldname,
			     *startblock, *endblock, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;
  dim[2] = databuf.nblock;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(3, dim, datatype, (UCHAR *)databuf.vdata[0][0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  if (databuf.vdata != NULL) {
    if (databuf.vdata[0] != NULL) {
      free(databuf.vdata[0]);
    }
    free(databuf.vdata);
  }

  IDL_DELTMP(vstartblock);
  IDL_DELTMP(vendblock);
  IDL_VarCopy(vdatabuf, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vstartblock)
  IDL_DELTMP(vendblock)
  MtkDataBufferFree3D(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ReadData                                                    */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ReadData( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim1[1] = { 1 };
  IDL_MEMINT dim2[2];
  int datatype;

  /* Input argv[0] to argv[3] */
  char *filename, *gridname, *fieldname;
  MTKt_Region *region;
  IDL_MEMINT nregion;

  /* Output argv[4] and optionally argv[5] (mapinfo) */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  IDL_VPTR vmapinfo = NULL;
  void *smapinfo;
  MTKt_MapInfo *mapinfo;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);
  IDL_VarGetData(argv[3], &nregion, (char **)&region, IDL_FALSE);

  /* Outputs */
  mapinfo = (MTKt_MapInfo *)malloc(sizeof(MTKt_MapInfo));
  if (argc == 6) {
    smapinfo = IDL_MakeStruct(NULL, mapinfo_s_tags);
    vmapinfo = IDL_ImportArray(1, dim1, IDL_TYP_STRUCT, 
			       (UCHAR *)mapinfo, free_cb, smapinfo);
  }

  /* MISR Toolkit call */
  status = MtkReadData(filename, gridname, fieldname, *region,
		       &databuf, mapinfo);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim2[0] = databuf.nsample;
  dim2[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim2, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  /* Free mapinfo if not passing back */
  if (argc == 5) free(mapinfo);

  IDL_VarCopy(vdatabuf, argv[4]);
  if (argc == 6) IDL_VarCopy(vmapinfo, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  if (argc == 6) IDL_DELTMP(vmapinfo);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ReadRaw                                                     */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ReadRaw( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim1[1] = { 1 };
  IDL_MEMINT dim2[2];
  int datatype;

  /* Input argv[0] to argv[3] */
  char *filename, *gridname, *fieldname;
  MTKt_Region *region;
  IDL_MEMINT nregion;

  /* Output argv[4] and optionally argv[5] (mapinfo) */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  IDL_VPTR vmapinfo = NULL;
  void *smapinfo;
  MTKt_MapInfo *mapinfo;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);
  IDL_VarGetData(argv[3], &nregion, (char **)&region, IDL_FALSE);

  /* Outputs */
  mapinfo = (MTKt_MapInfo *)malloc(sizeof(MTKt_MapInfo));
  if (argc == 6) {
    smapinfo = IDL_MakeStruct(NULL, mapinfo_s_tags);
    vmapinfo = IDL_ImportArray(1, dim1, IDL_TYP_STRUCT, 
			       (UCHAR *)mapinfo, free_cb, smapinfo);
  }

  /* MISR Toolkit call */
  status = MtkReadRaw(filename, gridname, fieldname, *region,
		      &databuf, mapinfo);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim2[0] = databuf.nsample;
  dim2[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim2, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  /* Free mapinfo if not passing back */
  if (argc == 5) free(mapinfo);

  IDL_VarCopy(vdatabuf, argv[4]);
  if (argc == 6) IDL_VarCopy(vmapinfo, argv[5]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  if (argc == 6) IDL_DELTMP(vmapinfo);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Create_LatLon                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Create_LatLon( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  /* Output argv[1] and argv[2] */
  IDL_VPTR vlatbuf, vlonbuf;
  MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);

  /* MISR Toolkit call */
  status = MtkCreateLatLon(*mapinfo, &latbuf, &lonbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = latbuf.nsample;
  dim[1] = latbuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(latbuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vlatbuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)latbuf.vdata[0],
			     free_cb, NULL);
  vlonbuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)lonbuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(latbuf.vdata);
  free(lonbuf.vdata);

  IDL_VarCopy(vlatbuf, argv[1]);
  IDL_VarCopy(vlonbuf, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&latbuf);
  MtkDataBufferFree(&lonbuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_WriteEnviFile                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_WriteEnviFile( int argc, IDL_VPTR *argv ) {

  int nline, nsample;
  MTKt_DataType datatype;
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[5] */
  char *envifilename;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vdatabuf;
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;
  char *misrfilename, *misrgridname, *misrfieldname;

  /* Inputs */
  envifilename = IDL_VarGetString(argv[0]);
  vdatabuf = argv[1];
  IDL_VarGetData(argv[2], &nmapinfo, (char **)&mapinfo, IDL_FALSE);
  misrfilename = IDL_VarGetString(argv[3]);
  misrgridname = IDL_VarGetString(argv[4]);
  misrfieldname = IDL_VarGetString(argv[5]);

  /* Check if IDL databuf argv[0] is an array and extract dimensions and
     pointer to data */
  IDL_ENSURE_SIMPLE(vdatabuf);
  IDL_ENSURE_ARRAY(vdatabuf);
  if (vdatabuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  
  nline = vdatabuf->value.arr->dim[1];
  nsample = vdatabuf->value.arr->dim[0];

  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vdatabuf->type, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Allocate a data buffer */
  status = MtkDataBufferAllocate(nline, nsample, datatype, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Now point data buffer data pointer to IDL's buffer temporarily */
  free(databuf.dataptr);
  databuf.dataptr = vdatabuf->value.arr->data;
  
  /* MISR Toolkit call */
  status = MtkWriteEnviFile(envifilename, databuf, *mapinfo,
			    misrfilename, misrgridname, misrfieldname);
  MTK_ERR_IDL_COND_JUMP(status);

  databuf.dataptr = NULL;
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  databuf.dataptr = NULL;
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_CoreMetaData_Get                                       */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_CoreMetaData_Get( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1];
  int datatype, i;
  IDL_STRING *data = NULL;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *param;

  /* Output argv[2] */
  IDL_VPTR vmd;
  MtkCoreMetaData md = MTK_CORE_METADATA_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  param = IDL_VarGetString(argv[1]);

  /* MISR Toolkit call */
  status = MtkFileCoreMetaDataGet(filename, param, &md);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = md.num_values;

  /* Set idl type */
  switch (md.datatype) {
  case MTKMETA_CHAR:	datatype = IDL_TYP_BYTE;
    break;
  case MTKMETA_INT:	datatype = IDL_TYP_INT;
    break;
  case MTKMETA_DOUBLE:	datatype = IDL_TYP_DOUBLE;
    break;
  default:
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_NOT_SUPPORTED);
    break;
  }

  if (md.datatype == MTKMETA_CHAR) {
     /* Create temporary vector of strings */
     if (md.num_values > 0) {
        data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					          md.num_values,
					          IDL_ARR_INI_ZERO, &vmd);
     }
     /* Store the strings into IDL vector */
     for (i = 0; i < md.num_values; i++)
       IDL_StrStore(&data[i],md.data.s[i]);
  }
  else {
     /* Import array from C to IDL (does not copy the data) */
     vmd = IDL_ImportArray(1, dim, datatype, (UCHAR *)md.dataptr,
			     free_cb, NULL);
  }

  /* Free only the Illife vector, only the data is sent back to idl */
  /* free(fillbuf.vdata); */

  /* IDL_VarCopy(vmd, argv[2]); */

  if (md.num_values > 0) IDL_VarCopy(vmd, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkCoreMetaDataFree(&md);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_CoreMetaData_Query                                     */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_CoreMetaData_Query( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] */
  char *filename;

  /* Outputs argv[1] and argv[2] */
  IDL_VPTR vnparam;
  int *nparam;
  IDL_MEMINT nnparam;

  IDL_VPTR vparamlist = NULL;
  IDL_STRING *data = NULL;
  char **paramlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vnparam = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vnparam, &nnparam, (char **)&nparam, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileCoreMetaDataQuery(filename, nparam, &paramlist_tmp);

  MTK_ERR_IDL_COND_JUMP(status);

  /* Create temporary vector of strings */
  if (*nparam > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*nparam,
					    IDL_ARR_INI_ZERO, &vparamlist);
  }
  /* Store the strings into IDL vector */
  for (i = 0; i < *nparam; i++)
    IDL_StrStore(&data[i],paramlist_tmp[i]);

  /* Free temporary fieldlist */
  MtkStringListFree(*nparam, &paramlist_tmp);

  IDL_VarCopy(vnparam, argv[1]);
  if (*nparam > 0) IDL_VarCopy(vparamlist, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*nparam > 0) IDL_DELTMP(vparamlist);
  IDL_DELTMP(vnparam);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_FileAttr_List                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_FileAttr_List( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] */
  char *filename;

  /* Outputs argv[1] and argv[2] */
  IDL_VPTR vattrcnt;
  int *attrcnt;
  IDL_MEMINT nattrcnt;

  IDL_VPTR vattrlist = NULL;
  IDL_STRING *data = NULL;
  char **attrlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vattrcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vattrcnt, &nattrcnt, (char **)&attrcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileAttrList(filename, attrcnt, &attrlist_tmp);
  MTK_ERR_IDL_COND_JUMP(status);

 /* Create temporary vector of strings */
  if (*attrcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*attrcnt,
					    IDL_ARR_INI_ZERO, &vattrlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *attrcnt; i++)
    IDL_StrStore(&data[i],attrlist_tmp[i]);

  /* Free temporary gridlist */
  MtkStringListFree(*attrcnt, &attrlist_tmp);

  IDL_VarCopy(vattrcnt, argv[1]);
  if (*attrcnt > 0) IDL_VarCopy(vattrlist, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*attrcnt > 0) IDL_DELTMP(vattrlist);
  IDL_DELTMP(vattrcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_FileAttr_Get                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_FileAttr_Get( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *attrname;

  /* Output argv[2] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  attrname = IDL_VarGetString(argv[1]);

  /* MISR Toolkit call */
  status = MtkFileAttrGet(filename, attrname, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  IDL_VarCopy(vdatabuf, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Block_Meta_List                                        */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Block_Meta_List( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] */
  char *filename;

  /* Outputs argv[1] and argv[2] */
  IDL_VPTR vblkmetacnt;
  int *blkmetacnt;
  IDL_MEMINT nblkmetacnt;

  IDL_VPTR vblkmetalist = NULL;
  IDL_STRING *data = NULL;
  char **blkmetalist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Output */
  vblkmetacnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vblkmetacnt, &nblkmetacnt, (char **)&blkmetacnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileBlockMetaList(filename, blkmetacnt, &blkmetalist_tmp);
  MTK_ERR_IDL_COND_JUMP(status);

 /* Create temporary vector of strings */
  if (*blkmetacnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*blkmetacnt,
					    IDL_ARR_INI_ZERO, &vblkmetalist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *blkmetacnt; i++)
    IDL_StrStore(&data[i],blkmetalist_tmp[i]);

  /* Free temporary gridlist */
  MtkStringListFree(*blkmetacnt, &blkmetalist_tmp);

  IDL_VarCopy(vblkmetacnt, argv[1]);
  if (*blkmetacnt > 0) IDL_VarCopy(vblkmetalist, argv[2]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*blkmetacnt > 0) IDL_DELTMP(vblkmetalist);
  IDL_DELTMP(vblkmetacnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Block_Meta_Field_List                                  */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Block_Meta_Field_List( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] and argv[1] */
  char *filename;
  char *blkmetaname;

  /* Outputs argv[2] and argv[3] */
  IDL_VPTR vfieldcnt;
  int *fieldcnt;
  IDL_MEMINT nfieldcnt;

  IDL_VPTR vfieldlist = NULL;
  IDL_STRING *data = NULL;
  char **fieldlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  blkmetaname = IDL_VarGetString(argv[1]);

  /* Output */
  vfieldcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vfieldcnt, &nfieldcnt, (char **)&fieldcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFileBlockMetaFieldList(filename, blkmetaname, fieldcnt, &fieldlist_tmp);
  MTK_ERR_IDL_COND_JUMP(status);

 /* Create temporary vector of strings */
  if (*fieldcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*fieldcnt,
					    IDL_ARR_INI_ZERO, &vfieldlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *fieldcnt; i++)
    IDL_StrStore(&data[i],fieldlist_tmp[i]);

  /* Free temporary gridlist */
  MtkStringListFree(*fieldcnt, &fieldlist_tmp);

  IDL_VarCopy(vfieldcnt, argv[2]);
  if (*fieldcnt > 0) IDL_VarCopy(vfieldlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*fieldcnt > 0) IDL_DELTMP(vfieldlist);
  IDL_DELTMP(vfieldcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_File_Block_Meta_Field_Read                                  */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_File_Block_Meta_Field_Read( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *blkmetaname;
  char *fieldname;

  /* Output argv[3] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  blkmetaname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkFileBlockMetaFieldRead(filename, blkmetaname, fieldname, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  IDL_VarCopy(vdatabuf, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_FillValueGet                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_FillValue_Get( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[2] */
  char *filename, *gridname, *fieldname;

  /* Output argv[3] */
  IDL_VPTR vfillbuf;
  MTKt_DataBuffer fillbuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  fieldname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkFillValueGet(filename, gridname, fieldname, &fillbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = fillbuf.nsample;
  dim[1] = fillbuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(fillbuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vfillbuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)fillbuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(fillbuf.vdata);

  IDL_VarCopy(vfillbuf, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&fillbuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_GridAttr_List                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_GridAttr_List( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *gridname;

  /* Outputs argv[2] and arg[3] */
  IDL_VPTR vattrcnt;
  int *attrcnt;
  IDL_MEMINT nattrcnt;

  IDL_VPTR vattrlist = NULL;
  IDL_STRING *data = NULL;
  char **attrlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);

  /* Output */
  vattrcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vattrcnt, &nattrcnt, (char **)&attrcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkGridAttrList(filename, gridname, attrcnt, &attrlist_tmp);
  MTK_ERR_IDL_COND_JUMP(status);

 /* Create temporary vector of strings */
  if (*attrcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    (IDL_MEMINT)*attrcnt,
					    IDL_ARR_INI_ZERO, &vattrlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *attrcnt; i++)
    IDL_StrStore(&data[i],attrlist_tmp[i]);

  /* Free temporary gridlist */
  MtkStringListFree(*attrcnt, &attrlist_tmp);

  IDL_VarCopy(vattrcnt, argv[2]);
  if (*attrcnt > 0) IDL_VarCopy(vattrlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*attrcnt > 0) IDL_DELTMP(vattrlist);
  IDL_DELTMP(vattrcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_GridAttr_Get                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_GridAttr_Get( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *gridname;
  char *attrname;

  /* Output argv[3] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  gridname = IDL_VarGetString(argv[1]);
  attrname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkGridAttrGet(filename, gridname, attrname, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)databuf.vdata[0],
			     free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  IDL_VarCopy(vdatabuf, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_FieldAttr_List                                              */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_FieldAttr_List( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  int i;

  /* Input argv[0] to argv[1] */
  char *filename;
  char *fieldname;

  /* Outputs argv[2] and arg[3] */
  IDL_VPTR vattrcnt;
  int *attrcnt;
  IDL_MEMINT nattrcnt;

  IDL_VPTR vattrlist = NULL;
  IDL_STRING *data = NULL;
  char **attrlist_tmp;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  fieldname = IDL_VarGetString(argv[1]);

  /* Output */
  vattrcnt = IDL_GettmpLong((IDL_LONG)0);
  IDL_VarGetData(vattrcnt, &nattrcnt, (char **)&attrcnt, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkFieldAttrList(filename, fieldname, attrcnt, &attrlist_tmp);
  MTK_ERR_IDL_COND_JUMP(status);

 /* Create temporary vector of strings */
  if (*attrcnt > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
              (IDL_MEMINT)*attrcnt,
              IDL_ARR_INI_ZERO, &vattrlist);
  }

  /* Store the strings into IDL vector */
  for (i = 0; i < *attrcnt; i++)
    IDL_StrStore(&data[i],attrlist_tmp[i]);

  /* Free temporary fieldlist */
  MtkStringListFree(*attrcnt, &attrlist_tmp);

  IDL_VarCopy(vattrcnt, argv[2]);
  if (*attrcnt > 0) IDL_VarCopy(vattrlist, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  if (*attrcnt > 0) IDL_DELTMP(vattrlist);
  IDL_DELTMP(vattrcnt);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_FieldAttr_Get                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_FieldAttr_Get( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[2] */
  char *filename;
  char *fieldname;
  char *attrname;

  /* Output argv[3] */
  IDL_VPTR vdatabuf;
  MTKt_DataBuffer databuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);
  fieldname = IDL_VarGetString(argv[1]);
  attrname = IDL_VarGetString(argv[2]);

  /* MISR Toolkit call */
  status = MtkFieldAttrGet(filename, fieldname, attrname, &databuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = databuf.nsample;
  dim[1] = databuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(databuf.datatype, &datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vdatabuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)databuf.vdata[0],
           free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(databuf.vdata);

  IDL_VarCopy(vdatabuf, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&databuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LatLon_To_LS                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LatLon_To_LS( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  IDL_VPTR vlat, vlon;
  double *lat, *lon;
  IDL_MEMINT nlat, nlon;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vline, vsample;
  float *line, *sample;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);

  vlat = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat, &nlat, (char **)&lat, IDL_TRUE);

  vlon = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon, &nlon, (char **)&lon, IDL_TRUE);

  /* Output */
  line = (float *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_FLOAT, NULL,
					     &vline, IDL_TRUE);
  sample = (float *)IDL_VarMakeTempFromTemplate(vlat, IDL_TYP_FLOAT, NULL,
					     &vsample, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLatLonToLSAry( *mapinfo, nlat, lat, lon, line, sample );

  IDL_DELTMP(vlat);
  IDL_DELTMP(vlon);
  IDL_VarCopy(vline, argv[3]);
  IDL_VarCopy(vsample, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LS_To_LatLon                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LS_To_LatLon( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  IDL_VPTR vline, vsample;
  float *line, *sample;
  IDL_MEMINT nline, nsample;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vlat, vlon;
  double *lat, *lon;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);

  vline = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_FLOAT);
  IDL_VarGetData(vline, &nline, (char **)&line, IDL_TRUE);

  vsample = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_FLOAT);
  IDL_VarGetData(vsample, &nsample, (char **)&sample, IDL_TRUE);

  /* Output */
  lat = (double *)IDL_VarMakeTempFromTemplate(vline, IDL_TYP_DOUBLE, NULL,
					      &vlat, IDL_TRUE);
  lon = (double *)IDL_VarMakeTempFromTemplate(vline, IDL_TYP_DOUBLE, NULL,
					      &vlon, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLSToLatLonAry( *mapinfo, nline, line, sample, lat, lon );

  IDL_DELTMP(vline);
  IDL_DELTMP(vsample);
  IDL_VarCopy(vlat, argv[3]);
  IDL_VarCopy(vlon, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SomXY_To_LS                                                 */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SomXY_To_LS( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;
  IDL_MEMINT nsom_x, nsom_y;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vline, vsample;
  float *line, *sample;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);

  vsom_x = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_x, &nsom_x, (char **)&som_x, IDL_TRUE);

  vsom_y = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_y, &nsom_y, (char **)&som_y, IDL_TRUE);

  /* Output */
  line = (float *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_FLOAT, NULL,
					      &vline, IDL_TRUE);
  sample = (float *)IDL_VarMakeTempFromTemplate(vsom_x, IDL_TYP_FLOAT, NULL,
						&vsample, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkSomXYToLSAry( *mapinfo, nsom_x, som_x, som_y, line, sample );

  IDL_DELTMP(vsom_x);
  IDL_DELTMP(vsom_y);
  IDL_VarCopy(vline, argv[3]);
  IDL_VarCopy(vsample, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LS_To_SomXY                                                 */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LS_To_SomXY( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] to argv[2] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  IDL_VPTR vline, vsample;
  float *line, *sample;
  IDL_MEMINT nline, nsample;

  /* Output argv[3] to argv[4] */
  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);

  vline = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_FLOAT);
  IDL_VarGetData(vline, &nline, (char **)&line, IDL_TRUE);

  vsample = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_FLOAT);
  IDL_VarGetData(vsample, &nsample, (char **)&sample, IDL_TRUE);

  /* Output */
  som_x = (double *)IDL_VarMakeTempFromTemplate(vline, IDL_TYP_DOUBLE, NULL,
					      &vsom_x, IDL_TRUE);
  som_y = (double *)IDL_VarMakeTempFromTemplate(vline, IDL_TYP_DOUBLE, NULL,
					      &vsom_y, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkLSToSomXYAry( *mapinfo, nline, line, sample, som_x, som_y );

  IDL_DELTMP(vline);
  IDL_DELTMP(vsample);
  IDL_VarCopy(vsom_x, argv[3]);
  IDL_VarCopy(vsom_y, argv[4]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dd_To_Deg_Min_Sec                                           */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dd_To_Deg_Min_Sec( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdd;
  double *dd;
  IDL_MEMINT ndd;

  /* Output argv[1] to argv[3] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;

  /* Inputs */
  vdd = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdd, &ndd, (char **)&dd, IDL_TRUE);

  /* Output */ 
  deg = (int *)IDL_VarMakeTempFromTemplate(vdd, IDL_TYP_LONG, NULL,
					   &vdeg, IDL_TRUE);
  min = (int *)IDL_VarMakeTempFromTemplate(vdd, IDL_TYP_LONG, NULL,
					   &vmin, IDL_TRUE);
  sec = (double *)IDL_VarMakeTempFromTemplate(vdd, IDL_TYP_DOUBLE, NULL,
					      &vsec, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndd; i++) {
    result = MtkDdToDegMinSec( dd[i], &deg[i], &min[i], &sec[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdd);
  IDL_VarCopy(vdeg, argv[1]);
  IDL_VarCopy(vmin, argv[2]);
  IDL_VarCopy(vsec, argv[3]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dd_To_Dms						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dd_To_Dms( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdd;
  double *dd;
  IDL_MEMINT ndd;

  /* Output argv[1] */
  IDL_VPTR vdms;
  double *dms;

  /* Inputs */
  vdd = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdd, &ndd, (char **)&dd, IDL_TRUE);

  /* Output */ 
  dms = (double *)IDL_VarMakeTempFromTemplate(vdd, IDL_TYP_DOUBLE, NULL,
					      &vdms, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndd; i++) {
    result = MtkDdToDms( dd[i], &dms[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdd);
  IDL_VarCopy(vdms, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dd_To_Rad						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dd_To_Rad( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdd;
  double *dd;
  IDL_MEMINT ndd;

  /* Output argv[1] */
  IDL_VPTR vrad;
  double *rad;

  /* Inputs */
  vdd = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdd, &ndd, (char **)&dd, IDL_TRUE);

  /* Output */ 
  rad = (double *)IDL_VarMakeTempFromTemplate(vdd, IDL_TYP_DOUBLE, NULL,
					      &vrad, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndd; i++) {
    result = MtkDdToRad( dd[i], &rad[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdd);
  IDL_VarCopy(vrad, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Deg_Min_Sec_To_Dd					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Deg_Min_Sec_To_Dd( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] to argv[2] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;
  IDL_MEMINT ndeg, nmin, nsec;

  /* Output argv[3] */
  IDL_VPTR vdd;
  double *dd;

  /* Inputs */
  vdeg = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vdeg, &ndeg, (char **)&deg, IDL_TRUE);

  vmin = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vmin, &nmin, (char **)&min, IDL_TRUE);

  vsec = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsec, &nsec, (char **)&sec, IDL_TRUE);

  if ((ndeg != nmin) || (nmin != nsec)) 
    MTK_ERR_IDL_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Output */ 
  dd = (double *)IDL_VarMakeTempFromTemplate(vsec, IDL_TYP_DOUBLE, NULL,
					     &vdd, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nsec; i++) {
    result = MtkDegMinSecToDd( deg[i], min[i], sec[i], &dd[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  IDL_VarCopy(vdd, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Deg_Min_Sec_To_Dms					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Deg_Min_Sec_To_Dms( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] to argv[2] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;
  IDL_MEMINT ndeg, nmin, nsec;

  /* Output argv[3] */
  IDL_VPTR vdms;
  double *dms;

  /* Inputs */
  vdeg = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vdeg, &ndeg, (char **)&deg, IDL_TRUE);

  vmin = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vmin, &nmin, (char **)&min, IDL_TRUE);

  vsec = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsec, &nsec, (char **)&sec, IDL_TRUE);

  if ((ndeg != nmin) || (nmin != nsec)) 
    MTK_ERR_IDL_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Output */ 
  dms = (double *)IDL_VarMakeTempFromTemplate(vsec, IDL_TYP_DOUBLE, NULL,
					      &vdms, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nsec; i++) {
    result = MtkDegMinSecToDms( deg[i], min[i], sec[i], &dms[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  IDL_VarCopy(vdms, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Deg_Min_Sec_To_Rad					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Deg_Min_Sec_To_Rad( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] to argv[2] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;
  IDL_MEMINT ndeg, nmin, nsec;

  /* Output argv[3] */
  IDL_VPTR vrad;
  double *rad;

  /* Inputs */
  vdeg = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vdeg, &ndeg, (char **)&deg, IDL_TRUE);

  vmin = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vmin, &nmin, (char **)&min, IDL_TRUE);

  vsec = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsec, &nsec, (char **)&sec, IDL_TRUE);

  if ((ndeg != nmin) || (nmin != nsec)) 
    MTK_ERR_IDL_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Output */ 
  rad = (double *)IDL_VarMakeTempFromTemplate(vsec, IDL_TYP_DOUBLE, NULL,
					      &vrad, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nsec; i++) {
    result = MtkDegMinSecToRad( deg[i], min[i], sec[i], &rad[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  IDL_VarCopy(vrad, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vdeg);
  IDL_DELTMP(vmin);
  IDL_DELTMP(vsec);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dms_To_Dd						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dms_To_Dd( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdms;
  double *dms;
  IDL_MEMINT ndms;

  /* Output argv[1] */
  IDL_VPTR vdd;
  double *dd;

  /* Inputs */
  vdms = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdms, &ndms, (char **)&dms, IDL_TRUE);

  /* Output */ 
  dd = (double *)IDL_VarMakeTempFromTemplate(vdms, IDL_TYP_DOUBLE, NULL,
					     &vdd, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndms; i++) {
    result = MtkDmsToDd( dms[i], &dd[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdms);
  IDL_VarCopy(vdd, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dms_To_Deg_Min_Sec                                          */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dms_To_Deg_Min_Sec( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdms;
  double *dms;
  IDL_MEMINT ndms;

  /* Output argv[1] to argv[3] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;

  /* Inputs */
  vdms = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdms, &ndms, (char **)&dms, IDL_TRUE);

  /* Output */ 
  deg = (int *)IDL_VarMakeTempFromTemplate(vdms, IDL_TYP_LONG, NULL,
					   &vdeg, IDL_TRUE);
  min = (int *)IDL_VarMakeTempFromTemplate(vdms, IDL_TYP_LONG, NULL,
					   &vmin, IDL_TRUE);
  sec = (double *)IDL_VarMakeTempFromTemplate(vdms, IDL_TYP_DOUBLE, NULL,
					      &vsec, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndms; i++) {
    result = MtkDmsToDegMinSec( dms[i], &deg[i], &min[i], &sec[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdms);
  IDL_VarCopy(vdeg, argv[1]);
  IDL_VarCopy(vmin, argv[2]);
  IDL_VarCopy(vsec, argv[3]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Dms_To_Rad						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Dms_To_Rad( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vdms;
  double *dms;
  IDL_MEMINT ndms;

  /* Output argv[1] */
  IDL_VPTR vrad;
  double *rad;

  /* Inputs */
  vdms = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vdms, &ndms, (char **)&dms, IDL_TRUE);

  /* Output */ 
  rad = (double *)IDL_VarMakeTempFromTemplate(vdms, IDL_TYP_DOUBLE, NULL,
					      &vrad, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < ndms; i++) {
    result = MtkDmsToRad( dms[i], &rad[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vdms);
  IDL_VarCopy(vrad, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Rad_To_Dd						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Rad_To_Dd( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vrad;
  double *rad;
  IDL_MEMINT nrad;

  /* Output argv[1] */
  IDL_VPTR vdd;
  double *dd;

  /* Inputs */
  vrad = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vrad, &nrad, (char **)&rad, IDL_TRUE);

  /* Output */ 
  dd = (double *)IDL_VarMakeTempFromTemplate(vrad, IDL_TYP_DOUBLE, NULL,
					     &vdd, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nrad; i++) {
    result = MtkRadToDd( rad[i], &dd[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vrad);
  IDL_VarCopy(vdd, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Rad_To_Deg_Min_Sec                                          */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Rad_To_Deg_Min_Sec( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vrad;
  double *rad;
  IDL_MEMINT nrad;

  /* Output argv[1] to argv[3] */
  IDL_VPTR vdeg, vmin, vsec;
  int *deg, *min;
  double *sec;

  /* Inputs */
  vrad = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vrad, &nrad, (char **)&rad, IDL_TRUE);

  /* Output */ 
  deg = (int *)IDL_VarMakeTempFromTemplate(vrad, IDL_TYP_LONG, NULL,
					   &vdeg, IDL_TRUE);
  min = (int *)IDL_VarMakeTempFromTemplate(vrad, IDL_TYP_LONG, NULL,
					   &vmin, IDL_TRUE);
  sec = (double *)IDL_VarMakeTempFromTemplate(vrad, IDL_TYP_DOUBLE, NULL,
					      &vsec, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nrad; i++) {
    result = MtkRadToDegMinSec( rad[i], &deg[i], &min[i], &sec[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vrad);
  IDL_VarCopy(vdeg, argv[1]);
  IDL_VarCopy(vmin, argv[2]);
  IDL_VarCopy(vsec, argv[3]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Rad_To_Dms						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Rad_To_Dms( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] */
  IDL_VPTR vrad;
  double *rad;
  IDL_MEMINT nrad;

  /* Output argv[1] */
  IDL_VPTR vdms;
  double *dms;

  /* Inputs */
  vrad = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vrad, &nrad, (char **)&rad, IDL_TRUE);

  /* Output */ 
  dms = (double *)IDL_VarMakeTempFromTemplate(vrad, IDL_TYP_DOUBLE, NULL,
					      &vdms, IDL_TRUE);

  /* MISR Toolkit call */
  for (i = 0; i < nrad; i++) {
    result = MtkRadToDms( rad[i], &dms[i] );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
    }
  }

  IDL_DELTMP(vrad);
  IDL_VarCopy(vdms, argv[1]);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Julian_To_Datetime					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Julian_To_Datetime( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  IDL_VPTR vjulian;
  double *julian;
  IDL_MEMINT njulian;

  /* Output argv[1] */
  char datetime[MTKd_DATETIME_LEN];

  /* Inputs */
  vjulian = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vjulian, &njulian, (char **)&julian, IDL_TRUE);

 /* MISR Toolkit call */
  status = MtkJulianToDateTime( *julian, datetime );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_DELTMP(vjulian);
  IDL_VarCopy(IDL_StrToSTRING(datetime), argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Datetime_To_Julian					   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Datetime_To_Julian( int argc, IDL_VPTR *argv ) {

  MTKt_status status;

  /* Input argv[0] */
  char *datetime;

  /* Output argv[1] */
  double julian;

  /* Inputs */
  datetime = IDL_VarGetString(argv[0]);

  /* MISR Toolkit call */
  status = MtkDateTimeToJulian( datetime, &julian );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_StoreScalar(argv[1], IDL_TYP_DOUBLE, (IDL_ALLTYPES *)&julian);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Time_Meta_Read						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Time_Meta_Read( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[1] = { 1 };

  /* Input argv[0] */
  char *filename;

  /* Output argv[1] */
  IDL_VPTR vtime_metadata;
  void *stime_metadata;
  MTKt_TimeMetaData *time_metadata;

  /* Inputs */
  filename = IDL_VarGetString(argv[0]);

  /* Outputs */
  time_metadata = (MTKt_TimeMetaData *)calloc(1, sizeof(MTKt_TimeMetaData));
  stime_metadata = IDL_MakeStruct(NULL, time_metadata_s_tags);
  vtime_metadata = IDL_ImportArray(1, dim, IDL_TYP_STRUCT, 
			    (UCHAR *)time_metadata, free_cb, stime_metadata);

  /* MISR Toolkit call */
  status = MtkTimeMetaRead( filename, time_metadata );
  MTK_ERR_IDL_COND_JUMP(status);

  IDL_VarCopy(vtime_metadata, argv[1]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vtime_metadata);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Pixel_Time						   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Pixel_Time( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_SUCCESS;
  MTKt_status result;
  int i;

  /* Input argv[0] to argv[2] */
  MTKt_TimeMetaData *time_metadata;
  IDL_MEMINT ntime_metadata;

  IDL_VPTR vsom_x, vsom_y;
  double *som_x, *som_y;
  IDL_MEMINT nsom_x, nsom_y;

  /* Output argv[3] */
  IDL_VPTR vpixel_time = NULL;
  IDL_STRING *data = NULL;
  char pixel_time_tmp[MTKd_DATETIME_LEN];

  /* Inputs */
  IDL_VarGetData(argv[0], &ntime_metadata, (char **)&time_metadata, IDL_FALSE);

  vsom_x = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_x, &nsom_x, (char **)&som_x, IDL_TRUE);

  vsom_y = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vsom_y, &nsom_y, (char **)&som_y, IDL_TRUE);

  if (nsom_x != nsom_y) MTK_ERR_IDL_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Outputs */
  if (nsom_x > 0) {
    data = (IDL_STRING *)IDL_MakeTempVector(IDL_TYP_STRING,
					    nsom_x,
					    IDL_ARR_INI_ZERO, &vpixel_time);
  } else {
    MTK_ERR_IDL_COND_JUMP(MTK_DIMENSION_MISMATCH);
  }

  /* MISR Toolkit call */
  for (i = 0; i < nsom_x; i++) {
    result = MtkPixelTime( *time_metadata, som_x[i], som_y[i], pixel_time_tmp );
    if (result != MTK_SUCCESS) {
      MTK_ERR_IDL_LOG(result);
      status = result;
      IDL_StrStore(&data[i], "Not Available");
    } else {
      IDL_StrStore(&data[i], pixel_time_tmp);
    }
  }

  IDL_DELTMP(vsom_x);
  IDL_DELTMP(vsom_y);
  IDL_VarCopy(vpixel_time, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  IDL_DELTMP(vsom_x);
  IDL_DELTMP(vsom_y);
  IDL_DELTMP(vpixel_time);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_CreateGeoGrid                                               */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_CreateGeoGrid( int argc, IDL_VPTR *argv ) {

  MTKt_status status;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[5] */
  IDL_VPTR vulc_lat_dd, vulc_lon_dd, vlrc_lat_dd, vlrc_lon_dd, vlat_cellsize_dd, vlon_cellsize_dd;
  double *ulc_lat_dd, *ulc_lon_dd, *lrc_lat_dd, *lrc_lon_dd, *lat_cellsize_dd, *lon_cellsize_dd;
  IDL_MEMINT nulc_lat_dd, nulc_lon_dd, nlrc_lat_dd, nlrc_lon_dd, nlat_cellsize_dd, nlon_cellsize_dd;

  /* Output argv[6] and argv[7] */
  IDL_VPTR vlatbuf;
  MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlonbuf;
  MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;

  /* Inputs */
  vulc_lat_dd = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lat_dd, &nulc_lat_dd, (char **)&ulc_lat_dd, IDL_TRUE);

  vulc_lon_dd = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vulc_lon_dd, &nulc_lon_dd, (char **)&ulc_lon_dd, IDL_TRUE);

  vlrc_lat_dd = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lat_dd, &nlrc_lat_dd, (char **)&lrc_lat_dd, IDL_TRUE);

  vlrc_lon_dd = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlrc_lon_dd, &nlrc_lon_dd, (char **)&lrc_lon_dd, IDL_TRUE);

  vlat_cellsize_dd = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlat_cellsize_dd, &nlat_cellsize_dd, (char **)&lat_cellsize_dd, IDL_TRUE);

  vlon_cellsize_dd = IDL_BasicTypeConversion(1, &argv[5], IDL_TYP_DOUBLE);
  IDL_VarGetData(vlon_cellsize_dd, &nlon_cellsize_dd, (char **)&lon_cellsize_dd, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkCreateGeoGrid(*ulc_lat_dd, *ulc_lon_dd, *lrc_lat_dd, *lrc_lon_dd,
			    *lat_cellsize_dd, *lon_cellsize_dd, &latbuf, &lonbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson and datatype */
  dim[0] = latbuf.nsample;
  dim[1] = latbuf.nline;
  datatype = IDL_TYP_DOUBLE;

  /* Import array from C to IDL (does not copy the data) */
  vlatbuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)latbuf.vdata[0],
			    free_cb, NULL);
  vlonbuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)lonbuf.vdata[0],
			    free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(latbuf.vdata);
  free(lonbuf.vdata);

  IDL_DELTMP(vulc_lat_dd);
  IDL_DELTMP(vulc_lon_dd);
  IDL_DELTMP(vlrc_lat_dd);
  IDL_DELTMP(vlrc_lon_dd);
  IDL_DELTMP(vlat_cellsize_dd);
  IDL_DELTMP(vlon_cellsize_dd);
  IDL_VarCopy(vlatbuf, argv[6]);
  IDL_VarCopy(vlonbuf, argv[7]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&latbuf);
  MtkDataBufferFree(&lonbuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_TransformCoordinates                                        */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_TransformCoordinates( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_FAILURE;
  IDL_MEMINT dim[2];
  int datatype;

  /* Input argv[0] to argv[2] */
  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;
  MTKt_DataBuffer latbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlatbuf;
  int latbuf_nline, latbuf_nsample;
  MTKt_DataBuffer lonbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlonbuf;
  int lonbuf_nline, lonbuf_nsample;

  /* Output argv[3] and argv[4] */
  MTKt_DataBuffer linebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlinebuf;
  MTKt_DataBuffer samplebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsamplebuf;

  /* Inputs */
  IDL_VarGetData(argv[0], &nmapinfo, (char **)&mapinfo, IDL_FALSE);
  vlatbuf = argv[1];
  vlonbuf = argv[2];

  /* Check if the IDL lat data buffer argv[1] is an array. 
     Extract dimensions, datatype and pointer to data to create 
     the latbuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vlatbuf);
  IDL_ENSURE_ARRAY(vlatbuf);
  if (vlatbuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  if (vlatbuf->type != IDL_TYP_DOUBLE)
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_MISMATCH);
  latbuf_nline = vlatbuf->value.arr->dim[1];
  latbuf_nsample = vlatbuf->value.arr->dim[0];

  /* Allocate a data buffer */
  status = MtkDataBufferImport(latbuf_nline, latbuf_nsample, MTKe_double,
			       vlatbuf->value.arr->data, &latbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Check if the IDL lon data buffer argv[2] is an array. 
     Extract dimensions, datatype and pointer to data to create 
     the lonbuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vlonbuf);
  IDL_ENSURE_ARRAY(vlonbuf);
  if (vlonbuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  if (vlonbuf->type != IDL_TYP_DOUBLE)
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_MISMATCH);
  lonbuf_nline = vlonbuf->value.arr->dim[1];
  lonbuf_nsample = vlonbuf->value.arr->dim[0];

  /* Allocate a data buffer */
  status = MtkDataBufferImport(lonbuf_nline, lonbuf_nsample, MTKe_double,
			       vlonbuf->value.arr->data, &lonbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* MISR Toolkit call */
  status = MtkTransformCoordinates(*mapinfo, latbuf, lonbuf,
				   &linebuf, &samplebuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson and datatype */
  dim[0] = linebuf.nsample;
  dim[1] = linebuf.nline;
  datatype = IDL_TYP_FLOAT;

  /* Import array from C to IDL (does not copy the data) */
  vlinebuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)linebuf.vdata[0],
			    free_cb, NULL);
  vsamplebuf = IDL_ImportArray(2, dim, datatype, (UCHAR *)samplebuf.vdata[0],
			    free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(linebuf.vdata);
  free(samplebuf.vdata);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&latbuf);
  MtkDataBufferFree(&lonbuf);

  IDL_VarCopy(vlinebuf, argv[3]);
  IDL_VarCopy(vsamplebuf, argv[4]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&latbuf);
  MtkDataBufferFree(&lonbuf);
  MtkDataBufferFree(&linebuf);
  MtkDataBufferFree(&samplebuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ResampleNearestNeighbor                                     */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ResampleNearestNeighbor( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_FAILURE;
  MTKt_DataType mtk_datatype;
  int idl_datatype;
  IDL_MEMINT dim[2];

  /* Input argv[0] to argv[2] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;
  int srcbuf_nline, srcbuf_nsample;
  MTKt_DataBuffer linebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlinebuf;
  int linebuf_nline, linebuf_nsample;
  MTKt_DataBuffer samplebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsamplebuf;
  int samplebuf_nline, samplebuf_nsample;

  /* Output argv[3] */
  MTKt_DataBuffer resampbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresampbuf;

  /* Inputs */
  vsrcbuf = argv[0];
  vlinebuf = argv[1];
  vsamplebuf = argv[2];

  /* Check if the IDL src data buffer argv[0] is an array. 
     Extract dimensions, datatype and pointer to data to create 
     the srcbuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vsrcbuf);
  IDL_ENSURE_ARRAY(vsrcbuf);
  if (vsrcbuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  srcbuf_nline = vsrcbuf->value.arr->dim[1];
  srcbuf_nsample = vsrcbuf->value.arr->dim[0];

  /* Set Mtk datatype */
  status = Mtk_IdlToMtkDatatype(vsrcbuf->type, &mtk_datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Allocate a data buffer */
  status = MtkDataBufferImport(srcbuf_nline, srcbuf_nsample, mtk_datatype,
			       vsrcbuf->value.arr->data, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Check if the IDL line data buffer argv[1] is an array. 
     Extract dimensions, datatype and pointer to data to create 
     the linebuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vlinebuf);
  IDL_ENSURE_ARRAY(vlinebuf);
  if (vlinebuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  if (vlinebuf->type != IDL_TYP_FLOAT)
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_MISMATCH);
  linebuf_nline = vlinebuf->value.arr->dim[1];
  linebuf_nsample = vlinebuf->value.arr->dim[0];

  /* Allocate a data buffer */
  status = MtkDataBufferImport(linebuf_nline, linebuf_nsample, MTKe_float,
			       vlinebuf->value.arr->data, &linebuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Check if the IDL sample data buffer argv[2] is an array. 
     Extract dimensions, datatype and pointer to data to create 
     the samplebuf MtkDataBuffer */
  IDL_ENSURE_SIMPLE(vsamplebuf);
  IDL_ENSURE_ARRAY(vsamplebuf);
  if (vsamplebuf->value.arr->n_dim != 2)
    MTK_ERR_IDL_JUMP(MTK_DIMENSION_MISMATCH);
  if (vsamplebuf->type != IDL_TYP_FLOAT)
    MTK_ERR_IDL_JUMP(MTK_DATATYPE_MISMATCH);
  samplebuf_nline = vsamplebuf->value.arr->dim[1];
  samplebuf_nsample = vsamplebuf->value.arr->dim[0];

  /* Allocate a data buffer */
  status = MtkDataBufferImport(samplebuf_nline, samplebuf_nsample, MTKe_float,
			       vsamplebuf->value.arr->data, &samplebuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* MISR Toolkit call */
  status = MtkResampleNearestNeighbor(srcbuf, linebuf, samplebuf, &resampbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Set idl dimenson */
  dim[0] = resampbuf.nsample;
  dim[1] = resampbuf.nline;

  /* Set idl type */
  status = Mtk_MtkToIdlDatatype(resampbuf.datatype, &idl_datatype);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  vresampbuf = IDL_ImportArray(2, dim, idl_datatype, (UCHAR *)resampbuf.vdata[0],
			       free_cb, NULL);

  /* Free only the Illife vector, only the data is sent back to idl */
  free(resampbuf.vdata);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&linebuf);
  MtkDataBufferFree(&samplebuf);

  IDL_VarCopy(vresampbuf, argv[3]);
  return IDL_GettmpLong(status);
 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&linebuf);
  MtkDataBufferFree(&samplebuf);
  MtkDataBufferFree(&resampbuf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ResampleCubicConvolution                                    */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ResampleCubicConvolution( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[4] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  MTKt_DataBuffer src_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrc_mask_buf;

  MTKt_DataBuffer linebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vlinebuf;

  MTKt_DataBuffer samplebuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsamplebuf;

  IDL_VPTR va;
  double *a;
  IDL_MEMINT na;

  /* Output argv[5] and argv[6] */
  MTKt_DataBuffer resampbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresampbuf;
  MTKt_DataBuffer resamp_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresamp_mask_buf;

  /* Inputs */
  vsrcbuf = argv[0];
  vsrc_mask_buf = argv[1];
  vlinebuf = argv[2];
  vsamplebuf = argv[3];
  va = IDL_BasicTypeConversion(1, &argv[4], IDL_TYP_DOUBLE);
  IDL_VarGetData(va, &na, (char **)&a, IDL_TRUE);
	status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
	MTK_ERR_IDL_COND_JUMP(status);
	status = IDL_toMtkDataBuffer(&vsrc_mask_buf, &src_mask_buf);
	MTK_ERR_IDL_COND_JUMP(status);
	status = IDL_toMtkDataBuffer(&vlinebuf, &linebuf);
	MTK_ERR_IDL_COND_JUMP(status);
	status = IDL_toMtkDataBuffer(&vsamplebuf, &samplebuf);
	MTK_ERR_IDL_COND_JUMP(status);  

  /* MISR Toolkit call */
  status = MtkResampleCubicConvolution(&srcbuf, &src_mask_buf, &linebuf, &samplebuf, a[0], &resampbuf, &resamp_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
    
  /* Import array from C to IDL (does not copy the data) */
  status = Mtk_toIDLDataBuffer(&resampbuf, &vresampbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = Mtk_toIDLDataBuffer(&resamp_mask_buf, &vresamp_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&linebuf);
  MtkDataBufferFree(&samplebuf);
	
	/* Free pointer for a */
  IDL_DELTMP(va);

  IDL_VarCopy(vresampbuf, argv[5]);
  IDL_VarCopy(vresamp_mask_buf, argv[6]);
  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&linebuf);
  MtkDataBufferFree(&samplebuf);
  MtkDataBufferFree(&resampbuf);
  MtkDataBufferFree(&resamp_mask_buf);
  IDL_DELTMP(va);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ApplyRegression                                              */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ApplyRegression( int argc, IDL_VPTR *argv ) {
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[4] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  MTKt_DataBuffer src_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrc_mask_buf;

  MTKt_MapInfo *src_mapinfo;
  IDL_MEMINT nsrc_mapinfo;

  MTKt_RegressionCoeff reg_coeff = MTKT_REGRESSION_COEFF_INIT;
  IDL_VPTR vreg_coeff;

  MTKt_MapInfo *reg_mapinfo;
  IDL_MEMINT nreg_mapinfo;

  /* Output argv[5] and argv[6] */
  MTKt_DataBuffer regressed_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vregressed_buf;
  MTKt_DataBuffer regressed_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vregressed_mask_buf;

    /* Inputs */
  vsrcbuf = argv[0];
  vsrc_mask_buf = argv[1];
  status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vsrc_mask_buf, &src_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  IDL_VarGetData(argv[2], &nsrc_mapinfo, (char **)&src_mapinfo, IDL_FALSE);
  vreg_coeff = argv[3];
  IDL_toMtkRegCoeffStruct(&vreg_coeff, &reg_coeff);
  MTK_ERR_IDL_COND_JUMP(status);
  IDL_VarGetData(argv[4], &nreg_mapinfo, (char **)&reg_mapinfo, IDL_FALSE);
	
  /* MISR Toolkit call */
  status = MtkApplyRegression(&srcbuf, &src_mask_buf, src_mapinfo, &reg_coeff, reg_mapinfo, &regressed_buf, &regressed_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
	
  /* Outputs */
  status = Mtk_toIDLDataBuffer( &regressed_buf, &vregressed_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = Mtk_toIDLDataBuffer( &regressed_mask_buf, &vregressed_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  IDL_VarCopy(vregressed_buf, argv[5]);
  IDL_VarCopy(vregressed_mask_buf, argv[6]);
					
  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);

  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&regressed_buf);
  MtkDataBufferFree(&regressed_mask_buf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_Downsample                                                  */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_Downsample( int argc, IDL_VPTR *argv ) {

  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[2] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  MTKt_DataBuffer src_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrc_mask_buf;
	
  IDL_VPTR vsize_factor = NULL;
  int *size_factor;
  IDL_MEMINT nsize_factor;
	
  /* Output argv[3] and argv[4] */
  MTKt_DataBuffer resampbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresampbuf;
  MTKt_DataBuffer resamp_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresamp_mask_buf;

  /* Inputs */
  vsrcbuf = argv[0];
  vsrc_mask_buf = argv[1];
  status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vsrc_mask_buf, &src_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  vsize_factor = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(vsize_factor, &nsize_factor, (char **)&size_factor, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkDownsample(&srcbuf, &src_mask_buf, size_factor[0], &resampbuf, &resamp_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
	
  /* Import array from C to IDL (does not copy the data) */
  status = Mtk_toIDLDataBuffer( &resampbuf, &vresampbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = Mtk_toIDLDataBuffer( &resamp_mask_buf, &vresamp_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
	
  /* Free pointer for a */
  IDL_DELTMP(vsize_factor);

  IDL_VarCopy(vresampbuf, argv[3]);
  IDL_VarCopy(vresamp_mask_buf, argv[4]);
  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&resampbuf);
  MtkDataBufferFree(&resamp_mask_buf);
  IDL_DELTMP(vsize_factor);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_LinearRegressionCalc                                        */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_LinearRegressionCalc( int argc, IDL_VPTR *argv ) {
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[3] */
  IDL_VPTR vnum;
  int *num;
  IDL_MEMINT nnum;

  IDL_VPTR vx, vy, vysigma;
  double *x, *y, *ysigma;
  IDL_MEMINT nx, ny, nysigma;

  /* Output argv[4] and argv[6] */
  IDL_VPTR va, vb, vcorrelation;
  double *a, *b, *correlation;

  /* Inputs */
  vnum = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(vnum, &nnum, (char **)&num, IDL_TRUE);

  vx = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_DOUBLE);
  IDL_VarGetData(vx, &nx, (char **)&x, IDL_TRUE);

  vy = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_DOUBLE);
  IDL_VarGetData(vy, &ny, (char **)&y, IDL_TRUE);

  vysigma = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_DOUBLE);
  IDL_VarGetData(vysigma, &nysigma, (char **)&ysigma, IDL_TRUE);
	
	/* Initialize outputs */
  a = (double *)IDL_VarMakeTempFromTemplate(vx, IDL_TYP_DOUBLE, NULL, &va, IDL_TRUE);
  b = (double *)IDL_VarMakeTempFromTemplate(vy, IDL_TYP_DOUBLE, NULL, &vb, IDL_TRUE);
  correlation = (double *)IDL_VarMakeTempFromTemplate(vx, IDL_TYP_DOUBLE, NULL, &vcorrelation, IDL_TRUE);


  /* MISR Toolkit call */
  status = MtkLinearRegressionCalc(num[0], x, y, ysigma, a, b, correlation);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  IDL_VarCopy(va, argv[4]);
  IDL_VarCopy(vb, argv[5]);
  IDL_VarCopy(vcorrelation, argv[6]);

  /*    Free pointers    */
  IDL_DELTMP(vnum);
  IDL_DELTMP(vx);
  IDL_DELTMP(vy);
  IDL_DELTMP(vysigma);

  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  IDL_DELTMP(vnum);
  IDL_DELTMP(vx);
  IDL_DELTMP(vy);
  IDL_DELTMP(vysigma);
  IDL_DELTMP(va);
  IDL_DELTMP(vb);
  IDL_DELTMP(vcorrelation);

  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_SmoothData                                                  */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_SmoothData( int argc, IDL_VPTR *argv ) {
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[3] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  MTKt_DataBuffer src_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrc_mask_buf;

  IDL_VPTR vwidth_line = NULL;
  IDL_VPTR vwidth_sample = NULL;
  int *width_line, *width_sample;
  IDL_MEMINT nwidth_line, nwidth_sample;

  /* Output argv[4] */
  MTKt_DataBuffer smoothed = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsmoothed = NULL;

  /* Inputs */
  vsrcbuf = argv[0];
  vsrc_mask_buf = argv[1];
  status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vsrc_mask_buf, &src_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  vwidth_line = IDL_BasicTypeConversion(1, &argv[2], IDL_TYP_LONG);
  IDL_VarGetData(vwidth_line, &nwidth_line, (char **)&width_line, IDL_TRUE);
  vwidth_sample = IDL_BasicTypeConversion(1, &argv[3], IDL_TYP_LONG);
  IDL_VarGetData(vwidth_sample, &nwidth_sample, (char **)&width_sample, IDL_TRUE);
	
  /* MISR Toolkit call */
  status = MtkSmoothData(&srcbuf, &src_mask_buf, width_line[0], width_sample[0], &smoothed);
  MTK_ERR_IDL_COND_JUMP(status);
	
  /* Import array from C to IDL (does not copy the data) */
  status = Mtk_toIDLDataBuffer( &smoothed, &vsmoothed);
  MTK_ERR_IDL_COND_JUMP(status);
  IDL_VarCopy(vsmoothed, argv[4]);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);

  /* Free pointers */
  IDL_DELTMP(vwidth_line);
  IDL_DELTMP(vwidth_sample);

  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  IDL_DELTMP(vsmoothed);
  IDL_DELTMP(vwidth_line);
  IDL_DELTMP(vwidth_sample);
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_UpsampleMask                                                */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_UpsampleMask( int argc, IDL_VPTR *argv ) {
	MTKt_status status = MTK_FAILURE;

  /* Input argv[0] and argv[1] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  IDL_VPTR vsize_factor = NULL;
  int *size_factor;
  IDL_MEMINT nsize_factor;

  /* Output argv[2]    */
  MTKt_DataBuffer resampbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vresampbuf;

  /* Inputs */
  vsrcbuf = argv[0];
  status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  vsize_factor = IDL_BasicTypeConversion(1, &argv[1], IDL_TYP_LONG);
  IDL_VarGetData(vsize_factor, &nsize_factor, (char **)&size_factor, IDL_TRUE);

  /* MISR Toolkit call */
  status = MtkUpsampleMask(&srcbuf, size_factor[0], &resampbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Import array from C to IDL (does not copy the data) */
  status = Mtk_toIDLDataBuffer( &resampbuf, &vresampbuf);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Free MtkDataBuffers */
  MtkDataBufferFree(&srcbuf);

  /* Free pointer for a */
  IDL_DELTMP(vsize_factor);

  IDL_VarCopy(vresampbuf, argv[2]);
  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&resampbuf);
  IDL_DELTMP(vsize_factor);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_RegressionCoeffCalc                                         */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_RegressionCoeffCalc( int argc, IDL_VPTR *argv ) {
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[6] */
  MTKt_DataBuffer srcbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrcbuf;

  MTKt_DataBuffer src_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vsrc_mask_buf;

  MTKt_DataBuffer matchbuf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vmatchbuf;

  MTKt_DataBuffer match_sig_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vmatch_sig_buf;

  MTKt_DataBuffer match_mask_buf = MTKT_DATABUFFER_INIT;
  IDL_VPTR vmatch_mask_buf;

  MTKt_MapInfo *mapinfo;
  IDL_MEMINT nmapinfo;

  IDL_VPTR vsize_factor;
  int *size_factor;
  IDL_MEMINT nsize_factor;

  /* Output argv[7] and argv[8] */
  MTKt_RegressionCoeff regression_coeff = MTKT_REGRESSION_COEFF_INIT;
  IDL_VPTR      vregression_coeff = NULL;

  IDL_VPTR vregression_coeff_mapinfo = NULL;
  void *sregression_coeff_mapinfo;
  MTKt_MapInfo *regression_coeff_mapinfo;
  IDL_MEMINT mapinfo_dim[1] = { 1 };

  /* Inputs */
  vsrcbuf = argv[0];
  vsrc_mask_buf = argv[1];
  vmatchbuf = argv[2];
  vmatch_sig_buf = argv[3];
  vmatch_mask_buf = argv[4];
  IDL_VarGetData(argv[5], &nmapinfo, (char **)&mapinfo, IDL_FALSE);
  vsize_factor = IDL_BasicTypeConversion(1, &argv[6], IDL_TYP_LONG);

  status = IDL_toMtkDataBuffer(&vsrcbuf, &srcbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vsrc_mask_buf, &src_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vmatchbuf, &matchbuf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vmatch_sig_buf, &match_sig_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  status = IDL_toMtkDataBuffer(&vmatch_mask_buf, &match_mask_buf);
  MTK_ERR_IDL_COND_JUMP(status);
  IDL_VarGetData(vsize_factor, &nsize_factor, (char **)&size_factor, IDL_TRUE);
  regression_coeff_mapinfo = (MTKt_MapInfo *)malloc(sizeof(MTKt_MapInfo));
  sregression_coeff_mapinfo = IDL_MakeStruct(NULL, mapinfo_s_tags);
  vregression_coeff_mapinfo = IDL_ImportArray(1, mapinfo_dim, IDL_TYP_STRUCT,(UCHAR *)regression_coeff_mapinfo, free_cb, sregression_coeff_mapinfo);

  /* Misr Toolkit Call */
  status = MtkRegressionCoeffCalc(&srcbuf, &src_mask_buf, &matchbuf, &match_sig_buf, &match_mask_buf, mapinfo, size_factor[0], &regression_coeff, regression_coeff_mapinfo);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Outputs */
  Mtk_toIDLRegCoeffStruct(&regression_coeff, &vregression_coeff );
  MTK_ERR_IDL_COND_JUMP(status);
	
  /* Free MtkDataBuffer's */
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&matchbuf);
  MtkDataBufferFree(&match_sig_buf);
  MtkDataBufferFree(&match_mask_buf);
	
  IDL_VarCopy(vregression_coeff, argv[7]);
  IDL_VarCopy(vregression_coeff_mapinfo, argv[8]);
  IDL_DELTMP(vsize_factor);

  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  MtkDataBufferFree(&srcbuf);
  MtkDataBufferFree(&src_mask_buf);
  MtkDataBufferFree(&matchbuf);
  MtkDataBufferFree(&match_sig_buf);
  MtkDataBufferFree(&match_mask_buf);
  IDL_DELTMP(vsize_factor);
  IDL_DELTMP(vregression_coeff);
  IDL_DELTMP(vregression_coeff_mapinfo);
  return IDL_GettmpLong(status);
}

/* --------------------------------------------------------------- */
/* Mtk_ResampleRegressionCoeff                                     */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ResampleRegressionCoeff( int argc, IDL_VPTR *argv ) {
  MTKt_status status = MTK_FAILURE;

  /* Input argv[0] to argv[2] */
  MTKt_RegressionCoeff reg_coeff = MTKT_REGRESSION_COEFF_INIT;
  IDL_VPTR vreg_coeff;

  MTKt_MapInfo *reg_mapinfo;
  IDL_MEMINT nreg_mapinfo;

  MTKt_MapInfo *target_mapinfo;
  IDL_MEMINT ntarget_mapinfo;

  /* Output argv[3] */
  MTKt_RegressionCoeff reg_coeff_out = MTKT_REGRESSION_COEFF_INIT;
  IDL_VPTR      vreg_coeff_out;

  /* Inputs */
  vreg_coeff = argv[0];
  IDL_toMtkRegCoeffStruct(&vreg_coeff, &reg_coeff);
  IDL_VarGetData(argv[1], &nreg_mapinfo, (char **)&reg_mapinfo, IDL_FALSE);
  IDL_VarGetData(argv[2], &ntarget_mapinfo, (char **)&target_mapinfo, IDL_FALSE);

  /* MISR Toolkit call */
  status = MtkResampleRegressionCoeff(&reg_coeff, reg_mapinfo, target_mapinfo, &reg_coeff_out);
  MTK_ERR_IDL_COND_JUMP(status);

  /* Outputs */
  Mtk_toIDLRegCoeffOutStruct(&reg_coeff_out, &vreg_coeff_out);
  MTK_ERR_IDL_COND_JUMP(status);
    
  IDL_VarCopy(vreg_coeff_out, argv[3]);
  return IDL_GettmpLong(status);

 ERROR_HANDLE:
  return IDL_GettmpLong(status);
  IDL_DELTMP(vreg_coeff_out);
}

/* --------------------------------------------------------------- */
/* Mtk_ErrorMessage		                         				   */
/* --------------------------------------------------------------- */

static IDL_VPTR Mtk_ErrorMessage( int argc, IDL_VPTR *argv ) {

  char *error_msg[] = MTK_ERR_DESC;

  /* Input argv[0] */
  IDL_VPTR verrorcode;
  int *errorcode;
  IDL_MEMINT nerrorcode;

  /* Inputs */
  verrorcode = IDL_BasicTypeConversion(1, &argv[0], IDL_TYP_LONG);
  IDL_VarGetData(verrorcode, &nerrorcode, (char **)&errorcode, IDL_TRUE);

  if (*errorcode >= 0 && *errorcode < sizeof(error_msg)/sizeof(*error_msg)) {
    IDL_DELTMP(verrorcode);
    return IDL_StrToSTRING(error_msg[*errorcode]);
  } else {
    IDL_DELTMP(verrorcode);
    return IDL_StrToSTRING(error_msg[MTK_INVALID_ERROR_CODE]);
  }
}

/* --------------------------------------------------------------- */
/* IDL_Load                                                        */
/* --------------------------------------------------------------- */

int IDL_Load( void ) {

  /* -------------------- */
  /* MisrToolkit routines */
  /* -------------------- */
  /* From docs: defs must be sorted by routine name in ascending lexical order. */
	static IDL_SYSFUN_DEF2 function_addr[] = {
		{ { Mtk_ApplyRegression },			"MTK_APPLY_REGRESSION",		        7, 7, 0, 0 },
		{ { Mtk_Bls_To_LatLon },				"MTK_BLS_TO_LATLON",				7, 7, 0, 0 },
		{ { Mtk_Bls_To_SomXY },      				"MTK_BLS_TO_SOMXY",				7, 7, 0, 0 },
		{ { Mtk_CreateGeoGrid },				"MTK_CREATE_GEOGRID",				8, 8, 0, 0 },
		{ { Mtk_Create_LatLon },				"MTK_CREATE_LATLON",	       			3, 3, 0, 0 },
		{ { Mtk_Datetime_To_Julian },			       	"MTK_DATETIME_TO_JULIAN",			2, 2, 0, 0 },
		{ { Mtk_Dd_To_Deg_Min_Sec },				"MTK_DD_TO_DEG_MIN_SEC",			4, 4, 0, 0 },
		{ { Mtk_Dd_To_Dms },					"MTK_DD_TO_DMS",				2, 2, 0, 0 },
		{ { Mtk_Dd_To_Rad },					"MTK_DD_TO_RAD",				2, 2, 0, 0 },
		{ { Mtk_Deg_Min_Sec_To_Dd },				"MTK_DEG_MIN_SEC_TO_DD",			4, 4, 0, 0 },
		{ { Mtk_Deg_Min_Sec_To_Dms },				"MTK_DEG_MIN_SEC_TO_DMS",			4, 4, 0, 0 },
		{ { Mtk_Deg_Min_Sec_To_Rad },				"MTK_DEG_MIN_SEC_TO_RAD",			4, 4, 0, 0 },
		{ { Mtk_Dms_To_Dd },					"MTK_DMS_TO_DD",				2, 2, 0, 0 },
		{ { Mtk_Dms_To_Deg_Min_Sec },				"MTK_DMS_TO_DEG_MIN_SEC",			4, 4, 0, 0 },
		{ { Mtk_Dms_To_Rad },					"MTK_DMS_TO_RAD",				2, 2, 0, 0 },
		{ { Mtk_Downsample },			"MTK_DOWNSAMPLE",		        5, 5, 0, 0 },
		{ { Mtk_ErrorMessage },				"MTK_ERROR_MESSAGE",				1, 1, 0, 0 },
		{ { Mtk_FieldAttr_Get },             "MTK_FIELDATTR_GET",       4, 4, 0, 0 },
		{ { Mtk_FieldAttr_List },            "MTK_FIELDATTR_LIST",        4, 4, 0, 0 },
		{ { Mtk_FileAttr_Get },     				"MTK_FILEATTR_GET",				3, 3, 0, 0 },
		{ { Mtk_FileAttr_List },     				"MTK_FILEATTR_LIST",				3, 3, 0, 0 },
		{ { Mtk_File_Block_Meta_Field_List },           	"MTK_FILE_BLOCK_META_FIELD_LIST",	   	4, 4, 0, 0 },
		{ { Mtk_File_Block_Meta_Field_Read },             	"MTK_FILE_BLOCK_META_FIELD_READ",	       	4, 4, 0, 0 },
		{ { Mtk_File_Block_Meta_List },             		"MTK_FILE_BLOCK_META_LIST",	       	       	3, 3, 0, 0 },
		{ { Mtk_File_CoreMetaData_Get },             		"MTK_FILE_COREMETADATA_GET",	       	       	3, 3, 0, 0 },
		{ { Mtk_File_CoreMetaData_Query },      		"MTK_FILE_COREMETADATA_QUERY",	       	       	3, 3, 0, 0 },
		{ { Mtk_File_Grid_Field_Check },		        "MTK_FILE_GRID_FIELD_CHECK",		        3, 3, 0, 0 },
		{ { Mtk_File_Grid_Field_To_Datatype },		"MTK_FILE_GRID_FIELD_TO_DATATYPE",		4, 4, 0, 0 },
		{ { Mtk_File_Grid_Field_To_DimList }, 		"MTK_FILE_GRID_FIELD_TO_DIMLIST",		6, 6, 0, 0 },
		{ { Mtk_File_Grid_To_FieldList }, 			"MTK_FILE_GRID_TO_FIELDLIST",			4, 4, 0, 0 },
		{ { Mtk_File_Grid_To_Native_FieldList }, 		"MTK_FILE_GRID_TO_NATIVE_FIELDLIST",		4, 4, 0, 0 },
		{ { Mtk_File_Grid_To_Resolution }, 			"MTK_FILE_GRID_TO_RESOLUTION",			3, 3, 0, 0 },
		{ { Mtk_File_LGID },					"MTK_FILE_LGID",				2, 2, 0, 0 },
		{ { Mtk_File_To_BlockRange },				"MTK_FILE_TO_BLOCKRANGE",			3, 3, 0, 0 },
		{ { Mtk_File_To_GridList },				"MTK_FILE_TO_GRIDLIST",				3, 3, 0, 0 },
		{ { Mtk_File_To_Orbit },     				"MTK_FILE_TO_ORBIT",				2, 2, 0, 0 },
		{ { Mtk_File_To_Path },      				"MTK_FILE_TO_PATH",				2, 2, 0, 0 },
		{ { Mtk_File_Type },					"MTK_FILE_TYPE",				2, 2, 0, 0 },
		{ { Mtk_File_Version },      				"MTK_FILE_VERSION",				2, 2, 0, 0 },
		{ { Mtk_FillValue_Get },      	       		"MTK_FILLVALUE_GET",	       			4, 4, 0, 0 },
		{ { Mtk_Find_FileList },				"MTK_FIND_FILELIST",				8, 8, 0, 0 },
		{ { Mtk_GridAttr_Get },     				"MTK_GRIDATTR_GET",				4, 4, 0, 0 },
		{ { Mtk_GridAttr_List },     				"MTK_GRIDATTR_LIST",				4, 4, 0, 0 },
		{ { Mtk_Julian_To_Datetime },		       		"MTK_JULIAN_TO_DATETIME",			2, 2, 0, 0 },
		{ { Mtk_LS_To_LatLon },      				"MTK_LS_TO_LATLON",				5, 5, 0, 0 },
		{ { Mtk_LS_To_SomXY },       				"MTK_LS_TO_SOMXY",				5, 5, 0, 0 },
		{ { Mtk_LatLon_To_Bls },				"MTK_LATLON_TO_BLS",				7, 7, 0, 0 },
		{ { Mtk_LatLon_To_LS },      				"MTK_LATLON_TO_LS",				5, 5, 0, 0 },
		{ { Mtk_LatLon_To_PathList },				"MTK_LATLON_TO_PATHLIST",			4, 4, 0, 0 },
		{ { Mtk_LatLon_To_SomXY },				"MTK_LATLON_TO_SOMXY",				5, 5, 0, 0 },
		{ { Mtk_LinearRegressionCalc },			"MTK_LINEAR_REGRESSION_CALC",		        7, 7, 0, 0 },
		{ { Mtk_Make_Filename },				"MTK_MAKE_FILENAME",				7, 7, 0, 0 },
		{ { Mtk_Orbit_To_Path },				"MTK_ORBIT_TO_PATH",				2, 2, 0, 0 },
		{ { Mtk_Orbit_To_TimeRange },				"MTK_ORBIT_TO_TIMERANGE",			3, 3, 0, 0 },
		{ { Mtk_Path_BlockRange_To_BlockCorners },	       	"MTK_PATH_BLOCKRANGE_TO_BLOCKCORNERS",	       	4, 4, 0, 0 },
		{ { Mtk_Path_TimeRange_To_OrbitList }, 		"MTK_PATH_TIMERANGE_TO_ORBITLIST",		5, 5, 0, 0 },
		{ { Mtk_Path_To_ProjParam },				"MTK_PATH_TO_PROJPARAM",			3, 3, 0, 0 },
		{ { Mtk_Pixel_Time },					"MTK_PIXEL_TIME",	       			4, 4, 0, 0 },
		{ { Mtk_Rad_To_Dd },					"MTK_RAD_TO_DD",				2, 2, 0, 0 },
		{ { Mtk_Rad_To_Deg_Min_Sec },				"MTK_RAD_TO_DEG_MIN_SEC",			4, 4, 0, 0 },
		{ { Mtk_Rad_To_Dms },					"MTK_RAD_TO_DMS",				2, 2, 0, 0 },
		{ { Mtk_ReadBlock },					"MTK_READBLOCK",				5, 5, 0, 0 },
		{ { Mtk_ReadBlockRange },				"MTK_READBLOCKRANGE",				6, 6, 0, 0 },
		{ { Mtk_ReadData },					"MTK_READDATA",					5, 6, 0, 0 },
		{ { Mtk_ReadRaw },					"MTK_READRAW",					5, 6, 0, 0 },
		{ { Mtk_Region_Path_To_BlockRange }, 			"MTK_REGION_PATH_TO_BLOCKRANGE",		4, 4, 0, 0 },
		{ { Mtk_Region_To_PathList },				"MTK_REGION_TO_PATHLIST",			3, 3, 0, 0 },
		{ { Mtk_RegressionCoeffCalc },			"MTK_REGRESSION_COEFF_CALC",		        9, 9, 0, 0 },
		{ { Mtk_ResampleCubicConvolution },			"MTK_RESAMPLE_CUBICCONVOLUTION",		        7, 7, 0, 0 },
		{ { Mtk_ResampleNearestNeighbor },			"MTK_RESAMPLE_NEARESTNEIGHBOR",		        4, 4, 0, 0 },
		{ { Mtk_ResampleRegressionCoeff },			"MTK_RESAMPLE_REGRESSION_COEFF",		        4, 4, 0, 0 },
		{ { Mtk_SetRegion_By_LatLon_Extent }, 		"MTK_SETREGION_BY_LATLON_EXTENT",		6, 6, 0, 0 },
		{ { Mtk_SetRegion_By_Path_BlockRange }, 		"MTK_SETREGION_BY_PATH_BLOCKRANGE",		4, 4, 0, 0 },
		{ { Mtk_SetRegion_By_Path_Som_Ulc_Lrc }, 			"MTK_SETREGION_BY_PATH_SOM_ULC_LRC",			6, 6, 0, 0 },
		{ { Mtk_SetRegion_By_Ulc_Lrc }, 			"MTK_SETREGION_BY_ULC_LRC",			5, 5, 0, 0 },
		{ { Mtk_SmoothData },			"MTK_SMOOTH_DATA",		        5, 5, 0, 0 },
		{ { Mtk_Snap_To_Grid },      				"MTK_SNAP_TO_GRID",				4, 4, 0, 0 },
		{ { Mtk_SomXY_To_Bls },      				"MTK_SOMXY_TO_BLS",				7, 7, 0, 0 },
		{ { Mtk_SomXY_To_LS },       				"MTK_SOMXY_TO_LS",				5, 5, 0, 0 },
		{ { Mtk_SomXY_To_LatLon },				"MTK_SOMXY_TO_LATLON",				5, 5, 0, 0 },
		{ { Mtk_TimeRange_To_OrbitList }, 			"MTK_TIMERANGE_TO_ORBITLIST",			4, 4, 0, 0 },
		{ { Mtk_Time_Meta_Read },				"MTK_TIME_META_READ",				2, 2, 0, 0 },
		{ { Mtk_Time_To_Orbit_Path },				"MTK_TIME_TO_ORBIT_PATH",			3, 3, 0, 0 },
		{ { Mtk_TransformCoordinates },			"MTK_TRANSFORM_COORDINATES",			5, 5, 0, 0 },
		{ { Mtk_UpsampleMask },			"MTK_UPSAMPLE_MASK",		        3, 3, 0, 0 },
		{ { Mtk_Version },					"MTK_VERSION",					0, 0, 0, 0 },
		{ { Mtk_WriteEnviFile },				"MTK_WRITE_ENVI_FILE",				6, 6, 0, 0 }
	};
  
 
  char err_msg[128];

  /* ----------------------------------------------------------- */
  /* Compare the versions of the IDL and C MisrToolkit libraries */
  /* and don't load Mtk IDL dlm module if different              */
  /* ----------------------------------------------------------- */

  if (strcmp(MTK_VERSION, MtkVersion()) != 0) {
    sprintf(err_msg, "IDL MISR Toolkit DLM V%s does not match MISR Toolkit Library V%s.",
	    MTK_VERSION, MtkVersion());
    IDL_Message(IDL_M_GENERIC, IDL_MSG_RET, err_msg);
    return IDL_FALSE;
  }
  /* --------------------------------------------------- */
  /* Create a message block to hold MisrToolkit messages */
  /* --------------------------------------------------- */

  if (!(msg_block = IDL_MessageDefineBlock("MisrToolkit",
                                           IDL_CARRAY_ELTS(msg_arr), msg_arr)))
    return IDL_FALSE;

  /* ---------------------------- */
  /* Register MisrToolkit routine */
  /* ---------------------------- */

  return IDL_SysRtnAdd(function_addr, IDL_TRUE,
		       IDL_CARRAY_ELTS(function_addr));
}
