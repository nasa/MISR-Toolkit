/*===========================================================================
=                                                                           =
=                            read_attributes                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <string>
#include "DAS.h"
#include "MisrToolkit.h"

static const char *error_msg[] = MTK_ERR_DESC;

/* ------------------------------------------------------------------------ */
/*  Create and return DAS for a MISR file                                   */
/* ------------------------------------------------------------------------ */

void read_attributes(DAS &das, const string &filename) throw (Error)
{
  AttrTable *attr_table, *attr_table2;
  int iattr, nattrs;
  char **attrlist = NULL;
  int igrid, ngrids;
  char **gridlist = NULL;
  int ifield, nfields;
  char **fieldlist = NULL;
  int iblockmeta, nblockmeta;
  char **blockmetalist = NULL;
  int iparam, nparams;
  char **paramlist = NULL;
  MTKt_status status;

#ifdef DEBUG
  std::cerr << "Mtk[read_attributes()]: " << filename << std::endl;
#endif

  try {

/* -------------------------------------------------------------------- */
/* Create File Attributes table						*/
/* -------------------------------------------------------------------- */

    attr_table = das.add_table( string("FileAttributes"), new AttrTable );

    status = MtkFileAttrList(filename.c_str(), &nattrs, &attrlist);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileAttrList(): "+string(error_msg[status]));

    for (iattr = 0; iattr < nattrs; iattr++) {
      attr_table->append_attr(string(attrlist[iattr]), string("String"),
			      string("hello"));
    }
    MtkStringListFree(nattrs, &attrlist);
    attrlist = NULL;

/* -------------------------------------------------------------------- */
/* Create Grid Attributes table						*/
/* -------------------------------------------------------------------- */

    attr_table = das.add_table( string("GridAttributes"), new AttrTable );

    status = MtkFileToGridList(filename.c_str(), &ngrids, &gridlist);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileToGridList(): "+string(error_msg[status]));

    for (igrid = 0; igrid < ngrids; igrid++) {

      attr_table2 = attr_table->append_container(string(gridlist[igrid]));

      status = MtkGridAttrList(filename.c_str(), gridlist[igrid], &nattrs,
			       &attrlist);
      if (status != MTK_SUCCESS)
	throw Error("MtkGridAttrList(): "+string(error_msg[status]));

      for (iattr = 0; iattr < nattrs; iattr++) {
	attr_table2->append_attr(string(attrlist[iattr]), string("String"),
				 string("hello"));
      }
      MtkStringListFree(nattrs, &attrlist);
      attrlist = NULL;
    }
    MtkStringListFree(ngrids, &gridlist);
    gridlist = NULL;

/* -------------------------------------------------------------------- */
/* Create Block Attributes table				      	*/
/* -------------------------------------------------------------------- */

    attr_table = das.add_table( string("BlockAttributes"), new AttrTable );

    status = MtkFileBlockMetaList(filename.c_str(), &nblockmeta,
				  &blockmetalist);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileBlockMetaList(): "+string(error_msg[status]));

    for (iblockmeta = 0; iblockmeta < nblockmeta; iblockmeta++) {

      attr_table2 =
	attr_table->append_container(string(blockmetalist[iblockmeta]));

      status = MtkFileBlockMetaFieldList(filename.c_str(),
					 blockmetalist[iblockmeta],
					 &nfields, &fieldlist);
      if (status != MTK_SUCCESS)
	throw Error("MtkFileBlockMetaFieldList(): "+string(error_msg[status]));

      for (ifield = 0; ifield < nfields; ifield++) {

	attr_table2->append_attr(string(fieldlist[ifield]),
				 string("String"), string("hello"));
      }
      MtkStringListFree(nfields, &fieldlist);
      fieldlist = NULL;
    }
    MtkStringListFree(nblockmeta, &blockmetalist);
    blockmetalist = NULL;

/* -------------------------------------------------------------------- */
/* Create Core MetaData table						*/
/* -------------------------------------------------------------------- */

    attr_table = das.add_table( string("CoreMetaData"), new AttrTable );

    status = MtkFileCoreMetaDataQuery(filename.c_str(), &nparams, &paramlist);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileCoreMetaDataQuery(): "+string(error_msg[status]));

    for (iparam = 0; iparam < nparams; iparam++) {

      attr_table->append_attr(string(paramlist[iparam]),
			      string("String"), string("hello"));
    }
    MtkStringListFree(nparams, &paramlist);
    paramlist = NULL;

  }
  catch (const Error &e) {
    if (attrlist != NULL)
      MtkStringListFree(nattrs, &attrlist);
    if (gridlist != NULL)
      MtkStringListFree(ngrids, &gridlist);
    if (fieldlist != NULL)
      MtkStringListFree(nfields, &fieldlist);
    if (blockmetalist != NULL)
      MtkStringListFree(nblockmeta, &blockmetalist);
    if (paramlist != NULL)
      MtkStringListFree(nparams, &paramlist);
    throw e;
  }
}
