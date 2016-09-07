/*===========================================================================
=                                                                           =
=                           read_descriptors                                =
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
#include "DDS.h"
#include "cgi_util.h"
#include "misr_types.h"
#include "MisrToolkit.h"

static const char *error_msg[] = MTK_ERR_DESC;

BaseType *misr_daptype(const char *filename, const char *gridname,
		       const char *fieldname);

/* ------------------------------------------------------------------------ */
/*  Create and return DDS for a MISR file                                   */
/* ------------------------------------------------------------------------ */

void read_descriptors(DDS &dds, const string &filename) throw (Error)
{
  BaseType *bt;
  Array *ar;
  Structure *st;
  int igrid, ngrids;
  char **gridlist = NULL;
  int ifield, nfields;
  char **fieldlist = NULL;
  int idim, ndims;
  char **dimlist = NULL;
  int *dimsize = NULL;
  int path, start_block, end_block, resolution;
  MTKt_Region region = MTKT_REGION_INIT;
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
  MTKt_status status;

#ifdef DEBUG
  std::cerr << "Mtk[read_descriptors()]: " << filename << std::endl;
#endif

  try {

/* ------------------------------------------------------------------------ */
/* Set name of dataset							    */
/* ------------------------------------------------------------------------ */

    dds.set_dataset_name(name_path(filename));

/* ------------------------------------------------------------------------ */
/* Query filename for datasets using MisrToolkit and create OpenDAP dataset */
/* ------------------------------------------------------------------------ */

    status = MtkFileToPath(filename.c_str(), &path);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileToPath(): "+string(error_msg[status]));

    status = MtkFileToBlockRange(filename.c_str(), &start_block, &end_block);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileToBlockRange(): "+string(error_msg[status]));

    status = MtkSetRegionByPathBlockRange(path, start_block, end_block,
					  &region);
    if (status != MTK_SUCCESS)
      throw Error("MtkSetRegionByPathBlockRange(): "+string(error_msg[status]));

    status = MtkFileToGridList(filename.c_str(), &ngrids, &gridlist);
    if (status != MTK_SUCCESS)
      throw Error("MtkFileToGridList(): "+string(error_msg[status]));

    for (igrid = 0; igrid < ngrids; igrid++) {

      string gridname = string(gridlist[igrid]);

      st = NewStructure(string(gridname));

      status = MtkFileGridToResolution(filename.c_str(), gridlist[igrid],
				       &resolution);
      if (status != MTK_SUCCESS)
	throw Error("MtkFileGridToResolution(): "+string(error_msg[status]));

      status = MtkSnapToGrid(path, resolution, region, &mapinfo);
      if (status != MTK_SUCCESS)
	throw Error("MtkSnapToGrid(): "+string(error_msg[status]));

      status = MtkFileGridToFieldList(filename.c_str(), gridlist[igrid],
				      &nfields, &fieldlist);
      if (status != MTK_SUCCESS)
	throw Error("MtkFileGridToFieldList(): "+string(error_msg[status]));

      for (ifield = 0; ifield < nfields; ifield++) {

	status = MtkFileGridFieldToDimList(filename.c_str(), gridlist[igrid], 
					   fieldlist[ifield],
					   &ndims, &dimlist, &dimsize);
	if (status != MTK_SUCCESS)
	  throw Error("MtkFileGridFieldToDimList(): "+string(error_msg[status]));

	string fieldname = string(fieldlist[ifield]);

	bt = misr_daptype(filename.c_str(), gridlist[igrid], fieldlist[ifield]);
	ar = NewArray(fieldname);
	ar->add_var(bt);
	ar->append_dim(mapinfo.nline, "NLine");
	ar->append_dim(mapinfo.nsample, "NSample");
	for (idim = 0; idim < ndims; idim++) {
	  ar->append_dim(dimsize[idim], dimlist[idim]);
	}
	dynamic_cast<MISRArray*>(ar)->set_mapinfo(mapinfo);
	st->add_var(ar);

	MtkStringListFree(ndims, &dimlist);
	dimlist = NULL;
	free(dimsize);
	dimsize = NULL;
      }

      bt = NewFloat64(gridname+string("Latitude"));
      ar = NewArray(gridname+string("Latitude"));
      ar->add_var(bt);
      ar->append_dim(mapinfo.nline, "NLine");
      ar->append_dim(mapinfo.nsample, "NSample");
      dynamic_cast<MISRArray*>(ar)->set_mapinfo(mapinfo);
      st->add_var(ar);

      bt = NewFloat64(gridname+string("Longitude"));
      ar = NewArray(gridname+string("Longitude"));
      ar->add_var(bt);
      ar->append_dim(mapinfo.nline, "NLine");
      ar->append_dim(mapinfo.nsample, "NSample");
      dynamic_cast<MISRArray*>(ar)->set_mapinfo(mapinfo);
      st->add_var(ar);

      dds.add_var(st);

      MtkStringListFree(nfields, &fieldlist);
      fieldlist = NULL;
    }
    MtkStringListFree(ngrids, &gridlist);
    gridlist = NULL;
  }
  catch (const Error &e) {
    if (gridlist != NULL)
      MtkStringListFree(ngrids, &gridlist);
    if (fieldlist != NULL)
      MtkStringListFree(nfields, &fieldlist);
    if (dimlist != NULL)
      MtkStringListFree(ndims, &dimlist);
    if (dimsize != NULL)
      free(dimsize);
    throw e;
  }
}
