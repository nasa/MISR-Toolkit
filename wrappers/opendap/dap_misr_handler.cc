/*===========================================================================
=                                                                           =
=                            dap_misr_handler                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <iostream>
#include <string>

#include "DDS.h"
#include "cgi_util.h"
#include "DODSFilter.h"
#include "ConstraintEvaluator.h"
#include "MisrToolkit.h"

extern void read_descriptors(DDS &dds, const string &filename)
  throw (Error);
extern void read_attributes(DAS &dds, const string &filename)
  throw (Error);

const string cgi_version = "misr-dods/"MTK_VERSION;

/* ------------------------------------------------------------------------ */
/*                               main()                                     */
/* ------------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
#ifdef DEBUG
  std::cerr << "Mtk[misr_handler()]: ";
  for( int i = 0; i < argc; i++ )
    std::cerr << argv[i] << " ";
  std::cerr << std::endl;
#endif

  try {
    DODSFilter df(argc, argv);
    if (df.get_cgi_version() == "")
      df.set_cgi_version(cgi_version);

/* ------------------------------------------------------------------------ */
/*  Switch based on the request made.					    */
/* ------------------------------------------------------------------------ */

    switch (df.get_response()) {

    case DODSFilter::DAS_Response: {
      DAS das;

      read_attributes(das, df.get_dataset_name());
      df.read_ancillary_das(das);
      df.send_das(das);
      break;
    }

    case DODSFilter::DDS_Response: {
      DDS dds( NULL );
      ConstraintEvaluator ce;

      read_descriptors(dds, df.get_dataset_name());
      df.read_ancillary_dds(dds);
      df.send_dds(dds, ce, true);
      break;
    }

    case DODSFilter::DataDDS_Response: {
      DDS dds( NULL );
      ConstraintEvaluator ce;

      dds.filename(df.get_dataset_name());
      read_descriptors(dds, df.get_dataset_name());
      df.read_ancillary_dds(dds);
      df.send_data(dds, ce, stdout);
      break;
    }

    case DODSFilter::DDX_Response: {
      DDS dds( NULL );
      DAS das;
      ConstraintEvaluator ce;

      dds.filename(df.get_dataset_name());
      read_descriptors(dds, df.get_dataset_name());
      df.read_ancillary_dds(dds);
      read_attributes(das, df.get_dataset_name());
      df.read_ancillary_das(das);
      dds.transfer_attributes(&das);
      df.send_ddx(dds, ce, stdout);
      break;
    }

    case DODSFilter::Version_Response: {
      df.send_version_info();
      break;
    }

    default:
      df.print_usage();	// Throws Error
    }
  }
  catch (Error &e) {
    set_mime_text(stdout, dods_error, cgi_version);
    e.print(stdout);
    return 1;
  }

  return 0;
}
