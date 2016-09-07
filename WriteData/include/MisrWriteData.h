/*===========================================================================
=                                                                           =
=                              MisrWriteData                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRWRITEDATA_H
#define MISRWRITEDATA_H

#include "MisrError.h"
#include "MisrMapQuery.h"
#include "MisrUtil.h"


MTKt_status MtkWriteBinFile( const char *filename,
			     MTKt_DataBuffer buf,
			     MTKt_MapInfo mapinfo );

MTKt_status MtkWriteBinFile3D( const char *filename,
			       MTKt_DataBuffer3D buf );

MTKt_status MtkWriteEnviFile( const char *filename,
			      MTKt_DataBuffer buf,
			      MTKt_MapInfo mapinfo,
			      const char *misrfilename,
			      const char *misrgridname,
			      const char *misrfieldname );

#endif /* MISRWRITEDATA_H */
