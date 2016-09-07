extern "C" {
#include <hdf.h>		// Definition of int32
}
#include "exception.h"

void Exception::add_hdf_error() 
{
  if(HEvalue(1) != DFE_NONE)
    what_ = what_ + HEstring((hdf_err_code_t) HEvalue(1));
}
