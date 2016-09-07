// This contains HdfFile stuff. There is enough state information
// that we wrap this is a class rather than just having C functions
// as we did with the MtkFile stuff.

#ifndef HDF_VDATA_H
#define HDF_VDATA_H
#include "hdf_file.h"
extern "C" {
#include "ruby.h"
}

class HdfVdata {
public:
  HdfVdata() {vdata_id_ = -1; }
  void open(VALUE Hdf_obj, char* name);
  void close();
  inline ~HdfVdata() {close();}
  inline VALUE hdf_obj() const {return hdf_obj_;}
  inline int32 num_rec() const {return num_rec_;}
  inline const std::string& field_list() const {return field_list_;}
  VALUE read(char* field_name, int rec_i, int num_read) const;
private:
  VALUE hdf_obj_;
  int32 vdata_id_;
  int32 num_rec_;
  std::string field_list_;
  std::string vname_;
  HdfFile* h_;
};
#endif
