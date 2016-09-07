// This contains HdfFile stuff. There is enough state information
// that we wrap this is a class rather than just having C functions
// as we did with the MtkFile stuff.

#ifndef HDF_FILE_H
#define HDF_FILE_H
extern "C" {
#include <hdf.h>		// Definition of int32
#include "mfhdf.h"		// Prototype of SDxxx functions.
				// Bunch of symbols compile complains
				// about.
#undef HAVE_PROTOTYPES
#undef PACKAGE_NAME		
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef PACKAGE_STRING
#undef PACKAGE_BUGREPORT
#include <HdfEosDef.h>		// Prototyoe of HDF routines.
}
#include <string>		// Definition of string.
#include <iostream>		// Definition of std::cerr
#include <vector>
#include <map>
#include <set>
#include "exception.h"		// Definition of Exception.
#include "hdf_attribute.h"
// #define DEBUG t

class HdfGrid;
class HdfVdata;
class HdfFile {
public:
  inline HdfFile() {gid_ = -1;}
  void open(char* Fname);
  void reg(HdfGrid* G) {grid_.insert(G);}
  void dereg(HdfGrid* G) {grid_.erase(G);}
  void reg(HdfVdata* V) {vdata_.insert(V);}
  void dereg(HdfVdata* V) {vdata_.erase(V);}
  inline const std::string& grid_list() const { return grid_list_; }
  inline const std::vector<std::string>& sds_list() const { return sds_list_; }
  inline const std::vector<int32>& sds_dim(const std::string Sname) const
  { 
    if(sds_dim_.count(Sname) < 1)
      throw Exception("Bad SDS name " + Sname + " for file " + fname_);
    return (*sds_dim_.find(Sname)).second;
  }
  inline const std::vector<std::string>& attribute_list() const 
      {return att_list_; }
  inline int32 attribute_data_type(const std::string& Aname) const
  {
    if(att_data_type_.count(Aname) < 1)
      throw Exception("Bad Attribute name " + Aname + " for file " + fname_);
    return (*att_data_type_.find(Aname)).second;
  }
  inline int32 attribute_data_size(const std::string& Aname) const
  {
    if(att_data_type_.count(Aname) < 1)
      throw Exception("Bad Attribute name " + Aname + " for file " + fname_);
    return (*att_data_size_.find(Aname)).second;
  }
  template<class T> inline T read_attribute(const std::string& Att_name) const 
  {
    Attribute::Helper<T> h;
    int32 aindex = SDfindattr(sid_, const_cast<char*>(Att_name.c_str()));
    if(aindex ==-1)
      throw Exception("Trouble reading SDS attribute for file " + fname_);
    int32 nvalue = attribute_data_size(Att_name);
    if(h.must_be_one() &&
     nvalue != 1)
      throw Exception("Trouble reading SDS attribute for file " + fname_);
    T val = h.create(nvalue);
    intn status = SDreadattr(sid_, aindex, h.pointer(val));
    if(status)
      throw Exception("Trouble reading SDS attribute for file " + fname_);
    return val;
  }
  inline int32 sds_data_type(const std::string& Sname) const
  { 
    if(sds_data_type_.count(Sname) < 1)
      throw Exception("Bad SDS name " + Sname + " for file " + fname_);
    return (*sds_data_type_.find(Sname)).second;
  }
  void sds_read(char* Sds_name, void* Buf, int32* Start, int32* Stride,
	        int32* Edge) const;
  std::set<std::string> vdata_list() const;
  inline int32 gid() const {return gid_;}
  inline int32 sid() const {return sid_;}
  inline int32 hid() const {return hid_;}
  inline const std::string& file_name() const {return fname_;}
  void close();
  inline ~HdfFile() { close();}
private:
  static int count_open_;
  int32	gid_;
  int32	hid_;
  int32	sid_;
  std::set<HdfGrid*> grid_;
  std::set<HdfVdata*> vdata_;
  std::string grid_list_;
  std::string fname_;
  std::vector<std::string> sds_list_;
  std::map<std::string, int32> sds_data_type_;
  std::vector<std::string> att_list_;
  std::map<std::string, int32> att_data_type_;
  std::map<std::string, int32> att_data_size_;
  std::map<std::string, std::vector<int32> > sds_dim_;
};

#endif
