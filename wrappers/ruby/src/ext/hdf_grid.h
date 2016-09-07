// This contains HdfFile stuff. There is enough state information
// that we wrap this is a class rather than just having C functions
// as we did with the MtkFile stuff.

#ifndef HDF_GRID_H
#define HDF_GRID_H
#include "hdf_file.h"
extern "C" {
#include "ruby.h"
intn
GDSDfldsrch(int32 gridID, int32 sdInterfaceID, char *fieldname, int32 * sdid,
	    int32 * rankSDS, int32 * rankFld, int32 * offset, int32 dims[],
	    int32 * solo);
}

class HdfGrid {
public:
  HdfGrid() {grid_id_ = -1; }
  void open(VALUE Hdf_obj, char* name);
  void open2(HdfFile* h, char* name);
  void close();
  inline ~HdfGrid() {close(); }
  inline const std::string& field_list() const { return field_list_;}
  inline const std::vector<std::string>& attribute_list() const 
      {return att_list_; }
  inline int32 attribute_data_type(const std::string& Aname) const
  {
    if(att_data_type_.count(Aname) < 1)
      throw Exception("Bad Attribute name " + Aname + " for file " + h_->file_name());
    return (*att_data_type_.find(Aname)).second;
  }
  inline int32 attribute_data_size(const std::string& Aname) const
  {
    if(att_data_type_.count(Aname) < 1)
      throw Exception("Bad Attribute name " + Aname + " for file " + h_->file_name());
    return (*att_data_size_.find(Aname)).second;
  }
  template<class T> inline T read_attribute(const std::string& Att_name) const 
  {
    Attribute::Helper<T> h;
    int32 nvalue = attribute_data_size(Att_name);
    if(h.must_be_one() &&
     nvalue != 1)
      throw Exception("Trouble reading grid attribute for file " + h_->file_name());
    T val = h.create(nvalue);
    intn status = GDreadattr(grid_id_, const_cast<char*>(Att_name.c_str()), 
			     h.pointer(val));
    if(status)
      throw Exception("Trouble reading grid attribute for file " + h_->file_name());
    return val;
  }
  void populate_field_info(char* Field_name);
  void set_chunk_cache(char* Field_name, int Number_chunk);
  void field_read(char* Field_name, void* Buf, int32* Start, int32* Stride,
		  int32* Edge) const;
  inline VALUE hdf_obj() const {return hdf_obj_;}
  inline int32 rank() const {return rank_;}
  inline const int32* dim() const {return dim_;}
  inline const char* dim_name_buf() const {return dim_name_buf_;}
  inline int32 data_type() const {return data_type_;}
  inline int32 tile_code() const {return tile_code_;}
  inline int32 tile_rank() const {return tile_rank_;}
  inline const int32* tile_dim() const {return tile_dim_;}
private:
  VALUE hdf_obj_;
  int32 rank_;
  int32 dim_[MAX_VAR_DIMS];
  char dim_name_buf_[HDFE_DIMBUFSIZE];
  int32 data_type_;
  int32 tile_code_, tile_rank_;
  int32 tile_dim_[MAX_VAR_DIMS];
  int32 grid_id_;
  std::string gname_;
  std::string field_list_;
  std::vector<std::string> att_list_;
  std::map<std::string, int32> att_data_type_;
  std::map<std::string, int32> att_data_size_;
  HdfFile* h_;
};
#endif
