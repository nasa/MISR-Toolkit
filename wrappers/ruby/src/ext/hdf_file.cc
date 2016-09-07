#include "hdf_file.h"
#include "hdf_grid.h"
#include "hdf_vdata.h"
#include <sstream>

int HdfFile::count_open_ = 0;

void HdfFile::open(char* Fname) {
  if(count_open_ >= 31) {
    std::ostringstream os;
    os << "Trouble opening file " << fname_ << "\n"
       << "Too many files open. HDF limits the number of open files\n"
       << "to 32. You can use File#close to close some open files";
    throw Exception(os.str());
  }
  fname_ = std::string(Fname);
  gid_ = GDopen(Fname, DFACC_READ);
#ifdef DEBUG
  std::cerr << "Opening file " << Fname << " gid = " << gid_ << "\n";
#endif
  if(gid_ == -1)
    throw Exception("Trouble opening file " + fname_);
  count_open_++;
  int32 buf_size;
  intn status = EHidinfo(gid_, &hid_, &sid_);
  if(status == -1)
    throw Exception("Trouble reading grid for file " + fname_);
  status = GDinqgrid(Fname, 0, &buf_size);
  if(status == -1)
    throw Exception("Trouble reading grid for file " + fname_);
  std::vector<char> buf(buf_size + 1);
  status = GDinqgrid(Fname, &(*buf.begin()), &buf_size);
  if(status == -1)
    throw Exception("Trouble reading grid for file " + fname_);
  grid_list_ = std::string(buf.begin(), buf.end());
  int32 ndataset, nfileattr;
  status = SDfileinfo(sid(), &ndataset, &nfileattr);
  if(status)
    throw Exception("Trouble reading SDS for file " + fname_);
  for(int32 i = 0; i < ndataset; ++i) {
    int32 sds_id = SDselect(sid_, i);
    int32 dim_sizes[MAX_VAR_DIMS];
    int32 rank, data_type, n_attrs;
    char name[MAX_NC_NAME];
    status = SDgetinfo(sds_id, name, &rank, dim_sizes, &data_type,
		       &n_attrs);
    if(status)
      throw Exception("Trouble reading SDS for file " + fname_);
    status = SDendaccess(sds_id);
    sds_id = -1;
    if(status)
      throw Exception("Trouble reading SDS for file " + fname_);
    std::string n(name);
    sds_list_.push_back(n);
    std::vector<int32> ds(dim_sizes, dim_sizes + rank);
    sds_dim_[n] = ds;
    sds_data_type_[n] = data_type;
  }
  for(int32 i = 0; i < nfileattr; ++i) {
    char att_name[MAX_NC_NAME];
    int32 data_type, nvalue;
    status = SDattrinfo(sid_, i, att_name, &data_type, &nvalue);
    if(status)
      throw Exception("Trouble reading SDS attribute for file " + fname_);
    att_list_.push_back(std::string(att_name));
    att_data_type_[std::string(att_name)] = data_type;
    att_data_size_[std::string(att_name)] = nvalue;
  }
}

void HdfFile::close()
{
  if(gid_ != -1) {
    std::set<HdfGrid*> grid_copy = grid_;
    for(std::set<HdfGrid*>::iterator i = grid_copy.begin();
	i != grid_copy.end(); ++i)
      (*i)->close();
    std::set<HdfVdata*> vdata_copy = vdata_;
    for(std::set<HdfVdata*>::iterator i = vdata_copy.begin();
	i != vdata_copy.end(); ++i)
      (*i)->close();
#ifdef DEBUG
    std::cerr << "Closing file " << fname_ << " gid = " << gid_ << "\n";
#endif
    GDclose(gid_);
    count_open_--;
    gid_ = -1;
  }
}

void HdfFile::sds_read(char* Field_name, void* Buf, int32* Start, 
			 int32* Stride, int32* Edge) const
{
  int32 index = SDnametoindex(sid_, Field_name);
  if(index ==-1)
    throw Exception("Trouble reading SDS " + 
		    std::string(Field_name) +
		    " for file " + fname_);
  int32 sds_id = SDselect(sid_, index);
  if(sds_id ==-1)
    throw Exception("Trouble reading SDS " + 
		    std::string(Field_name) +
		    " for file " + fname_);
  intn status = SDreaddata(sds_id, Start, Stride, Edge, Buf);
  if(status)
    throw Exception("Trouble reading SDS " + 
		    std::string(Field_name) +
		    " for file " + fname_);
}

std::set<std::string> HdfFile::vdata_list() const
{
  int32 vdata_ref = -1;
  std::set<std::string> res;
  while((vdata_ref = VSgetid(hid(), vdata_ref)) !=-1) {
    int32 vid = -1;
    try {
      vid = VSattach(hid(), vdata_ref, "r");
      if(vid ==-1)
	throw Exception("Trouble reading vdata for file " + fname_);
      char vdata_name[VSNAMELENMAX + 1];
      int32 hdf_status = VSgetname(vid, vdata_name);
      if(hdf_status ==-1)
	throw Exception("Trouble reading vdata for file " + fname_);
      hdf_status = VSdetach(vid);
      vid = -1;
      if(hdf_status ==-1)
	throw Exception("Trouble reading vdata for file " + fname_);
      res.insert(std::string(vdata_name));
    } catch(...) {
      if(vid != -1)
	VSdetach(vid);
      throw;
    }
  }
  return res;
}
