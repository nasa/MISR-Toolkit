#include "hdf_grid.h"

void HdfGrid::close() 
{ 
  if(grid_id_ != -1) {
#ifdef DEBUG
    std::cerr << "Closing grid " << gname_ << " for file " << h_->file_name()
	      << " gid = " << h_->gid()
	      << " grid_id = " << grid_id_ << "\n";
#endif
    GDdetach(grid_id_);
    grid_id_ = -1;
    h_->dereg(this);
  }
}

void HdfGrid::open(VALUE Hdf_obj, char* name) 
{
  hdf_obj_ = Hdf_obj;
  HdfFile* h;
  Data_Get_Struct(hdf_obj_, HdfFile, h);
  open2(h, name);
}

void HdfGrid::open2(HdfFile* h, char* name) {
  h_ = h;
  gname_ = std::string(name);
  grid_id_ = GDattach(h_->gid(), name);
#ifdef DEBUG
  std::cerr << "Opening grid " << name << " for file " << h_->file_name()
	    << " gid = " << h_->gid()
	    << " grid_id = " << grid_id_ << "\n";
#endif
  if(grid_id_ ==-1)
    throw Exception("Trouble opening grid " + gname_ + " of file "
		    + h_->file_name());
  char field_list_buf[HDFE_DIMBUFSIZE];
  intn status = GDinqfields(grid_id_, field_list_buf, 0, 0);
  if(status ==-1)
    throw Exception("Trouble getting field list for grid " + gname_ +
		    " of file " + h_->file_name());
  field_list_ = std::string(field_list_buf);
  int32 buf_size;
  int32 nattr = GDinqattrs(grid_id_, 0, &buf_size);
  if(nattr == -1)
    throw Exception("Trouble reading attributes for file " + h_->file_name());
  std::vector<char> attribute_list_buf(buf_size + 1);
				// +1 is for trailing '\0'.
  nattr = GDinqattrs(grid_id_, &(*(attribute_list_buf.begin())),
		     &buf_size); 
  std::vector<char>::const_iterator e2 = attribute_list_buf.end() - 1;
  std::vector<char>::const_iterator a2 = attribute_list_buf.begin();
  std::vector<char>::const_iterator b2 = a2;
  while(b2 != e2) {
    b2 = std::find(a2, e2, ',');
				// Split out attrbute list.
    att_list_.push_back(std::string(a2, b2));
    a2 = b2 + 1;
    int32 data_type, nvalue;
    intn hdf_status = GDattrinfo(grid_id_,
				 const_cast<char*>(att_list_.back().c_str()),
				 &data_type, &nvalue);
    if(hdf_status)
      throw Exception("Trouble reading attributes for file " + h_->file_name());
    // In an odd quirky buggy sort of way, GDattrinfo returns the number of
    // bytes, not the number of items, even though its documentation
    // claims otherwise.
    switch(data_type) {
    case DFNT_FLOAT64:
      nvalue /= 8;
      break;
    case DFNT_FLOAT32:
    case DFNT_INT32:
    case DFNT_UINT32:
      nvalue /= 4;
      break;
    case DFNT_INT16:
    case DFNT_UINT16:
      nvalue /= 2;
      break;
    default:
      break;
    }
    att_data_type_[att_list_.back()] = data_type;
    att_data_size_[att_list_.back()] = nvalue;
  }
  h_->reg(this);
}

void HdfGrid::field_read(char* Field_name, void* Buf, int32* Start, 
			 int32* Stride, int32* Edge) const
{
  intn status = GDreadfield(grid_id_, Field_name, Start, Stride, Edge, Buf);
  if(status)
    throw Exception("Trouble reading field " + 
		    std::string(Field_name) +
		    "for grid " + gname_ +
		    " of file " + h_->file_name());
}

void HdfGrid::set_chunk_cache(char* Field_name, int Number_chunk)
{
  int32 sds_id, rank_sds, rank_fld, offset, solo, dims[MAX_VAR_DIMS];
  intn status = GDSDfldsrch(grid_id_, h_->sid(), Field_name, &sds_id,
			    &rank_sds, &rank_fld, &offset, dims, &solo);
  if(status)
    throw Exception("Trouble setting chunk size for field " + 
		    std::string(Field_name) +
		    "for grid " + gname_ +
		    " of file " + h_->file_name());
  status = SDsetchunkcache(sds_id, Number_chunk, 0);
  if(status ==-1)
    throw Exception("Trouble setting chunk size for field " + 
		    std::string(Field_name) +
		    "for grid " + gname_ +
		    " of file " + h_->file_name());
}

// This fills in a bit of information about a specific field, which
// can then be read. This is a bit sloppy, but because the users of
// this class are tightly controlled, we can get away with this.

void HdfGrid::populate_field_info(char* Field_name) {
  intn status = GDfieldinfo(grid_id_, Field_name, &rank_, dim_, &data_type_,
			    dim_name_buf_);
  if(status)
    throw Exception("Trouble getting field info for field " + 
		    std::string(Field_name) +
		    "for grid " + gname_ +
		    " of file " + h_->file_name());
  status = GDtileinfo(grid_id_, Field_name, &tile_code_, &tile_rank_, 
		      tile_dim_);
  if(status)
    throw Exception("Trouble reading field " + 
		    std::string(Field_name) +
		    "for grid " + gname_ +
		    " of file " + h_->file_name());
}
