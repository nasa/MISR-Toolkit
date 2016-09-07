#include "hdf_vdata.h"

void HdfVdata::close() 
{
  if(vdata_id_ != -1) {
#ifdef DEBUG
    std::cerr << "Closing vdata " << vname_ << " for file " << h_->file_name()
	      << " gid = " << h_->gid()
	      << " vdata_id_ = " << vdata_id_ << "\n";
#endif
    VSdetach(vdata_id_);
    vdata_id_ = -1;
    h_->dereg(this);
  }
}

void HdfVdata::open(VALUE Hdf_obj, char* name) 
{
  hdf_obj_ = Hdf_obj;
  Data_Get_Struct(hdf_obj_, HdfFile, h_);
  vname_ = std::string(name);
  int32 vdata_ref = VSfind(h_->hid(), 
			   const_cast<char*>(vname_.c_str()));
  if(vdata_ref == -1)
    throw Exception("Trouble opening vdata " + vname_ + " of file "
		    + h_->file_name());
  vdata_id_ = VSattach(h_->hid(), vdata_ref, "r");
  if(vdata_id_ == -1)
    throw Exception("Trouble opening vdata " + vname_ + " of file "
		    + h_->file_name());
#ifdef DEBUG
  std::cerr << "Opening vdata " << name << " for file " << h_->file_name()
	    << " gid = " << h_->gid()
	    << " vdata_id_ = " << vdata_id_ << "\n";
#endif
  h_->reg(this);
  char vname[VSNAMELENMAX];
  char flist[(FIELDNAMELENMAX / 8 + 1) * VSFIELDMAX + 1];
				// The fieldname list contains field
				// names, seperated by commas. For
				// some bizarre reason FIELDNAMELENMAX
				// is in bits, rather than bytes, so
				// we divide by 8 to get the maximum
				// name of a field. The "+1" is for
				// the ",", and we have VSFIELDMAX
				// possible fields in a vdata. Second
				// "+1" is for trailing '\0'.
  int32 interlace_mode, vsize;
  intn hdf_status = VSinquire(vdata_id_, &num_rec_, &interlace_mode, 
			      flist, &vsize, vname);
  if(hdf_status ==-1)
    throw Exception("Trouble opening vdata " + vname_ + " of file "
		    + h_->file_name());
  field_list_ = std::string(flist);
}

VALUE HdfVdata::read(char* field_name, int rec_i, int num_read) const
{
  int32 field_index;
  intn hdf_status = VSfindex(vdata_id_, 
			     field_name, 
			     &field_index);
  if(hdf_status ==-1)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (1)");
  int32 order = VFfieldorder(vdata_id_, field_index);
  if(order ==-1)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (2)");
  int32 data_type = VFfieldtype(vdata_id_, field_index);
  if(data_type ==-1)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (3)");
  hdf_status = VSsetfields(vdata_id_, field_name);
  if(hdf_status ==-1)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (4)");
  hdf_status = VSseek(vdata_id_, rec_i);
  if(hdf_status ==-1)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (5)");
  std::vector<uint8> buf(order * sizeof(float64) * num_read);
  uint8* buf_ptr = (&(*buf.begin()));
  int32 nread = VSread(vdata_id_, buf_ptr, num_read, FULL_INTERLACE);
  if(nread != num_read)
    throw Exception("Trouble reading field " + std::string(field_name)
		    + " of vdata " + vname_ + " of file "
		    + h_->file_name() + " (6)");
  VALUE res_val;
  if(num_read != 1)
    res_val = rb_ary_new();
  for(int i = 0; i < num_read; ++i) {
    std::vector<char> res_string;
    VALUE res_i;
    if(order != 1)
      res_i = rb_ary_new();
    for(int j = 0; j < order; ++j) {
      VALUE res_ij;
      switch(data_type) {
      case DFNT_CHAR:
	{
	  char* p = reinterpret_cast<char*>(buf_ptr);
	  res_ij = CHR2FIX(p[i * order + j]);
	  if(order != 1)
	    res_string.push_back(p[i * order + j]);
	}
	break;
      case DFNT_FLOAT32:
	{
	  float32* p = reinterpret_cast<float32*>(buf_ptr);
	  res_ij = rb_float_new(p[i * order + j]);
	}
	break;
      case DFNT_FLOAT64:
	{
	  float64* p = reinterpret_cast<float64*>(buf_ptr);
	  res_ij = rb_float_new(p[i * order + j]);
	}
	break;
      case DFNT_INT8:
	{
	  int8* p = reinterpret_cast<int8*>(buf_ptr);
	  res_ij = CHR2FIX(p[i * order + j]);
	}
	break;
      case DFNT_UINT8:
	{
	  uint8* p = reinterpret_cast<uint8*>(buf_ptr);
	  res_ij = CHR2FIX(p[i * order + j]);
	}
	break;
      case DFNT_INT16:
	{
	  int16* p = reinterpret_cast<int16*>(buf_ptr);
	  res_ij = INT2NUM(p[i * order + j]);
	}
	break;
      case DFNT_UINT16:
	{
	  uint16* p = reinterpret_cast<uint16*>(buf_ptr);
	  res_ij = INT2NUM(p[i * order + j]);
	}
	break;
      case DFNT_INT32:
	{
	  int32* p = reinterpret_cast<int32*>(buf_ptr);
	  res_ij = INT2NUM(p[i * order + j]);
	}
	break;
      case DFNT_UINT32:
	{
	  uint32* p = reinterpret_cast<uint32*>(buf_ptr);
	  res_ij = INT2NUM(p[i * order + j]);
	}
	break;
      default:
	throw Exception("Unrecognized data type while reading field " + 
			std::string(field_name)
			+ " of vdata " + vname_ + " of file "
			+ h_->file_name());
      }
      if(order ==1)
	res_i = res_ij;
      else
	rb_ary_push(res_i, res_ij);
    }
    if(data_type ==DFNT_CHAR &&
       order != 1)
      res_i = rb_str_new2(&res_string[0]);
    if(num_read ==1)
      res_val = res_i;
    else
      rb_ary_push(res_val, res_i);
  }
  return res_val;
}
