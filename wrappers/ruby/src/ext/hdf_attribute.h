#ifndef HDF_ATTRIBUTE_H
#define HDF_ATTRIBUTE_H

//-----------------------------------------------------------------------
// For handling attributes, we want to treat a type T different from
// vector<T> and std::string, i.e. we want partial specialization.
// Because we do this in several places (HdfFile and HdfGrid), it is
// cleaner to factor out the differences using a Helper class, rather
// than writing a number of specializations in each class. This class
// is entirely an implementation detail, there is no reason why
// anybody other than the HDF class would want to use it.
//-----------------------------------------------------------------------

namespace Attribute {
  template<class T> class Helper {
  public:
    bool must_be_one() const {return true;}
    T create(int Size) const {return T();}
    void* pointer(const T& t) const 
    {return const_cast<void*>(static_cast<const void*>(&t));}
    int32 size(const T& t) const {return 1;}
    int size_base() const {return sizeof(T);}
  };
  
  template<class T> class Helper<std::vector<T> > {
  public:
    bool must_be_one() const {return false;}
    std::vector<T> create(int Size) const {return std::vector<T>(Size);}
    void* pointer(const std::vector<T>& t) const 
    {return const_cast<void*>(static_cast<const void*>(&(*(t.begin()))));}
    int32 size(const std::vector<T>& t) const {return t.size();}
    int size_base() const {return sizeof(T);}
  };
  
  template<> class Helper<std::string> {
  public:
    bool must_be_one() const {return false;}
    std::string create(int Size) const {return std::string(Size, 'A');}
    void* pointer(const std::string& t) const 
    {return const_cast<void*>(static_cast<const void*>(&(*(t.begin()))));}
    int32 size(const std::string& t) const {return t.size();}
    int size_base() const {return sizeof(char8);}
  };
} // End namespace

#endif /* #ifndef HDF_ATTRIBUTE_H */
