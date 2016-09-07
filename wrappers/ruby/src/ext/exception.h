#ifndef EXCEPTION_H
#define EXCEPTION_H
#include <string>		// Definition of string.
#include <stdio.h>

class Exception : public std::exception {
public:
  Exception(std::string What) : what_(What) 
  {
    add_hdf_error();
  }
  virtual ~Exception() throw() {}
  virtual const char* what() const throw() { return what_.c_str(); }
  void add_hdf_error();
private:
  std::string what_;
};

#endif
