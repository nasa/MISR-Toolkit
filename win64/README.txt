This directory contains the third party dependencies and solution files to build MISR Toolkit for Windows 10 64-bit.

The solution was last built on Windows 10 64-bit using Microsoft Visual C++ 2015. 
Solution output files have been tested with Python 3.6, and IDL 8.7.

Note that Python 2.7 is no longer supported as it is end of life and will not compile modules with MSVC 2015.

Building on Windows 10 depends on the following libraries which are included here for your convenience.

    HDF4 library (version 4.2.14)
    HDF5 library (version 1.8.21)
    HDF-EOS2 library (version 2.19)
    JPEG library (version 8d) (in HDF4 directory)
    netCDF4 library (version 4.7.4)
    SZIP library (version 2.1.1)  (in HDF4 directory)
    TRE library (version 0.8.0)
    ZLIB library (version 1.2.8) (in HDF4 directory)

Not Included:
    NumPy (Required if you want to use Python) http://www.numpy.org/
