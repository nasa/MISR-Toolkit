This directory contains a binary version of MISR Toolkit version 1.4.4 for Windows.

This software was built on Microsoft Windows 7 32-bit using Microsoft Visual C++ 2010.  It has been tested with Python 2.7 and IDL 8.2.

The binaries depend on the following libraries which are included in this package for your convenience.
 
    ZLIB library (version 1.2.5) 
    JPEG library (version 6b) (.lib only)
    SZIP library (version 2.1)  *Encoder ENABLED*
    HDF library (version 4.2r6)
    HDF-EOS library (version 2.18)
    TRE library (version 0.7.5)

To use the Windows IDL binaries requires IDL 8.2 (32-bit).
Note: the 32-bit IDL dll's will NOT load in 64-bit IDL. Use the 32-bit IDL that was installed alongside the 64-bit version.

Not Included:
    NumPy 1.x (Required if you want to use Python) http://www.numpy.org/


Directory structure:

Mtk_win32
    dll        (Dynamic library)
    doc        (MISR Toolkit Documentation)
    example    (Example project – Select “Release” configuration then compile.) 
    idl        (Set your IDL_DLM_PATH environment variable to this folder.)
    include    (Header files)
    lib        (Static library)
    Mtk_depend (Add this folder to you PATH environment variable) 
    python     (Set your PYTHONPATH environment variable to this folder.)


How to set environment variables:

1. Right Click "My Computer", and then click "Properties".
2. Click the "Advanced" tab.
3. Click "Environment Variables".
