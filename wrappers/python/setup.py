from distutils.core import setup, Extension
import os
import numpy

include_dirs = ['../../include', '../../Regression/include',
                '../../ReProject/include', '../../WriteData/include',
                '../../ReadData/include', '../../SetRegion/include',
                '../../OrbitPath/include', '../../MapQuery/include',
                '../../CoordQuery/include', '../../UnitConv/include',
                '../../FileQuery/include', '../../Util/include']

lib_dirs = ['../../lib']

gctpinc = os.getenv('GCTPINC', default=os.getenv('HDFEOS_INC'))
gctplib = os.getenv('GCTPLIB', default=os.getenv('HDFEOS_LIB'))
ncinc = os.getenv('NCINC', default=os.getenv('HDFEOS_INC'))
nclib = os.getenv('NCLIB', default=os.getenv('HDFEOS_LIB'))
jpeginc = os.getenv('JPEGINC', default=os.getenv('HDFEOS_INC'))
jpeglib = os.getenv('JPEGLIB', default=os.getenv('HDFEOS_LIB'))
hdf5inc = os.getenv('HDF5INC', default=os.getenv('HDFEOS_INC'))
hdf5lib = os.getenv('HDF5LIB', default=os.getenv('HDFEOS_LIB'))

module = Extension('MisrToolkit',
	include_dirs = [ numpy.get_include(), ncinc, hdf5inc, jpeginc, gctpinc,
                     os.getenv('HDFEOS_INC'), os.getenv('HDFINC'), jpeginc ] + include_dirs,
	library_dirs = [ '.', nclib, hdf5lib, os.getenv('HDFEOS_LIB'),
                     gctplib, os.getenv('HDFLIB'), jpeglib ],
    libraries = ['netcdf', 'hdf5_hl', 'hdf5', 'hdfeos', 'Gctp', 'mfhdf', 'df', 'jpeg', 'z', 'm'],
    extra_objects = ['../../lib/libMisrToolkit.a'],
    sources = ['pyMtkBlockCorners.c', 'pyMtkDataPlane.c', 'pyMtkField.c', 'pyMtkFile.c',
               'pyMtkFileId.c', 'pyMtkGeoBlock.c', 'pyMtkGeoCoord.c', 'pyMtkGeoRegion.c',
               'pyMtkGrid.c', 'pyMtkMapInfo.c', 'pyMtkProjParam.c', 'pyMtkReProject.c',
               'pyMtkRegCoeff.c', 'pyMtkRegion.c', 'pyMtkRegression.c', 'pyMtkSomCoord.c',
               'pyMtkSomRegion.c', 'pyMtkTimeMetaData.c', 'pycoordquery.c', 'pyfilequery.c',
               'pyhelpers.c', 'pymisrtoolkit.c', 'pyorbitpath.c', 'pyunitconv.c', 'pyutil.c'],
    )

setup (name = 'MisrToolkit',
       description = 'Python interface to MISR Toolkit',
       packages = ['MisrToolkit'],
       ext_package = 'MisrToolkit',
       ext_modules = [module],
       version = '1.5.0')
