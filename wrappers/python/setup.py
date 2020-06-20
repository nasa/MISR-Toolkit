from setuptools import setup
from distutils.core import Extension
import os
import numpy
import platform

mtk_include_dirs = ['../../include', '../../Regression/include',
                '../../ReProject/include', '../../WriteData/include',
                '../../ReadData/include', '../../SetRegion/include',
                '../../OrbitPath/include', '../../MapQuery/include',
                '../../CoordQuery/include', '../../UnitConv/include',
                '../../FileQuery/include', '../../Util/include']

lib_dirs = ['../../lib']

hdfeosinc = os.getenv('HDFEOS_INC')
hdfeoslib = os.getenv('HDFEOS_LIB')
gctpinc = os.getenv('GCTPINC', default=os.getenv('HDFEOS_INC'))
gctplib = os.getenv('GCTPLIB', default=os.getenv('HDFEOS_LIB'))
ncinc = os.getenv('NCINC', default=os.getenv('HDFEOS_INC'))
nclib = os.getenv('NCLIB', default=os.getenv('HDFEOS_LIB'))
jpeginc = os.getenv('JPEGINC', default=os.getenv('HDFEOS_INC'))
jpeglib = os.getenv('JPEGLIB', default=os.getenv('HDFEOS_LIB'))
hdf5inc = os.getenv('HDF5INC', default=os.getenv('HDFEOS_INC'))
hdf5lib = os.getenv('HDF5LIB', default=os.getenv('HDFEOS_LIB'))
hdfinc = os.getenv('HDFINC')
hdflib = os.getenv('HDFLIB')


if platform.system() == "Windows":
    mtk_extra_object = ['../../win64/MisrToolkit/x64/Release/MisrToolkit_bundled.lib']
    mtk_libraries = []
    if not (gctpinc or gctplib or ncinc or nclib or jpeginc or jpeglib or
            hdf5inc or hdf5lib or hdfeosinc or hdfeoslib or hdfinc or hdflib):
        hdfinc = "../../win64/HDF_4.2.14/include"
        hdflib = "../../win64/HDF_4.2.14/lib"
        jpeginc = hdfinc
        jpeglib = hdflib
        hdfeosinc = "../../win64/hdfeos_2.19/include"
        hdfeoslib = "../../win64/hdfeos_2.19/lib"
        gctpinc = hdfeosinc
        gctplib = hdfeoslib
        ncinc = "../../win64/netcdf_4.7.4/include"
        nclib = "../../win64/netcdf_4.7.4/lib"
        hdf5inc = "../../win64/HDF5_1.8.21/include"
        hdf5lib = "../../win64/HDF5_1.8.21/lib"
else:
    mtk_libraries = ['netcdf', 'hdf5_hl', 'hdf5', 'hdfeos', 'Gctp', 'mfhdf', 'df', 'jpeg', 'z', 'm']
    mtk_extra_object = ['../../lib/libMisrToolkit.a', ]

module = Extension('MisrToolkit',
	include_dirs = [ numpy.get_include(), ncinc, hdf5inc, jpeginc, gctpinc,
                     hdfeosinc, hdfinc, jpeginc ] + mtk_include_dirs,
	library_dirs = [ '.', nclib, hdf5lib, hdfeoslib,
                     gctplib, hdflib, jpeglib ],
    libraries = mtk_libraries,
    extra_objects = mtk_extra_object,
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
       version = '1.5.1',
       author="MISR Project",
       license = 'BSD-3-Clause',
       long_description='Python interface to MISR Toolkit',
       long_description_content_type="text/markdown",
       url="https://github.com/nasa/Misr-Toolkit",)
