from distutils.core import setup, Extension
import os
import glob
import numpy

include_dirs = ['../../include', '../../Regression/include',
                '../../ReProject/include', '../../WriteData/include',
                '../../ReadData/include', '../../SetRegion/include',
                '../../OrbitPath/include', '../../MapQuery/include',
                '../../CoordQuery/include', '../../UnitConv/include',
                '../../FileQuery/include', '../../Util/include']

lib_dirs = ['../../lib']


include_files = []
for idir in include_dirs:
  include_files += glob.glob(idir+'/*.h')

lib_files = []
for idir in lib_dirs:
  lib_files += glob.glob(idir+'/lib*.so')
  lib_files += glob.glob(idir+'/lib*.a')

for ifile in include_files:
  link_path = 'MisrToolkit/include/'+os.path.basename(ifile)
  if os.path.exists(link_path):
    os.unlink(link_path)
  os.symlink('../../'+ifile, link_path)

for ifile in lib_files:
  link_path = 'MisrToolkit/lib/'+os.path.basename(ifile)
  if os.path.exists(link_path):
    os.unlink(link_path)
  os.symlink('../../'+ifile, link_path)

module = Extension('MisrToolkit',
	include_dirs = [numpy.get_include(),
                        os.getenv('HDFEOS_INC'), os.getenv('HDFINC')] + include_dirs,
	library_dirs = ['.',os.getenv('HDFEOS_LIB'), os.getenv('HDFLIB')],
        libraries = ['hdfeos', 'mfhdf', 'df', 'z', 'jpeg', 'm',
                     'Gctp'],
        extra_objects = ['../../lib/libMisrToolkit.a'],
        sources = ['pymisrtoolkit.c', 'pyhelpers.c', 'pyMtkFile.c', 'pyMtkFileId.c', 'pyMtkGrid.c',
                   'pyMtkField.c', 'pyMtkRegion.c', 'pyMtkDataPlane.c',
                   'pyMtkProjParam.c', 'pyMtkGeoCoord.c', 'pyMtkGeoBlock.c',
                   'pyMtkBlockCorners.c', 'pyMtkMapInfo.c', 'pyMtkSomCoord.c', 'pyMtkReProject.c',
                   'pyMtkSomRegion.c', 'pyMtkGeoRegion.c', 'pyMtkTimeMetaData.c', 'pycoordquery.c',
                   'pyfilequery.c', 'pyorbitpath.c', 'pyunitconv.c', 'pyutil.c', 'pyMtkRegression.c',
                   'pyMtkRegCoeff.c'])

setup (name = 'MisrToolkit',
       description = 'Python interface to MISR Toolkit',
       packages = ['MisrToolkit'],
       package_dir = {'MisrToolkit': 'MisrToolkit' },
       package_data = {'MisrToolkit': ['include/*.h', 'lib/*']},
       ext_package = 'MisrToolkit',
       ext_modules = [module])
