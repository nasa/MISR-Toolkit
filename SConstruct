import os
import shutil
import platform
import re
import datetime
import string

IDLDIR = '/Applications/rsi/idl'
DOXYDIR = '/Applications/Doxygen/Doxygen.app/Contents/Resources/'
MTKHOME = os.getcwd()

Help("""
   Targets:

      install    - build and install MisrToolkit in "/usr/local"
      install PATH=/your/prefix - build and install MisrToolkit in "/your/prefix"
      cmdutil install - build and install command-line utilities and MisrToolkit in "/usr/local"
      cmdutil install PATH=/your/prefix - build and install command-line utilities and MisrToolkit in "/your/prefix"
      tags       - build emacs tags
      idl        - build the idl shared library
      test       - builds and runs MisrTookit tests
      cmdutil    - build the command-line utilities
      misrcoord  - build only the misrcoord library
      doc        - build documentation
      -c emacs   - delete emacs tilde files
""")

if 'help' in COMMAND_LINE_TARGETS:
   print 'Type: "scons -h" for help.'
   Exit()

# ---------------------------------------------------------------------
# Read MISR Toolkit version number
# ---------------------------------------------------------------------

version_file = open(MTKHOME + '/include/MisrToolkit.h')
contents = version_file.read()
sidx = contents.index('MTK_VERSION "') + 13
eidx = contents.index('"\n', sidx)
version_num = contents[sidx:eidx]
version_file.close()

# ---------------------------------------------------------------------
# Doxygen documentation
# ---------------------------------------------------------------------

if 'doc' in COMMAND_LINE_TARGETS:
   doxyfile_template = open(MTKHOME + '/doc/Doxyfile.template', 'r')
   doxyfile = open(MTKHOME + '/doc/Doxyfile', 'w')
   doxytemp = re.sub(r'<VERSION>',version_num,doxyfile_template.read())
   doxyfile.write(doxytemp)
   doxyfile_template.close()
   doxyfile.close()

   doxygen_status = Command(Dir(MTKHOME + '/doc/html'), MTKHOME + '/doc/Doxyfile', DOXYDIR + 'doxygen $SOURCE')

   Clean(doxygen_status, [MTKHOME + '/doc/html', MTKHOME + '/doc/Doxyfile'])
   doc_alias = Alias('doc', doxygen_status)
   AlwaysBuild(doc_alias)

# ---------------------------------------------------------------------
# Generate emacs tags
# ---------------------------------------------------------------------

if 'tags' in COMMAND_LINE_TARGETS:
   # ----------
   # List of standard modules to generate tags for
   # ----------
   mod_list = ['WriteData',
               'ReadData',
               'SetRegion',
               'OrbitPath',
               'MapQuery',
               'CoordQuery',
               'UnitConv',
               'FileQuery',
               'Util']

   mod_list.extend(['misrcoord', 'src'])

   file_list = []
   for mod_dir in mod_list:
      for root , dirs , files in os.walk ( mod_dir ) :
         file_list = file_list + [ os.path.join ( root , f ) for f in files if re.compile ( ".[ch]$" ).search ( f ) ]

   files = string.join(map(str,file_list))

   tags_status = Command(MTKHOME + '/TAGS ', None, 'etags -o ' + '$TARGET' + ' ' + files)

   Alias('tags',tags_status)
   Clean('tags',  MTKHOME + '/TAGS')

# ---------------------------------------------------------------------
# Clean emacs "~" files
# ---------------------------------------------------------------------

if 'emacs' in COMMAND_LINE_TARGETS:
   file_list = []
   for root , dirs , files in os.walk ( '.' ) :
      file_list = file_list + [ os.path.join ( root , f ) for f in files if re.compile ( ".*~" ).search ( f ) ]
   Clean ('emacs',file_list)

# ---------------------------------------------------------------------
# Check for HDF and HDF-EOS environment variables.
# ---------------------------------------------------------------------

if os.getenv('HDFINC') == None:
   print 'HDFINC environment variable not set.'
   Exit(1)
elif os.getenv('HDFEOS_INC') == None:
   print 'HDFEOS_INC environment variable not set.'
   Exit(1)
elif os.getenv('HDFLIB') == None:
   print 'HDFLIB environment variable not set.'
   Exit(1)
elif os.getenv('HDFEOS_LIB') == None:
   print 'HDFEOS_LIB environment variable not set.'
   Exit(1)

# ---------------------------------------------------------------------
# Setup Environment for MisrToolkit
# ---------------------------------------------------------------------

include_path = ['#include',
                '#WriteData/include',
                '#ReadData/include',
                '#SetRegion/include',
                '#OrbitPath/include',
                '#MapQuery/include',
                '#CoordQuery/include',
                '#UnitConv/include',
                '#misrcoord',
                '#odl',
                '#FileQuery/include',
                '#Util/include',
                os.getenv('HDFINC'),
                os.getenv('HDFEOS_INC'),
                os.getenv('HDFEOS_INC') + '/../gctp/include']

library_path = [os.getenv('HDFEOS_LIB'),
                os.getenv('HDFLIB'),
                os.getenv('HDFEOS_LIB') + '/../../gctp/lib']

if platform.system() == 'Solairs':
    libs = ['hdfeos', 'Gctp', 'mfhdf', 'df', 'jpeg', 'nsl', 'z', 'm']
else:
    libs = ['hdfeos', 'Gctp', 'mfhdf', 'df', 'jpeg', 'z', 'm']

#ENV = os.environ, 
opts = Options()
opts.Add(PathOption('PATH','Install Path','/usr/local'))
env = Environment(options=opts, CPPPATH=include_path, LIBS=libs, LIBPATH=library_path,SHLINKFLAGS='$LINKFLAGS -dynamic')


conf = Configure(env)

if not conf.CheckCHeader('gctp_prototypes.h'):
   Exit(1)
elif not conf.CheckCHeader('hdf.h'):
   Exit(1)
#elif not conf.CheckCHeader('HdfEosDef.h'):
#   Exit(1)
elif not conf.CheckCHeader('getopt.h'):
   Exit(1)
#elif not conf.CheckCHeader('jpeglib.h'):
#   Exit(1)
#elif not conf.CheckLib('df'):
#   print 'Did not find libdf.a, exiting!'
#   Exit(1)
#elif not conf.CheckLib('jpeg'):
#   print 'Did not find libjpeg.a, exiting!'
#   Exit(1)
if not conf.CheckCHeader('zlib.h'):
   Exit(1)
#elif not conf.CheckLib('m'):
#   print 'Did not find libm.a, exiting!'
#   Exit(1)

env = conf.Finish()

Export('env')

if 'test' in COMMAND_LINE_TARGETS: 
   env['RUN_UNIT_TESTS'] = True
else:
   env['RUN_UNIT_TESTS'] = False

# ---------------------------------------------------------------------
# misrcoord
# ---------------------------------------------------------------------

if 'misrcoord' in COMMAND_LINE_TARGETS:
   SConscript('misrcoord/SConscript')

# ---------------------------------------------------------------------
# Build MisrToolkit Library
# ---------------------------------------------------------------------

component_list = ['WriteData',
                  'ReadData',
                  'MapQuery',
                  'SetRegion',
                  'OrbitPath',
                  'CoordQuery',
                  'UnitConv',
                  'misrcoord',
                  'FileQuery',
                  'Util',
                  'odl']

objs = []
for comp in component_list:
    o = SConscript('%s/SConscript' % comp)
    objs.append(o)

MisrToolkit_static = env.StaticLibrary('lib/MisrToolkit', objs)
MisrToolkit_shared = env.SharedLibrary('lib/MisrToolkit', objs, SHLIBSUFFIX='.so')

# ---------------------------------------------------------------------
# Command-line utilities
# ---------------------------------------------------------------------

cmdutil_bin = []
if 'cmdutil' in COMMAND_LINE_TARGETS:
   env_cmdutil = Environment(CPPPATH=include_path + ['#include'], \
                             LIBS=['MisrToolkit'] + libs, \
          LIBPATH=library_path + ['#lib'],SHLINKFLAGS='$LINKFLAGS -dynamic')
   env_cmdutil.BuildDir('obj', 'src', duplicate=0)

   cmdutil_file_list = Split("""
         MtkLatLonToPathList.c           MtkReadData.c
         MtkFileToBlockRange.c           MtkFileLGID.c
         MtkFileVersion.c                MtkFileToGridList.c
         MtkBlsToLatLon.c                MtkBlsToSomXY.c
         MtkLatLonToBls.c                MtkLatLonToSomXY.c
         MtkSomXYToBls.c                 MtkSomXYToLatLon.c
         MtkPathTimeRangeToOrbitList.c   MtkOrbitToPath.c
         MtkTimeRangeToOrbitList.c       MtkTimeToOrbitPath.c
         MtkDdToDegMinSec.c              MtkDdToDms.c
         MtkDdToRad.c                    MtkDegMinSecToDd.c
         MtkDegMinSecToDms.c             MtkDegMinSecToRad.c
         MtkDmsToDd.c                    MtkDmsToDegMinSec.c
         MtkDmsToRad.c                   MtkRadToDd.c
         MtkRadToDms.c                   MtkRadToDegMinSec.c
         MtkFileGridToFieldList.c        MtkPathToProjParam.c
         MtkFindFileList.c               MtkMakeFilename.c
         MtkFileAttrList.c               MtkFileAttrGet.c
         MtkGridAttrList.c               MtkGridAttrGet.c
         MtkFileToOrbit.c                MtkFileCoreMetaDataQuery.c
         MtkFileCoreMetaDataGet.c        MtkJulianToDateTime.c
         MtkDateTimeToJulian.c           MtkReadBlockRange.c
         MtkOrbitToTimeRange.c           MtkFileGridFieldToDimList.c
         MtkRegionToPathList.c           MtkRegionPathToBlockRange.c
         MtkPathBlockRangeToBlockCorners.c""")

   env_cmdutil.BuildDir('obj', 'src', duplicate=0)

   for file in cmdutil_file_list:
      cmdutil_obj = env_cmdutil.Object('obj/' + file)
      util = env_cmdutil.Program('bin/' + file[:-2], cmdutil_obj)
      cmdutil_bin.append(util)

   Alias('cmdutil','bin')

# ---------------------------------------------------------------------
# Install Library
# ---------------------------------------------------------------------

if 'install' in COMMAND_LINE_TARGETS:
   env.Install(env['PATH'] + '/lib',[MisrToolkit_static,MisrToolkit_shared])
   env.Install(env['PATH'] + '/bin',cmdutil_bin)

   include_files = Split("""include/MisrToolkit.h    include/MisrError.h
       include/MisrProjParam.h            WriteData/include/MisrWriteData.h
       ReadData/include/MisrReadData.h    SetRegion/include/MisrSetRegion.h
       OrbitPath/include/MisrOrbitPath.h  MapQuery/include/MisrMapQuery.h
       CoordQuery/include/MisrCoordQuery.h UnitConv/include/MisrUnitConv.h
       FileQuery/include/MisrFileQuery.h  Util/include/MisrUtil.h
       misrcoord/misrproj.h               misrcoord/errormacros.h""")

   env.Install(env['PATH'] + '/include',include_files)
   Alias('install',env['PATH'])

# ---------------------------------------------------------------------
# IDL Wrapper
# ---------------------------------------------------------------------

idl_include = include_path + [IDLDIR + '/external/include', '#include']
idl_libs = ['MisrToolkit'] + libs
idl_lib_path = library_path + ['#lib']

idl_env = Environment(CPPPATH=idl_include, LIBS=idl_libs, \
                      LIBPATH=idl_lib_path, \
                      SHLINKFLAGS='$LINKFLAGS -flat_namespace \
           -undefined suppress -bundle -dynamic -all_load')

if 'idl' in COMMAND_LINE_TARGETS:
   dt = datetime.date.today()
   idl_template = open(MTKHOME + '/wrappers/idl/idl_Mtk_dlm.template', 'r')
   idlfile = open(MTKHOME + '/lib/idl_Mtk.dlm', 'w')
   idltemp = re.sub(r'<VERSION>',version_num,idl_template.read())
   idltemp = re.sub(r'<DATE>',str(dt),idltemp)
   idlfile.write(idltemp)
   idl_template.close()
   idlfile.close()

   idl_mtk = idl_env.SharedLibrary('lib/idl_mtk', '#wrappers/idl/idl_mtk.c', \
                        SHLIBPREFIX='', SHLIBSUFFIX='.so')
   idl = idl_env.Alias('idl', [idl_mtk])
   Clean('idl','lib/idl_Mtk.dlm')

