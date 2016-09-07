@echo off
set BINDIR=Mtk-bin-win32
set WIN32DIR=%CD%
REM REM line below to enable output of copy commands
set APPENDNUL=^>NUL

REM Remove and Recreate

echo Cleaning old directory.
rmdir /S /Q %BINDIR% 2>NUL
echo Creating new binary folder: %BINDIR%.
mkdir %BINDIR%

REM Building Root

copy README.txt %BINDIR% %APPENDNUL%
cd %BINDIR%

REM Building doc

mkdir doc
copy %WIN32DIR%\..\doc\Mtk_ug.pdf doc %APPENDNUL%

REM Building dll

mkdir dll
copy %WIN32DIR%\MisrToolkit\Release\MisrToolkit.dll dll %APPENDNUL%

REM Building example

mkdir example
copy %WIN32DIR%\example\example.sln example %APPENDNUL%
mkdir example\example
copy %WIN32DIR%\example\example\example.vcproj example\example %APPENDNUL%
copy %WIN32DIR%\..\examples\C\bar.c example\example %APPENDNUL%
copy %WIN32DIR%\..\examples\C\biz.c example\example %APPENDNUL%
copy %WIN32DIR%\..\examples\C\foo.c example\example %APPENDNUL%

REM Building idl

mkdir idl
copy %WIN32DIR%\MisrToolkit\Release\IDL\idl_Mtk.dll idl %APPENDNUL%
copy %WIN32DIR%\..\wrappers\idl\idl_Mtk_dlm.template idl\idl_Mtk.dlm %APPENDNUL%

REM Building include

mkdir include
copy %WIN32DIR%\..\include\MisrError.h include %APPENDNUL%
copy %WIN32DIR%\..\include\MisrProjParam.h include %APPENDNUL%
copy %WIN32DIR%\..\include\MisrToolkit.h include %APPENDNUL%
copy %WIN32DIR%\..\CoordQuery\include\MisrCoordQuery.h include %APPENDNUL%
copy %WIN32DIR%\..\FileQuery\include\dirent_win32.h include %APPENDNUL%
copy %WIN32DIR%\..\FileQuery\include\MisrFileQuery.h include %APPENDNUL%
copy %WIN32DIR%\..\MapQuery\include\MisrMapQuery.h include %APPENDNUL%
copy %WIN32DIR%\..\misrcoord\errormacros.h include %APPENDNUL%
copy %WIN32DIR%\..\misrcoord\misrproj.h include %APPENDNUL%
copy %WIN32DIR%\..\odl\odlparse.h include %APPENDNUL%
copy %WIN32DIR%\..\odl\odlinter.h include %APPENDNUL%
copy %WIN32DIR%\..\odl\odldef_prototypes.h include %APPENDNUL%
copy %WIN32DIR%\..\odl\odldef.h include %APPENDNUL%
copy %WIN32DIR%\..\OrbitPath\include\MisrOrbitPath.h include %APPENDNUL%
copy %WIN32DIR%\..\ReadData\include\MisrCache.h include %APPENDNUL%
copy %WIN32DIR%\..\ReadData\include\MisrReadData.h include %APPENDNUL%
copy %WIN32DIR%\..\Regression\include\MisrRegression.h include %APPENDNUL%
copy %WIN32DIR%\..\ReProject\include\MisrReProject.h include %APPENDNUL%
copy %WIN32DIR%\..\SetRegion\include\MisrSetRegion.h include %APPENDNUL%
copy %WIN32DIR%\..\UnitConv\include\MisrUnitConv.h include %APPENDNUL%
copy %WIN32DIR%\..\Util\include\MisrUtil.h include %APPENDNUL%
copy %WIN32DIR%\..\WriteData\include\MisrWriteData.h include %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Includes\* include %APPENDNUL%

REM Building lib

mkdir lib
copy %WIN32DIR%\HDF-EOS5Binaries\hd426m.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdf_fcstubdll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdf_fortrandll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hm426m.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\mfhdf_fcstubdll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\mfhdf_fortrandll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\tre.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\libjpeg.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\szip.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\libszip.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\szlibdll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\xdr_for_dll.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\zlib.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\zlib1.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\gctp.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\gctpd.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdfeos.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdfeosd.lib lib %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\MisrToolkit.lib lib %APPENDNUL%

REM Building Mtk_depend

mkdir Mtk_depend
copy %WIN32DIR%\HDF-EOS5Binaries\hd426m.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdf_fcstubdll.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hdf_fortrandll.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\hm426m.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\mfhdf_fcstubdll.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\mfhdf_fortrandll.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\szip.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\szlibdll.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\zlib1.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\tre4.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\libiconv2.dll Mtk_depend %APPENDNUL%
copy %WIN32DIR%\HDF-EOS5Binaries\libintl3.dll Mtk_depend %APPENDNUL%

REM Building python

mkdir python
copy %WIN32DIR%\MisrToolkit\Release\Python\MisrToolkit.pyd python %APPENDNUL%

REM Finished

echo All done. Now zip %BINDIR% to create a distributable win32 binary zip.
pause