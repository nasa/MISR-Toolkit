;NSIS Installer Creation Script
;For Mtk

;--------------------------------
;Include Modern UI
  !include "MUI.nsh"

;--------------------------------
;Include EnvVarUpdate for editing environmental variables
  !include "EnvVarUpdate.nsh"

;--------------------------------
;General

  ;Version
  !define VERSION "1.0.3"

  ;Name and file
  Name "MISR Toolkit ${VERSION}"
  outFile "MISR_Toolkit_Setup-${VERSION}.exe"

  ;Default installation folder
  installDir "$PROGRAMFILES\MisrToolkit-${VERSION}"

  ;Request application privileges for Windows Vista
  RequestExecutionLevel user

;--------------------------------
;Interface Settings

  !define MUI_ABORTWARNING

;--------------------------------
;Pages

  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_LICENSE "mtk_license.rtf"
  !insertmacro MUI_PAGE_COMPONENTS
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
  
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
;Languages

  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
;Installer Sections

  Section "MISR Toolkit ${VERSION}"

    ;Create Uninstaller 
    setOutPath $INSTDIR
    WriteUninstaller $INSTDIR\uninstaller.exe

    ;Copy root level files
    file ..\README.txt
    
    ;Copy dll level files
    SetOutPath $INSTDIR\dll
    file ..\MisrToolkit\Release\MisrToolkit.dll
    
    ;Copy lib level files
    SetOutPath $INSTDIR\lib
    file ..\MisrToolkit\Release\MisrToolkit.lib
    
    ;Copy include level files
    SetOutPath $INSTDIR\include
    file ..\..\include\*.h
    file ..\..\CoordQuery\include\*.h
    file ..\..\CoordQuery\include\*.h
    file ..\..\FileQuery\include\*.h
    file ..\..\MapQuery\include\*.h
    file ..\..\misrcoord\*.h
    file ..\..\odl\*.h
    file ..\..\OrbitPath\include\*.h
    file ..\..\CoordQuery\include\*.h
    file ..\..\ReadData\include\*.h
    file ..\..\Regression\include\*.h
    file ..\..\CoordQuery\include\*.h
    file ..\..\ReProject\include\*.h
    file ..\..\SetRegion\include\*.h
    file ..\..\CoordQuery\include\*.h
    file ..\..\UnitConv\include\*.h
    file ..\..\Util\include\*.h
    file ..\..\WriteData\include\*.h
    
    ;Copy Mtk_depend level files
    SetOutPath $INSTDIR\Mtk_depend
    file ..\..\..\Mtk_thirdparty\win32\hdfeos\gctp\lib\windows\gctp.lib
    file ..\..\..\Mtk_thirdparty\win32\42r1-win\release\dll\hd421m.dll
    file ..\..\..\Mtk_thirdparty\win32\42r1-win\release\dll\hd421m.lib
    file ..\..\..\Mtk_thirdparty\win32\hdfeos\lib\windows\hdfeos.lib
    file ..\..\..\Mtk_thirdparty\win32\42r1-win\release\dll\hm421m.dll
    file ..\..\..\Mtk_thirdparty\win32\42r1-win\release\dll\hm421m.lib
    file ..\..\..\Mtk_thirdparty\win32\szip20-win-xp-intel81-enc\dll\szlibdll.dll
    file ..\..\..\Mtk_thirdparty\win32\tre-0.7.5\win32\Release\tre.dll
    file ..\..\..\Mtk_thirdparty\win32\zlib122-windows\zlib1.dll
    
    ;Set MTKHOME Environmental Variable for All Users
    ${EnvVarUpdate} $0 "MTKHOME" "P" "HKLM" "$INSTDIR"
    
  SectionEnd

  Section "Documentation"
    ;Copy doc level files
    SetOutPath $INSTDIR\doc
    file ..\..\doc\*
    SetOutPath $INSTDIR\doc\html
    file ..\..\doc\html\*
    SetOutPath $INSTDIR\doc\IDL_HTML_DOCS
    file ..\..\doc\IDL_HTML_DOCS\*
    SetOutPath $INSTDIR\doc\pymtk
    ;file ..\..\doc\pymtk\*  ;TODO: uncomment this when stuff is in this directory
  SectionEnd

  Section "IDL Wrapper"
    ;Copy idl level files
    SetOutPath $INSTDIR\idl
    file ..\MisrToolkit\Release\idl\idl_Mtk.dll
    file ..\..\wrappers\idl\idl_Mtk.dlm

    ;Set IDL_DLM_PATH Environmental Variable for All Users
    ${EnvVarUpdate} $0 "IDL_DLM_PATH" "P" "HKLM" "$INSTDIR\idl"
  SectionEnd

  Section "Python Wrapper"
    ;Copy python level files
    SetOutPath $INSTDIR\python
    file ..\MisrToolkit\Release\Python\MisrToolkit.dll

    ;Set PYTHONPATH Environmental Variable for All Users
    ${EnvVarUpdate} $0 "PYTHONPATH" "P" "HKLM" "$INSTDIR\python"
  SectionEnd
  
  Section "Examples"
    ;Copy example level files
    SetOutPath $INSTDIR\example
    file ..\example\example.sln
    SetOutPath $INSTDIR\example\example
    file ..\example\example\example.vcproj

  SectionEnd

;--------------------------------
;Uninstaller Section

  Section "Uninstall"
    ;Always delete uninstaller first
    delete $INSTDIR\uninstaller.exe

    ;Remove root level files
    delete $INSTDIR\README.txt

    ;Remove dll level files
    delete $INSTDIR\dll\MisrToolkit.dll
    RMDir $INSTDIR\dll

    ;Remove doc level files
    delete $INSTDIR\doc\*
    delete $INSTDIR\doc\html\*
    delete $INSTDIR\doc\IDL_HTML_DOCS\*
    delete $INSTDIR\doc\pymtk\*
    RMDir $INSTDIR\doc\html
    RMDir $INSTDIR\doc\IDL_HTML_DOCS
    RMDir $INSTDIR\doc\pymtk
    RMDir $INSTDIR\doc

    ;Remove example level files
    ;TODO: decide if removing example solution is wise
    delete $INSTDIR\example\example.sln 
    delete $INSTDIR\example\example\example.vcproj
    RMDIR $INSTDIR\example\example
    RMDIR $INSTDIR\example

    ;Remove idl level files
    delete $INSTDIR\idl\idl_Mtk.dll
    delete $INSTDIR\idl\idl_Mtk.dlm
    RMDir $INSTDIR\idl

    ;Remove include level files
    delete $INSTDIR\include\*.h
    RMDir $INSTDIR\include

    ;Remove lib level files
    delete $INSTDIR\lib\MisrToolkit.lib
    RMDir $INSTDIR\lib

    ;Remove Mtk_depend level files
    delete $INSTDIR\Mtk_depend\gctp.lib
    delete $INSTDIR\Mtk_depend\hd421m.dll
    delete $INSTDIR\Mtk_depend\hd421m.lib
    delete $INSTDIR\Mtk_depend\hdfeos.lib
    delete $INSTDIR\Mtk_depend\hm421m.dll
    delete $INSTDIR\Mtk_depend\hm421m.lib
    delete $INSTDIR\Mtk_depend\szlibdll.dll
    delete $INSTDIR\Mtk_depend\tre.dll
    delete $INSTDIR\Mtk_depend\zlib1.dll
    RMDir $INSTDIR\Mtk_depend

    ;Remove python level files
    delete $INSTDIR\python\MisrToolkit.dll
    RMDir $INSTDIR\python

    ;Remove Mtk directory
    RMDir $INSTDIR

    ;Unset MTKHOME Environmental Variable for All Users
    ${un.EnvVarUpdate} $0 "MTKHOME" "R" "HKLM" "$INSTDIR"

    ;Unset IDL_DLM_PATH Environmental Variable for All Users
    ${un.EnvVarUpdate} $0 "IDL_DLM_PATH" "R" "HKLM" "$INSTDIR\idl"

    ;Unset PYTHONPATH Environmental Variable for All Users
    ${un.EnvVarUpdate} $0 "PYTHONPATH" "R" "HKLM" "$INSTDIR\python"

  SectionEnd 
