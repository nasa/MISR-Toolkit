#=============================================================================
#=                                                                           =
#=                                Makefile                                   =
#=                                                                           =
#=============================================================================
#
#                         Jet Propulsion Laboratory
#                                   MISR
#                               MISR Toolkit
#
#            Copyright 2005, California Institute of Technology.
#                           ALL RIGHTS RESERVED.
#                 U.S. Government Sponsorship acknowledged.
#
#=============================================================================

#------------------------------------------------------------------------------
# Make variables
#------------------------------------------------------------------------------

MTKHOME := 	$(shell pwd)
VERSION := 	$(shell grep MTK_VERSION $(MTKHOME)/include/MisrToolkit.h | cut -d'"' -f2)
BUILDDATE :=	$(shell date "+%b %d %Y")
ARCH :=		$(shell uname -s -m)
INSTALLDIR :=	$(shell printenv MTK_INSTALLDIR)
COMMON_MK := $(MTKHOME)/common.mk

ifeq ("$(ARCH)","Darwin x86_64")
  IDL_DIR ?=    $(shell dirname $$(dirname $$(find /Applications/harris /Applications/exelis /Applications/rsi -type f -name idl -print -quit)))
  DOXYDIR ?=	/Applications/Doxygen/Doxygen.app/Contents/Resources
  PYDOCDIR?=    /usr/local/bin
  ARCH_CFLAGS:=	-arch x86_64
  IDL_MODE:=
  IDL_CFLAGS := -Wno-macro-redefined
  IDL_LDFLAGS:=
  MATHLIB :=    -lmx
  DYNLDFLAGS := -dynamiclib -all_load -current_version $(VERSION) $(ARCH_CFLAGS)
  MTK_LD_PATH := DYLD_LIBRARY_PATH
else
ifeq ("$(ARCH)","Linux x86_64")
  IDL_DIR ?= 	/usr/local/harris/idl
  DOXYDIR ?=	/usr/bin
  PYDOCDIR?=	/usr/local/bin
  ARCH_CFLAGS:=	-m64 -fno-cse-follow-jumps -fno-gcse -D_XOPEN_SOURCE=500
  IDL_MODE:=
  IDL_CFLAGS:=  -fno-strict-aliasing
  IDL_LDFLAGS:= $(IDL_DIR)/bin/bin.linux.x86_64/idl_hdf.so
  MATHLIB :=    -lm
  DYNLDFLAGS := -shared -Bsymbolic
  MTK_LD_PATH := LD_LIBRARY_PATH
else
  $(error $(ARCH) not supported. Try to edit Makefile to add your architecture)
endif
endif

ifeq ($(DEBUG),t)
  OPTFLAG :=	-g -O0 -ggdb
else
ifeq ($(OPTLVL),0)
  OPTFLAG :=	-O0
else
ifeq ($(OPTLVL),1)
  OPTFLAG :=	-O1
else
ifeq ($(OPTLVL),2)
  OPTFLAG :=	-O2
else
ifeq ($(OPTLVL),3)
  OPTFLAG :=	-O3
else
ifeq ($(OPTLVL),s)
  OPTFLAG :=	-Os
else
  OPTFLAG :=	-O2
endif
endif
endif
endif
endif
endif

#------------------------------------------------------------------------------
# Nothing to change below this line
#------------------------------------------------------------------------------

ifndef INSTALLDIR
  INSTALLDIR := $(MTKHOME)/../Mtk-$(VERSION)
endif

MODULES :=  	Util FileQuery UnitConv CoordQuery MapQuery OrbitPath
MODULES += 	SetRegion ReadData WriteData ReProject Regression

CC := 		/usr/bin/gcc
AR := 		/usr/bin/ar
RM :=		/bin/rm
CP :=		/bin/cp
LN :=		/bin/ln
FIND := 	/usr/bin/find
INSTALL :=	/usr/bin/install
RANLIB :=	/usr/bin/ranlib
SED :=		sed
ETAGS :=	etags
TAR :=		tar

ifndef GCTPINC
  GCTPINC := $(HDFEOS_INC)
endif
ifndef GCTPLIB
  GCTPLIB := $(HDFEOS_LIB)
endif

ifndef NCINC
  NCINC := $(HDFEOS_INC)
endif
ifndef NCLIB
  NCLIB := $(HDFEOS_LIB)
endif

ifndef HDF5INC
  HDF5INC := $(HDFEOS_INC)
endif
ifndef HDF5LIB
  HDF5LIB := $(HDFEOS_LIB)
endif

ifndef JPEGINC
  JPEGINC := $(HDFEOS_INC)
endif
ifndef JPEGLIB
  JPEGLIB := $(HDFEOS_LIB)
endif

CFLAGS := 	$(OPTFLAG) $(ARCH_CFLAGS) $(ADDITIONAL_CFLAGS)
CFLAGS +=	-Wall -Werror -pedantic -fPIC -std=c99 -fno-common
CFLAGS += 	-I$(MTKHOME)/include
CFLAGS += 	$(patsubst %, -I$(MTKHOME)/%/include, $(MODULES))
CFLAGS +=	-I$(MTKHOME)/misrcoord -I$(MTKHOME)/odl
CFLAGS += 	-I$(NCINC) -I$(HDF5INC) -I$(HDFEOS_INC) -I$(GCTPINC) -I$(HDFINC) -I$(JPEGINC) 

LDFLAGS := 	-L$(NCLIB) -L$(HDF5LIB) -L$(HDFEOS_LIB) -L$(GCTPLIB) -L$(HDFLIB) -L$(JPEGLIB) 
LDFLAGS +=      -lnetcdf -lhdf5_hl -lhdf5 -lhdfeos -lGctp -lmfhdf -ldf -ljpeg -lz $(MATHLIB) -ldl

INC :=		include/MisrToolkit.h
INC +=		include/MisrProjParam.h
INC +=		include/MisrError.h

SRC :=

include $(patsubst %,%/module.mk,$(MODULES))

OBJ := 		$(patsubst %.c,%.o,$(filter %.c, $(SRC)))

DEPEND :=	$(OBJ:.o=.d)

LIB = 		libMisrToolkit

CMDSRC :=	src/MtkFileLGID.c \
		src/MtkFileVersion.c \
		src/MtkFileAttrList.c \
		src/MtkFileAttrGet.c \
		src/MtkGridAttrList.c \
		src/MtkGridAttrGet.c \
		src/MtkFieldAttrList.c \
		src/MtkFieldAttrGet.c \
		src/MtkFileToPath.c \
		src/MtkFileToOrbit.c \
		src/MtkFileToBlockRange.c \
		src/MtkFileToGridList.c \
		src/MtkFileGridToFieldList.c \
		src/MtkFileGridToNativeFieldList.c \
		src/MtkFileCoreMetaDataQuery.c \
		src/MtkFileCoreMetaDataGet.c \
		src/MtkMakeFilename.c \
		src/MtkFindFileList.c \
		src/MtkDdToDegMinSec.c \
		src/MtkDdToDms.c \
		src/MtkDdToRad.c \
		src/MtkDegMinSecToDd.c \
		src/MtkDegMinSecToDms.c \
		src/MtkDegMinSecToRad.c \
		src/MtkDmsToDd.c \
		src/MtkDmsToDegMinSec.c \
		src/MtkDmsToRad.c \
		src/MtkRadToDd.c \
		src/MtkRadToDegMinSec.c \
		src/MtkRadToDms.c \
		src/MtkBlsToLatLon.c \
		src/MtkBlsToSomXY.c \
		src/MtkLatLonToBls.c \
		src/MtkLatLonToSomXY.c \
		src/MtkSomXYToBls.c \
		src/MtkSomXYToLatLon.c \
		src/MtkPathToProjParam.c \
		src/MtkLatLonToPathList.c \
		src/MtkOrbitToPath.c \
		src/MtkTimeToOrbitPath.c \
		src/MtkTimeRangeToOrbitList.c \
		src/MtkPathTimeRangeToOrbitList.c \
		src/MtkReadData.c \
		src/MtkJulianToDateTime.c \
		src/MtkDateTimeToJulian.c \
		src/MtkReadBlockRange.c \
		src/MtkOrbitToTimeRange.c \
		src/MtkFileGridFieldToDimList.c \
		src/MtkRegionToPathList.c \
		src/MtkRegionPathToBlockRange.c \
		src/MtkPathBlockRangeToBlockCorners.c \
		src/MtkMisrToEnvi.c \
		src/MtkVersion.c \
		src/MtkFileGridToResolution.c \
		src/MtkCreateLatLon.c \
		src/MtkFileType.c \
		src/MtkFillValueGet.c \
		src/MtkPixelTime.c

include misrcoord/module.mk
include odl/module.mk

STCLIB :=	$(LIB:%=lib/%.a)
DYNLIB :=	$(LIB:%=lib/%.so)
VERLIB :=	$(LIB:%=lib/%.so.$(VERSION))
IDLLIB :=	$(LIB:lib%=lib/idl_%.so)
IDLDLM :=	$(LIB:lib%=lib/idl_%.dlm)
PYTHONLIB :=    $(wildcard $(MTKHOME)/wrappers/python/build/lib.*/MisrToolkit/MisrToolkit*.so)
CENV :=		$(MTKHOME)/c_environ
IDLENV :=	$(MTKHOME)/idl_environ
PYTHONENV :=	$(MTKHOME)/python_environ
TESTSRC :=	$(OBJ:%.o=%_test.c)
TESTOBJ :=	$(OBJ:%.o=%_test.o)
TESTEXE :=	$(OBJ:%.o=%_test)
CMDOBJ :=	$(CMDSRC:%.c=%.o)
BINEXE :=	$(CMDOBJ:src/%.o=bin/%)

#------------------------------------------------------------------------------
# Targets
#------------------------------------------------------------------------------

all: lib cmdutil applications

#------------------------------------------------------------------------------
# Special target to stop .o deletion (with debug symbols)  during debugging
#------------------------------------------------------------------------------
ifeq ($(DEBUG),t)
.PRECIOUS: $(TESTOBJ)
endif
#------------------------------------------------------------------------------
# Library targets
#------------------------------------------------------------------------------

lib: $(STCLIB) $(DYNLIB) $(CENV)

$(STCLIB): $(OBJ) $(INC) $(MISRCOORDLIB) $(ODLLIB)
	@echo "Creating archive library $(STCLIB)..."
	$(AR) -rsu $(STCLIB) $(OBJ) $(MISRCOORDOBJ) $(ODLOBJ)

$(DYNLIB): $(OBJ) $(INC) $(MISRCOORDLIB) $(ODLLIB)
	@echo "Creating shared library $(VERLIB)..."
	$(CC) $(DYNLDFLAGS) -o $(VERLIB) $(OBJ) $(MISRCOORDOBJ) $(ODLOBJ) $(LDFLAGS)
	$(RM) -f $(DYNLIB)
	$(LN) -s $(VERLIB:lib/%=%) $(DYNLIB)

$(CENV):
	@$(RM) -f $@
	@echo 'setenv MTKHOME $(MTKHOME)' > $@
	@echo 'setenv MTK_CFLAGS "$(CFLAGS)"' >> $@
	@echo 'setenv MTK_LDFLAGS "-L$${MTKHOME}/lib -lMisrToolkit $(LDFLAGS)"' >> $@
	@echo 'if ($$?$(MTK_LD_PATH)) then' >> $@
	@echo '	setenv $(MTK_LD_PATH) "$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' >> $@
	@echo 'else' >> $@
	@echo '	setenv $(MTK_LD_PATH) "$${MTKHOME}/lib"' >> $@
	@echo 'endif' >> $@
	@$(RM) -f ${@}.sh
	@echo 'export MTKHOME=$(MTKHOME)' > ${@}.sh
	@echo 'export MTK_CFLAGS="$(CFLAGS)"' >> ${@}.sh
	@echo 'export MTK_LDFLAGS="-L$${MTKHOME}/lib -lMisrToolkit $(LDFLAGS)"' >> ${@}.sh
	@echo 'if [ $${#$(MTK_LD_PATH)} -ne 0 ]; then' >> ${@}.sh
	@echo '	export $(MTK_LD_PATH)="$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' >> ${@}.sh
	@echo 'else' >> ${@}.sh
	@echo '	export $(MTK_LD_PATH)="$${MTKHOME}/lib"' >> ${@}.sh
	@echo 'fi' >> ${@}.sh

#------------------------------------------------------------------------------
# IDL wrapper library targets
#------------------------------------------------------------------------------

IDL_CFLAGS+= $(CFLAGS)
IDL_LDFLAGS+= $(LDFLAGS)

idl: $(IDLLIB) $(IDLDLM) $(IDLENV)

$(IDLLIB): $(STCLIB) $(DYNLIB) $(MTKHOME)/wrappers/idl/idl_mtk.c
	@if [ -d "$(IDL_DIR)" ]; then \
		echo "Creating idl shared library $(IDLLIB)..." ;\
		CMD="make_dll,'idl_mtk','idl_MisrToolkit','IDL_Load',output_directory='lib', compile_directory='wrappers/idl',input_directory='wrappers/idl',/show_all_output,extra_cflags='$(IDL_CFLAGS)',extra_lflags='$(MTKHOME)/$(STCLIB)  $(IDL_LDFLAGS)',/platform_extension" ;\
		echo $$CMD | $(IDL_DIR)/bin/idl $(IDL_MODE);\
	else \
		echo "IDL not installed in" $(IDL_DIR) ;\
		echo "Please set the IDL_DIR environment variable and remake.";\
	fi

$(IDLDLM): $(MTKHOME)/wrappers/idl/idl_Mtk_dlm.template
	@$(SED) -e "s/<VERSION>/$(VERSION)/" -e "s/<DATE>/$(BUILDDATE)/" \
		$(MTKHOME)/wrappers/idl/idl_Mtk_dlm.template > $@

$(IDLENV):
	@$(RM) -f $@
	@echo 'setenv MTKHOME $(MTKHOME)' > $@
	@echo 'setenv IDL_PATH "<IDL_DEFAULT>:$${MTKHOME}/wrappers/idl:$${MTKHOME}/examples/idl"' >> $@
	@echo 'setenv IDL_DLM_PATH "<IDL_DEFAULT>:$${MTKHOME}/lib"' >> $@
	@$(RM) -f ${@}.sh
	@echo 'export MTKHOME=$(MTKHOME)' > ${@}.sh
	@echo 'export IDL_PATH="<IDL_DEFAULT>:$${MTKHOME}/wrappers/idl:$${MTKHOME}/examples/idl"' >> ${@}.sh
	@echo 'export IDL_DLM_PATH="<IDL_DEFAULT>:$${MTKHOME}/lib"' >> ${@}.sh

testidl: $(IDLLIB) $(IDLDLM) $(IDLENV)
	export MTKHOME=${MTKHOME}; \
	export IDL_DLM_PATH="<IDL_DEFAULT>:${MTKHOME}/lib"; \
	cd wrappers/idl; $(IDL_DIR)/bin/idl $(IDL_MODE) idl_testall.pro

#------------------------------------------------------------------------------
# Python wrapper library targets
#------------------------------------------------------------------------------

python: $(PYTHONDIR) $(PYTHONENV)

$(PYTHONENV): $(STCLIB) $(DYNLIB) 
	cd wrappers/python; python setup.py build
	@echo "---------------------------------------------------------------"
	@echo "To install MisrToolkit into Python's site-packages..."
	@echo "cd wrappers/python; sudo python setup.py install"
	@echo "---------------------------------------------------------------"
	@$(RM) -f $@
	@echo 'setenv MTKHOME $(MTKHOME)' > $@
	@echo 'setenv PYTHONPATH `echo $${MTKHOME}/wrappers/python/build/lib.*`' >> $@
	@echo 'if ($$?$(MTK_LD_PATH)) then' >> $@
	@echo '	setenv $(MTK_LD_PATH) "$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' >> $@
	@echo 'else' >> $@
	@echo '	setenv $(MTK_LD_PATH) "$${MTKHOME}/lib"' >> $@
	@echo 'endif' >> $@
	@$(RM) -f ${@}.sh
	@echo 'export MTKHOME=$(MTKHOME)' > ${@}.sh
	@echo 'export PYTHONPATH=`echo $${MTKHOME}/wrappers/python/build/lib.*`' >> ${@}.sh
	@echo 'if [ $${#$(MTK_LD_PATH)} -ne 0 ]; then' >> ${@}.sh
	@echo '	export $(MTK_LD_PATH)="$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' >> ${@}.sh
	@echo 'else' >> ${@}.sh
	@echo '	export $(MTK_LD_PATH)="$${MTKHOME}/lib"' >> ${@}.sh
	@echo 'fi' >> ${@}.sh

testpython: $(PYTHONENV)
	export PYTHONPATH=$(shell echo $(MTKHOME)/wrappers/python/build/lib.*); \
	export $(MTK_LD_PATH)=$(MTKHOME)/lib:$(LD_LIBRARY_PATH); \
	cd wrappers/python/test; python misrtoolkit_test.py

#------------------------------------------------------------------------------
# Test targets
#------------------------------------------------------------------------------

testall: test testpython testidl

unit_test: $(STCLIB) $(DYNLIB) $(TESTEXE) $(CENV)

test: $(STCLIB) $(DYNLIB) $(TESTEXE) $(CENV)
	@echo
	@echo "Running Tests..."
	@for i in $(TESTEXE); do \
		export $(MTK_LD_PATH)=$(MTKHOME)/lib; $$i 2>/dev/null ; \
	done

timetest: $(STCLIB) $(DYNLIB) $(TESTEXE) $(CENV)
	@echo
	@echo "Running Time Tests..."
	@for i in $(TESTEXE); do \
		export $(MTK_LD_PATH)=$(MTKHOME)/lib; /usr/bin/time $$i; echo;\
	done

valgrind: $(STCLIB) $(DYNLIB) $(TESTEXE) $(CENV)
	@echo
	@echo "Running Valgrind Tests..."
	@for i in $(TESTEXE); do \
		export $(MTK_LD_PATH)=$(MTKHOME)/lib; valgrind $$i; echo; \
	done

%_test: %_test.o $(STCLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $(OPTFLAGS) -o $@ $< $(MTKHOME)/lib/libMisrToolkit.a $(LDFLAGS)

#------------------------------------------------------------------------------
# Applications build rules
#------------------------------------------------------------------------------

applications: lib
	cd applications; make all MTK_CFLAGS="$(CFLAGS)" MTK_LDFLAGS="$(MTKHOME)/lib/libMisrToolkit.a $(LDFLAGS)"

#------------------------------------------------------------------------------
# Command-line utility build rules
#------------------------------------------------------------------------------

cmdutil: $(STCLIB) $(DYNLIB) $(CMDOBJ) $(BINEXE)

bin/%: src/%.o $(STCLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $(OPTFLAGS) -I$(MTKHOME)/include -o $@ $< $(MTKHOME)/lib/libMisrToolkit.a $(LDFLAGS)

#------------------------------------------------------------------------------
# Misrcoord build rules
#------------------------------------------------------------------------------

misrcoord: $(MISRCOORDLIB)

$(MISRCOORDLIB):
	cd misrcoord; make OPTFLAG="$(OPTFLAG)" ARCH_CFLAGS="$(ARCH_CFLAGS)" ADDITIONAL_CFLAGS="$(ADDITIONAL_CFLAGS)" lib

#------------------------------------------------------------------------------
# ODL build rules
#------------------------------------------------------------------------------

odl: $(ODLLIB)

$(ODLLIB):
	cd odl; make OPTFLAG="$(OPTFLAG)" ARCH_CFLAGS="$(ARCH_CFLAGS)" ADDITIONAL_CFLAGS="$(ADDITIONAL_CFLAGS) -O0" all

#------------------------------------------------------------------------------
# Clean targets
#------------------------------------------------------------------------------

clean: cleanlib cleanmisrcoord cleanodl cleanemacs cleanidl cleanpython cleantags cleandepend cleanapplications

cleanlib: cleanmisrcoord cleanodl
	$(RM) -f $(OBJ) $(STCLIB) $(DYNLIB) $(VERLIB) $(LIBOBJ) $(IDLLIB) \
			$(IDLDLM) $(TESTOBJ) $(TESTEXE) $(CMDOBJ) $(BINEXE) \
			$(MTKHOME)/wrappers/idl/idl_mtk.o $(CENV) $(CENV).sh

cleanmisrcoord:
	cd misrcoord; make clean

cleanodl:
	cd odl; make clean

cleanemacs:
	$(FIND) . -name "*~" -exec /bin/rm -f {} \;

cleanidl:
	$(RM) -f $(IDLENV) $(IDLENV).sh $(IDLDLM) $(IDLLIB)

cleanpython:
	$(RM) -rf $(MTKHOME)/wrappers/python/build $(MTKHOME)/wrappers/python/*.pyc $(PYTHONENV) $(PYTHONENV).sh

cleandoc:
	$(RM) -f $(MTKHOME)/doc/html/* $(MTKHOME)/doc/Doxyfile $(MTKHOME)/doc/pymtk/* $(MTKHOME)/doc/pymtk.pdf

cleantags:
	$(RM) -f $(MTKHOME)/TAGS

cleandepend:
	$(RM) -f $(DEPEND)

cleanapplications:
	cd applications; make clean

#------------------------------------------------------------------------------
# Tag rules
#------------------------------------------------------------------------------

tags: $(INC) $(SRC) $(TESTSRC) $(CMDSRC)
	$(ETAGS) -o $(MTKHOME)/TAGS $^

#------------------------------------------------------------------------------
# Distribution rules
#------------------------------------------------------------------------------

dist: distsrc

distsrc: clean doc
	mkdir /tmp/Mtk-src-$(VERSION); \
       $(TAR) --exclude "*/\.*" --exclude "./doc/Mtk_design.pdf" \
                --exclude "*/wrappers/ruby" --exclude "*/wrappers/opendap" \
                -cf - . | \
	$(TAR) xpf - -C /tmp/Mtk-src-$(VERSION) ; \
	cd /tmp; \
	$(TAR) czvf $(MTKHOME)/../Mtk-src-$(VERSION).tar.gz ./Mtk-src-$(VERSION)
	$(RM) -rf /tmp/Mtk-src-$(VERSION)

distdoc: clean doc
	$(TAR) --exclude "*/\.*" --exclude "./doc/Mtk_design.pdf" \
		-czvf $(MTKHOME)/../Mtk-doc-$(VERSION).tar.gz ./doc

distdata:
	$(RM) -f $(MTKHOME)/../Mtk_testdata/out/*; \
	cd ..; \
	tar --exclude "*/\.*" --exclude "*/*~" \
		-czvf Mtk-testdata-$(VERSION).tar.gz ./Mtk_testdata \

#------------------------------------------------------------------------------
# Install rules
#------------------------------------------------------------------------------

install: lib cmdutil
# Create directories
	$(INSTALL) -m 755 -d $(INSTALLDIR)/include $(INSTALLDIR)/lib \
		$(INSTALLDIR)/lib/idl $(INSTALLDIR)/lib/python \
		$(INSTALLDIR)/bin \
		$(INSTALLDIR)/doc $(INSTALLDIR)/doc/html \
		$(INSTALLDIR)/doc/IDL_HTML_DOCS $(INSTALLDIR)/doc/pymtk \
		$(INSTALLDIR)/examples $(INSTALLDIR)/examples/C \
		$(INSTALLDIR)/examples/idl $(INSTALLDIR)/examples/python
# Install include files
	$(INSTALL) -m 644 $(INC) $(INSTALLDIR)/include
# Installed C static and dynamic libraries
	$(INSTALL) -m 644 $(STCLIB) $(VERLIB) $(INSTALLDIR)/lib
	$(RANLIB) $(INSTALLDIR)/lib/libMisrToolkit.a
	$(RM) -f $(INSTALLDIR)/lib/$(LIB).so
	cd $(INSTALLDIR)/lib; $(LN) -s $(LIB).so.$(VERSION) $(LIB).so
# Install binary command utilities
	$(INSTALL) -m 755 $(BINEXE) $(INSTALLDIR)/bin
# Install binary command applications
	cd applications; make install INSTALLDIR="$(INSTALLDIR)"
	$(INSTALL) -m 755 $(BINEXE) $(INSTALLDIR)/bin
# Install documentation
	-@if [ -d "$(MTKHOME)/doc/html" ]; then \
		$(INSTALL) -m 644 doc/MISRToolkitConceptDiagramSmall.png \
			doc/Mtk_struct.html doc/Mtk_ug.pdf doc/pymtk.pdf \
			doc/cmdtable.html doc/footer.html doc/index.html $(INSTALLDIR)/doc ;\
		$(INSTALL) -m 644 doc/html/* $(INSTALLDIR)/doc/html ;\
		$(INSTALL) -m 644 doc/IDL_HTML_DOCS/* $(INSTALLDIR)/doc/IDL_HTML_DOCS ;\
		$(INSTALL) -m 755 doc/pymtk/* $(INSTALLDIR)/doc/pymtk ;\
	fi
# Install examples
	$(INSTALL) -m 644 $(MTKHOME)/examples/C/* $(INSTALLDIR)/examples/C
	$(INSTALL) -m 644 $(MTKHOME)/examples/idl/* $(INSTALLDIR)/examples/idl
	$(INSTALL) -m 644 $(MTKHOME)/examples/python/* $(INSTALLDIR)/examples/python
# Create Mtk_c_env.csh
	@$(RM) -f $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'setenv MTKHOME "$(INSTALLDIR)"' 					> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'setenv MTK_CFLAGS "-I$${MTKHOME}/include' \
		'-I$${HDFINC} -I$${HDFEOS_INC}"' 				>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'setenv MTK_LDFLAGS "-L$${MTKHOME}/lib -lMisrToolkit' \
		'-L$${HDFEOS_LIB} -L$${HDFLIB}' \
		'-lhdfeos -lGctp -lmfhdf -ldf -ljpeg -lz $(MATHLIB)"' 		>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'setenv PATH "$${PATH}:$${MTKHOME}/bin"' 				>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'if ($$?$(MTK_LD_PATH)) then'					>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo '	setenv $(MTK_LD_PATH) "$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' 	>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'else' 								>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo '	setenv $(MTK_LD_PATH) "$${MTKHOME}/lib"' 			>> $(INSTALLDIR)/bin/Mtk_c_env.csh
	@echo 'endif' 								>> $(INSTALLDIR)/bin/Mtk_c_env.csh
# Create Mtk_c_env.sh
	@$(RM) -f $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'export MTKHOME="$(INSTALLDIR)"' 					> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'export MTK_CFLAGS="-I$${MTKHOME}/include' \
		'-I$${HDFINC} -I$${HDFEOS_INC}"' 				>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'export MTK_LDFLAGS="-L$${MTKHOME}/lib -lMisrToolkit' \
		'-L$${HDFEOS_LIB} -L$${HDFLIB}' \
		'-lhdfeos -lGctp -lmfhdf -ldf -ljpeg -lz $(MATHLIB)"' 		>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'export PATH="$${PATH}:$${MTKHOME}/bin"' 				>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'if [ $${#$(MTK_LD_PATH)} -ne 0 ]; then' 			>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo '	export $(MTK_LD_PATH)="$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' 	>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'else' 								>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo '	export $(MTK_LD_PATH)="$${MTKHOME}/lib"' 			>> $(INSTALLDIR)/bin/Mtk_c_env.sh
	@echo 'fi' 								>> $(INSTALLDIR)/bin/Mtk_c_env.sh
# Install IDL MTK libraries if they were built
	@if [ -d "$(IDL_DIR)" -a -f "$(IDLLIB)" -a -f "$(IDLDLM)" ]; then \
		CMD='$(INSTALL) -m 644 $(IDLLIB) $(IDLDLM) $(INSTALLDIR)/lib/idl' ;\
		echo $$CMD; $$CMD ;\
		$(RM) -f $(INSTALLDIR)/bin/Mtk_idl_env.csh ;\
		echo 'setenv MTKHOME "$(INSTALLDIR)"' 				> $(INSTALLDIR)/bin/Mtk_idl_env.csh ;\
		echo 'setenv IDL_DLM_PATH "<IDL_DEFAULT>:$${MTKHOME}/lib/idl"' 	>> $(INSTALLDIR)/bin/Mtk_idl_env.csh ;\
		echo 'setenv IDL_PATH "<IDL_DEFAULT>:$${MTKHOME}/examples/idl"'	>> $(INSTALLDIR)/bin/Mtk_idl_env.csh ;\
		$(RM) -f $(INSTALLDIR)/bin/Mtk_idl_env.sh ;\
		echo 'export MTKHOME="$(INSTALLDIR)"' 				> $(INSTALLDIR)/bin/Mtk_idl_env.sh ;\
		echo 'export IDL_DLM_PATH="<IDL_DEFAULT>:$${MTKHOME}/lib/idl"' 	>> $(INSTALLDIR)/bin/Mtk_idl_env.sh ;\
		echo 'export IDL_PATH="<IDL_DEFAULT>:$${MTKHOME}/examples/idl"'	>> $(INSTALLDIR)/bin/Mtk_idl_env.sh ;\
	fi
# Install Python MTK libraries if they were built
	@if [ -f "$(PYTHONLIB)" ]; then \
		CMD='$(INSTALL) -m 644 $(PYTHONLIB) $(INSTALLDIR)/lib/python' ;\
		echo $$CMD; $$CMD ;\
		$(RM) -f $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'setenv MTKHOME "$(INSTALLDIR)"' 						> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'if ($$?PYTHONPATH) then' 							>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo '	setenv PYTHONPATH "$${PYTHONPATH}:$${MTKHOME}/lib/python"' 		>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'else' 									>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo '	setenv PYTHONPATH "$${MTKHOME}/lib/python"' 				>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'endif' 									>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'if ($$?$(MTK_LD_PATH)) then' 						>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo '	setenv $(MTK_LD_PATH) "$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"' 		>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'else' 									>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo '	setenv $(MTK_LD_PATH) "$${MTKHOME}/lib"' 				>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		echo 'endif' 									>> $(INSTALLDIR)/bin/Mtk_python_env.csh ;\
		$(RM) -f $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'export MTKHOME="$(INSTALLDIR)"' 						> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'if [ $${#PYTHONPATH} -ne 0 ]; then' 					>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo '	export PYTHONPATH="$${PYTHONPATH}:$${MTKHOME}/lib/python"'		>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'else' 									>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo '	export PYTHONPATH="$${MTKHOME}/lib/python"' 				>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'fi' 									>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'if [ $${#$(MTK_LD_PATH)} -ne 0 ]; then' 					>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo '	export $(MTK_LD_PATH)="$${$(MTK_LD_PATH)}:$${MTKHOME}/lib"'		>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'else' 									>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo '	export $(MTK_LD_PATH)="$${MTKHOME}/lib"' 				>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
		echo 'fi' 									>> $(INSTALLDIR)/bin/Mtk_python_env.sh ;\
	fi

#------------------------------------------------------------------------------
# Documentation build rules
#------------------------------------------------------------------------------

doc:
	@$(RM) -f $(MTKHOME)/doc/Doxyfile
	@$(SED) "s/<VERSION>/$(VERSION)/" $(MTKHOME)/doc/Doxyfile.template \
		> $(MTKHOME)/doc/Doxyfile
	@if [ -x "$(DOXYDIR)/doxygen" ]; then \
		$(DOXYDIR)/doxygen doc/Doxyfile ;\
	else \
		echo "*** WARNING *** Can not generate C documentation - doxygen not installed." ;\
	fi
	@if [ -x "$(PYDOCDIR)/Doc/tools/mkhowto" ]; then \
		cd doc; $(PYDOCDIR)/Doc/tools/mkhowto --html --pdf pymtk.tex; \
	else \
		echo "*** WARNING *** Can not generate Python documentation - mkhowto not installed." ;\
	fi

#------------------------------------------------------------------------------
# Routine list
#------------------------------------------------------------------------------

routinelist:
	@for i in $(SRC); do \
		echo $${i}; \
	done

#------------------------------------------------------------------------------
# Help target
#------------------------------------------------------------------------------

help:
	@echo "Usage example:"
	@echo "   make DEBUG=t test"
	@echo "                  - builds and runs debug tests"
	@echo "Environment variables:"
	@echo "   DEBUG=t        - enables -g debug compiler flag"
	@echo "   OPTLVL=x       - enables -Ox compiler flag (-O2 is default); x=(0,1,2,3,s)"
	@echo "   ADDITIONAL_CFLAGS=<additional cflags>"
	@echo "                  - use to specify additional CFLAGS such as -march=opteron"
	@echo "   MTK_INSTALLDIR=<Mtk install path>"
	@echo "                  - Mtk install path; defaults to $(INSTALLDIR)"
	@echo "   IDL_DIR=<idl path>"
	@echo "                  - path to IDL; used to build IDL libraries"
	@echo "Targets:"
	@echo "   misrcoord      - build only the misrcoord library"
	@echo "   odl            - build only the ODL library"
	@echo "   lib            - builds libMisrToolkit"
	@echo "   cmdutil        - build the command-line utilities"
	@echo "   applications   - build the applications"
	@echo "   all            - build all the above"
	@echo "   idl            - build the IDL wrapper shared library"
	@echo "   python         - build the Python wrapper shared library"
	@echo "   unit_test      - builds libMisrToolkit and Mtk tests"
	@echo "   test           - builds libMisrToolkit and runs Mtk tests"
	@echo "   timetest       - builds libMisrToolkit and runs Mtk tests using time"
	@echo "   valgrind       - builds libMisrToolkit and runs Mtk tests using valgrind"
	@echo "   testpython     - build the Python wrapper and run Python tests"
	@echo "   testidl        - build the IDL wrapper and run IDL tests (requires user interaction)"
	@echo "   testall        - build and run all tests (requires user interaction)"
	@echo "   doc            - build documentation"
	@echo "   cleanlib       - deletes libraries, objects and tests"
	@echo "   cleanmisrcoord - deletes misrcoord library"
	@echo "   cleanodl       - deletes ODL library"
	@echo "   cleanidl       - deletes IDL wrapper library"
	@echo "   cleanpython    - deletes Python wrapper library"
	@echo "   cleanemacs     - deletes emacs tilde files"
	@echo "   cleandepend    - performs dependency files"
	@echo "   clean          - performs all above cleans"
	@echo "   cleandoc       - deletes documentation"
	@echo "   cleanapplications - deletes applications"
	@echo "   tags           - build emacs tags"
	@echo "   distsrc        - creates a tar.gz of source in parent directory"
	@echo "   distdoc        - creates a tar.gz of documentation in parent directory"
	@echo "   distdata       - creates a tar.gz of testdata in parent directory"
	@echo "   dist           - creates a tar.gz of source and testdata in parent directory"
	@echo "   install        - installs Mtk into $(INSTALLDIR)"
	@echo "   routinelist    - list all Mtk routines by category"
	@echo "   help           - target help"

#------------------------------------------------------------------------------
# Implicit build rules
#------------------------------------------------------------------------------

# Rule to determine C header dependencies
%.d: %.c
	$(MTKHOME)/scripts/depend.sh `dirname $*.c` $(CFLAGS) $*.c > $@

# Rule to build .o file into the obj directory from .c files
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

#------------------------------------------------------------------------------
# Phony targets
#------------------------------------------------------------------------------

.PHONY: misrcoord odl lib all test idl python testpython testidl testall cmdutil applications valgrind doc clean cleanlib cleanmisrcoord cleanodl cleandoc cleantags cleanidl cleanpython cleanemacs cleantags cleandepend tags dist distsrc distdoc distdata install routinelist help

#------------------------------------------------------------------------------
# Include auto generated header dependencies
#------------------------------------------------------------------------------

ifneq "$(MAKECMDGOALS)" "clean"
  -include $(DEPEND)
endif
