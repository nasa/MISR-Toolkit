MTKHOME :=      $(shell pwd)/../..

DAPDIR :=	/usr
#DAPDIR :=	/usr/sciflo/sciflo

DAPCXXFLAGS :=	$(shell $(DAPDIR)/bin/dap-config --cflags)
DAPLDFLAGS :=	$(shell $(DAPDIR)/bin/dap-config --server-libs)

MODULES :=      Util FileQuery UnitConv CoordQuery MapQuery OrbitPath
MODULES +=      SetRegion ReadData WriteData

CXXFLAGS :=	-DDEBUG -Wall -pedantic -fPIC
CXXFLAGS +=	-I$(MTKHOME)/include 
CXXFLAGS +=	$(patsubst %, -I$(MTKHOME)/%/include, $(MODULES))
CXXFLAGS +=	-I$(MTKHOME)/misrcoord -I$(MTKHOME)/odl
CXXFLAGS +=	-I$(HDFINC) -I$(HDFEOS_INC) -I$(HDFEOS_INC)/../gctp/include
CXXFLAGS +=	$(DAPCXXFLAGS)

LDFLAGS := 	-L$(MTKHOME)/lib -lMisrToolkit
LDFLAGS +=      -L$(HDFEOS_LIB) -L$(HDFLIB) -L$(HDFEOS_LIB)/../../gctp/lib
LDFLAGS +=      -lhdfeos -lGctp -lmfhdf -ldf -ljpeg -lz -lm
LDFLAGS +=	$(DAPLDFLAGS)

dap_misr_handler: dap_misr_handler.o read_attributes.o read_descriptors.o misr_types.o misr_daptype.o

misr_types.o: misr_types.h

clean:
	@rm -f *.o dap_misr_handler

cleanemacs:
	@rm -f *~


