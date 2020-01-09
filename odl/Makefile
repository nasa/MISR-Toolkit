ARCH :=		$(shell uname -s)

ifeq ($(ARCH),Darwin)
  MATHLIB := -lmx
endif
ifeq ($(ARCH),Linux)
  MATHLIB := -lm
endif

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

CC=     gcc
AR=	ar
CFLAGS= $(OPTFLAG) $(ARCH_CFLAGS) $(ADDITIONAL_CFLAGS) -Wall -pedantic -fPIC -fno-common \
	-I$(HDFEOS_INC) -I$(GCTPINC) -I$(HDFINC) -I$(JPEGINC)
LDFLAGS=-L$(HDFEOS_LIB) -L$(GCTPLIB) -L$(HDFLIB) -L$(JPEGLIB) \
        -lhdfeos -lGctp -lmfhdf -ldf -ljpeg -lz $(MATHLIB)

LIB =	libodl.a

INC=	odldef.h \
        odlinter.h \
        odldef_prototypes.h \
        odlparse.h


OBJS=	a_nodesa.o    cvtvalue.o   p_nodesp.o   prtsrc.o    wrtlabel.o \
        ag_nodesag.o  fmtvalue.o   parsact.o    rdlabel.o   wrtsrc.o \
        ao_nodesao.o  lexan.o      parser.o     rdvalue.o \
        comments.o    output.o     prtlabel.o   v_nodesv.o

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	@echo "Creating archive library $(LIB)..."
	$(AR) -rsu $(LIB) $(OBJS)

clean:
	/bin/rm -f $(LIB) $(OBJS)

$OBJS: $(INC)

.PHONY: lib all clean
