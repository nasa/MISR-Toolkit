ARCH :=		$(shell uname -s)

ifeq ($(ARCH),Darwin)
  MATHLIB := -lmx
endif
ifeq ($(ARCH),Linux)
  MATHLIB := -lm
endif

CC=     gcc
AR=	ar
CFLAGS= $(OPTFLAG) $(ARCH_CFLAGS) $(ADDITIONAL_CFLAGS) -Wall -pedantic -fPIC -fno-common \
	-I$(HDFINC) -I$(HDFEOS_INC) -I$(HDFEOS_INC)/../gctp/include
LDFLAGS=-L$(HDFEOS_LIB) -L$(HDFLIB) -L$(HDFEOS_LIB)/../../gctp/lib \
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
