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

LIB =	libmisrcoord.a

INC=	misrproj.h \
	errormacros.h

OBJS=	misr_init.o \
	misrinv.o \
	misrfor.o

all: misrcoordex lib

misrcoordex: misrcoordex.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $@.o $(OBJS) $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	@echo "Creating archive library $(LIB)..."
	$(AR) -rsu $(LIB) $(OBJS)

clean:
	/bin/rm -f misrcoordex misrcoordex.o $(LIB) $(OBJS)

misrcoordex.o $OBJS: $(INC)

.PHONY: lib all clean
