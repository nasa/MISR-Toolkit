#
# Makefile using MISR Toolkit
#

CFLAGS := $(MTK_CFLAGS)		# Add your own CFLAGS here
LDFLAGS := $(MTK_LDFLAGS)	# Add your own LDFLAGS here

baz: foo.o bar.o biz.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	@rm -f foo.o bar.o biz.o baz