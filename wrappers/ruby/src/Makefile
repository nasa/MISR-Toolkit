RDOC = rdoc
RUBY = ruby

# The Ruby library NArray should be installed and on the Ruby path.

# These are normally set by the setup of the HDFEOS and HDF library. Set by 
# hand if needed.

# HDFEOS_INC
# HDFEOS_LIB
# HDFLIB
# HDFINC

# Location of MISR toolkit

export MTKHOME = $(PWD)/../../../
export MTKTESTHOME = $(PWD)/../../../../Mtk_testdata/
all:
	cd ext; $(RUBY) extconf.rb; $(MAKE) all

make_doc:
	$(RDOC) --title "MISR Toolkit - Ruby Interface" --template lib/doc_template/jamis.rb --inline-source -m "lib/README" ext lib lib/README

test: all
	$(RUBY) test/test_all.rb

clean:
	-rm ext/Makefile ext/mkmf.log ext/*.o ext/*.so ext/*.bundle

gem:
	gem build misrtoolkit.gemspec
	mv misrtoolkit-*.gem ..

install:
	-mkdir $(MTK_INSTALLDIR)/lib/ruby
	-install -m 644 ext/*.h lib/* $(MTK_INSTALLDIR)/lib/ruby
