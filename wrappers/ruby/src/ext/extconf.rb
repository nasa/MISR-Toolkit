require "mkmf"
mtkinc, mtklib = dir_config('mtk', ENV['MTKHOME'])
dir_config('hdfeos', ENV['HDFEOS_INC'], ENV['HDFEOS_LIB'])
dir_config('hdf', ENV['HDFINC'], ENV['HDFLIB'])
# narray.h gets installed in the ruby site directory
$INCFLAGS = $INCFLAGS + " -I" + Config::CONFIG["sitearchdir"]
$CPPFLAGS = $CPPFLAGS + " -I" + Config::CONFIG["sitearchdir"]
no_narray = false
no_misr = false
no_hdf = false
no_hdfeos = false
no_narray = true unless have_header('narray.h')
no_hdf = true unless have_library("z")
no_hdf = true unless have_library("jpeg")
no_hdf = true unless have_library("df")
no_hdf = true unless have_library("mfhdf")
no_hdfeos = true unless have_library("Gctp")
no_hdfeos = true unless have_library("hdfeos")
have_library("stdc++")
no_misr = true unless have_library('MisrToolkit')
# Extra stuff needed on older versions of Ruby. Newest version doesn't seem
# to need this.
$libs = "-Wl,-rpath #{mtklib} #{$libs}"
if no_narray
  puts(<<END)
This requires the NArray Ruby library. This can be found 
at http://www.ir.isas.jaxa.jp/~masa/ruby/index-e.html 
(it is not a gem, so this gem can't automatically install it)
END
end
if no_misr
  puts(<<END)
This requires the MISR Toolkit C libraries be installed, and the location 
pointed to by the environment variable MTKHOME

END
end
if no_hdfeos
  puts(<<END)
This requires the HDF-EOS library be installed, and pointed to by the
environment variables HDFEOS_INC and HDFEOS_LIB. These variables are
normally set automatically by sourcing the appropriate hdfeos_env.csh.

END
end
if no_hdf
  puts(<<END)
This requires the HDF library be installed, and pointed to by the
environment variables HDFINC and HDFLIB. These variables are
normally set automatically by sourcing the appropriate hdfeos_env.csh.

END
end
unless 
    no_narray ||
    no_misr ||
    no_hdfeos ||
    no_hdf
  create_makefile("mtk")
end
