require 'rubygems'
SPEC = Gem::Specification.new do |s|
  s.name = "misrtoolkit"
  s.version = "1.0.1"
  s.author = "Brian Rheingans"
  s.email = "Brian.Rheingans@jpl.nasa.gov"
  s.homepage = "http://www-misr.jpl.nasa.gov"
  s.summary = "A package for reading MISR Level 1 and Level 2 data"
  candidates = Dir.glob("{bin,lib,test,ext}/**/*")
  candidates.push("doc/oo_uml_design.png")
  s.files = candidates.delete_if do |item|
    item.include?(".svn") || item.include?(".o") || 
      item.include?(".log") || item.include?(".bundle") ||
      item.include?(".so") || item.include?("Makefile") ||
      item.include?("~") || item.include?("doc_template")
  end
  s.require_path = "lib"
  s.autorequire = "mtk_file"
  s.has_rdoc = true
  s.rdoc_options << "--title" << "'MISR Toolkit - Ruby Interface'" <<
    "-m" << "lib/README" << "--exclude" << "lib/doc_template/jamis.rb" <<
    "--include" <<  "./doc" 
  s.extra_rdoc_files = ["lib/README", "ext/mtk_ext.cc"]
  s.test_file = "test/test_all.rb"
  s.extensions << "ext/extconf.rb"
end     
