########################################################################################################
# C++
#
# Get list of source files.
cxx_source := $(wildcard $(local_dir)/src/*.cc)

# Generate list of dependency files, including unit-tests.
cxx_dep := $(patsubst $(local_dir)/src/%.cc, $(local_dir)/dep/%.d, $(cxx_source))

# Generate list of object files, including unit-tests.
cxx_obj := $(patsubst $(local_dir)/src/%.cc, $(local_dir)/src/%.o, $(cxx_source))

# Generate list of object files, excluding unit-tests and executables.
cxx_libobj := $(filter-out %_main.o, $(filter-out %_test.o, $(cxx_obj)))

# Generate list of unit-tests and outputs
cxx_unit_test_source := $(filter %_test.cc, $(cxx_source))
cxx_unit_test := $(patsubst $(local_dir)/src/%_test.cc, $(local_dir)/unit_test/%_test, $(cxx_unit_test_source))
cxx_test_out := $(patsubst $(local_dir)/unit_test/%_test, $(local_dir)/testout/%_test.out, $(cxx_unit_test))

# Generate list of main exectables
cxx_exec_source := $(filter %_main.cc, $(cxx_source))
cxx_exec := $(patsubst $(local_dir)/src/%_main.cc, $(local_dir)/bin/%_main, $(cxx_exec_source))
