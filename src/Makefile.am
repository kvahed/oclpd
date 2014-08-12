SUBDIRS = opencl
COMMON = Allocator.hpp Container.hpp cl.hpp CLProcessor.hpp CLProcessor.cpp cycle.h File.hpp InputParser.hpp HDF5File.hpp HDF5File.cpp NDData.hpp Options.cpp Options.hpp SimpleTimer.hpp
bin_PROGRAMS = oclpd
oclpd_SOURCES = $(COMMON) Pulsedesign.hpp oclpd.cpp
olcpd_CPPFLAGS = -Wno-deprecated-declarations


