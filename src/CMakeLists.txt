list (APPEND CORE_SRC Allocator.hpp Container.hpp cl.hpp
  CLProcessor.hpp CLProcessor.cpp cycle.h File.hpp InputParser.hpp
  HDF5File.hpp HDF5File.cpp NDData.hpp Options.cpp Options.hpp
  SimpleTimer.hpp) 

add_executable (oclpd ${CORE_SRC} oclpd.cpp)
target_link_libraries (oclpd hdf5 hdf5_cpp ${OPENCL_LIBRARIES}) 
install (TARGETS oclpd DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

include (TestMacro)

add_test(NAME oclpd
    COMMAND oclpd
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

