#include "CLProcessor.hpp"
#include "NDData.hpp"

using namespace codeare::opencl;


static const float GB = 1024.*1024.*1024.;

inline static std::pair<const char*, unsigned>
ReadCLFile (const std::string& fname) {
    std::string str, src;
    std::ifstream in;
    if(!in)
        std::cerr << "File not found??" << std::endl;
    in.open (fname.c_str());
    std::getline (in, str);
    while (in) {
        src += str + "\n";
        std::getline (in, str);
    }
    in.close();
    return std::make_pair(src.c_str(), src.length());
}


inline static const void
PlatformInfo (const std::vector<cl::Platform>& _platforms) {
    fprintf (stderr, "    %zu platforms:\n", _platforms.size());
    for (size_t i = 0; i < _platforms.size(); ++i)
    	fprintf (stderr, "        %zu: %s\n", i,
                _platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
}


inline static const void
DeviceInfo (const std::vector<cl::Device>& _devices) {
	fprintf (stderr, "    %zu devices:\n", _devices.size());
    for (size_t i = 0; i < _devices.size(); ++i)
        if (_devices[i].getInfo<CL_DEVICE_AVAILABLE>())
        	fprintf (stderr, "        %zu: %s mem(%.1fGB) ver(%s)\n", i,
                    _devices[i].getInfo<CL_DEVICE_NAME>().c_str(),
                    (float)_devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/GB,
                    _devices[i].getInfo<CL_DRIVER_VERSION>().c_str());
}


CLProcessor::CLProcessor (const std::vector<unsigned short>& devs,
		const cl_device_type dtype) : _status (CL_SUCCESS) {

	std::vector<cl::Platform> platforms;

    try {
        // Platform
        cl::Platform::get(&platforms);
        PlatformInfo (platforms);
    } catch (const cl::Error& cle) {
        _status = cle.err();
        fprintf (stderr, "  ERROR(Platform): %s(%d)\n", cle.what(), cle.err());
	return;
    }

    for (size_t i = 0; i < platforms.size(); ++i) {
        // Context
        try {
            cl_context_properties properties[] = 
                { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(), 0};
            _context = cl::Context (dtype, properties);
            // Devices
            std::vector<cl::Device> tmppf = _context.getInfo<CL_CONTEXT_DEVICES>(); 
            _devices.insert(_devices.end(), tmppf.begin(), tmppf.end());
        } catch (const cl::Error&) {}
    }

    if (!devs.empty())
	    for (unsigned short i = 0; i < devs[0]; ++i)
	    	_devices.erase(_devices.begin());

    Queue(devs,true);
    DeviceInfo (_devices);

}


CLProcessor::~CLProcessor () {
	_queue.finish();
}


const int
CLProcessor::Build (const std::string& ksrc) {

    std::pair<const char*, unsigned> kpair = ReadCLFile (ksrc);
    fprintf (stderr, "    OpenCL program %s is %d bytes.\n    Assembling program ... ",
           ksrc.c_str(), (int) kpair.second); fflush (stdout);
    
    try {
        cl::Program::Sources cps (1, kpair);
        _program = cl::Program(_context, cps);
        fprintf (stderr, "done.\n    Builing    program ... "); fflush (stdout);
    } catch (const cl::Error& cle) {
        _status = cle.err();
        fprintf (stderr, "FAILED: %s(%s)\n", cle.what(), StatusStr());
        return cle.err();
    }
    
    try {
        _status = _program.build(_devices);
        fprintf (stderr, "done.\n");
    } catch (const cl::Error& cle) {
        _status = cle.err();
        fprintf (stderr, "FAILED. Check logfile!\n");
        fprintf (stderr, "      Build Status:\t %d\n",
        		_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(_devices[0]));
        fprintf (stderr, "      Build Options:\t %s\n",
        		_program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(_devices[0]).c_str());
        fprintf (stderr, "      Build Log:\t %s\n",
        		_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_devices[0]).c_str());
        return cle.err();
    }

    return CL_SUCCESS;
    
}

cl::Context& CLProcessor::Context () {
    return _context;
}

cl::Program& CLProcessor::Program () {
    return _program;
}

cl::Kernel CLProcessor::MakeKernel (const std::string& name) {
	return cl::Kernel (_program, name.c_str());
}


cl::CommandQueue& CLProcessor::Queue (const std::vector<unsigned short>& devs,
		const bool profiling) {

	try {
		_queue = cl::CommandQueue
				(_context, _devices[0], profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &_status);
	} catch (const cl::Error& cle) {
		fprintf (stderr, "    Failed to create command queue.\n");
		_status = cle.err();
	}

	return _queue;

}


const double CLProcessor::Run (const cl::Kernel& kern,
		const size_t nkern, const size_t wsize, const bool profiling) {

	size_t optsize = kern.getWorkGroupInfo <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(_devices[0]);
	double wtime = 0.;

    try {
    	if (profiling) {
    		printf ("    Running  %zu x %s ... ", nkern, kern.getInfo<CL_KERNEL_FUNCTION_NAME>().c_str());
    		fflush (stdout);
    	}
        cl::Event event;

        optsize = wsize * kern.getWorkGroupInfo <CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(_devices[0]);

        _queue.enqueueNDRangeKernel (kern, cl::NullRange, cl::NDRange(nkern),
            		(optsize) ? cl::NDRange(optsize) : cl::NullRange, NULL, &event);
        event.wait();
        _queue.finish();
    	wtime = 1.0e-6 * (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
                - event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

        if (profiling)
            printf (" wsize (%zu) (%03.1f ms) ... done. \n", optsize, wtime);

    } catch (const cl::Error& cle) {
    	_status = cle.err();
        fprintf (stderr, "  ERROR: %s(%d)\n", cle.what(), cle.err());
        return 0.0;;
    }

    return wtime;
    
}


const int
CLProcessor::Status () const {
    return _status;
}


static const char* statusString[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
};

inline const char*
CLProcessor::StatusStr () {
    
    const size_t index = -_status;
    return (index >= 0 && index < sizeof(statusString)/sizeof(statusString[0])) ?
    		statusString[index] : "";

}
    



