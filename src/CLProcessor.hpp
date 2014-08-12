#ifndef __CL_CONTEXT_HPP__
#define __CL_CONTEXT_HPP__

#include "NDData.hpp"

#include <iostream>
#include <fstream>
#include <map>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

/**
 * @brief Create CL context
 */
namespace codeare {
    namespace opencl {
        
        class CLProcessor {
            
        public:
            
            CLProcessor (const std::vector<unsigned short>& devs = std::vector<unsigned short>(),
            		const cl_device_type dtype = CL_DEVICE_TYPE_DEFAULT);
            ~CLProcessor ();
            const int Status () const;
            const int Build (const std::string& ksrc);
            const double Run (const cl::Kernel& kern, const size_t nkern,
            		const size_t wsize, const bool profiling = false);

            const char* StatusStr ();

            cl::Context& Context();

            cl::Program& Program();

            cl::CommandQueue& Queue(const std::vector<unsigned short>& devs, const bool profiling);

            cl::Kernel  MakeKernel (const std::string& name);

            template<class T> const int
            Copy (NDData<T>& data, cl::Buffer& buf) {
            	buf = cl::Buffer (_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            			data.Size()*sizeof(T), data.Ptr());
            }

            template<class T> const int
            Copy (const cl::Buffer& buf, NDData<T>& data){
            	size_t sf = sizeof(cl_float);
            	cl::CommandQueue queue;
            	try{
            	    queue = cl::CommandQueue(_context, _devices[0], 0, &_status);
            		queue.enqueueReadBuffer(buf, CL_TRUE, 0, data.Size()*sizeof(T), data.Ptr());
            	} catch (const cl::Error& cle) {
            	    printf("ERROR: %s(%d)\n", cle.what(), cle.err());
            	    _status = cle.err();
            	}
            	queue.finish();
            	return _status;
            }

        protected:

            std::string _fname;
            std::vector<cl::Device>  _devices;
            cl::Context              _context;
            cl::Program              _program;    /**!<   */
        	cl::Event _event;
        	cl::CommandQueue _queue;

            int                      _status;  // error code returned from api calls
            
        };
        
    }
}

#endif //__CL_CONTEXT_HPP__
