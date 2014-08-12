#include "InputParser.hpp"
#include "PulseDesign.hpp"


int main (int args, char** argv) {

	std::string code_uri;
	std::string din_uri;
	std::string dout_uri;
	bool verbose;
	bool query;
	std::vector<unsigned short> devs;
	cl_device_type cldtype;

	if (!ParseInput (args, argv, verbose, query, cldtype, devs, code_uri, din_uri, dout_uri))
		return 0;

    using namespace codeare::opencl;

    // GPU platform, devices, program and queue
    CLProcessor clp (devs);
    if (clp.Status() != CL_SUCCESS)
    	return 1;
    else if (query)
    	return 0;

    // Build OpenCL program
    if (code_uri.empty())
    	code_uri = "src/opencl/sim.cl";
    clp.Build (code_uri);
    if (clp.Status() != CL_SUCCESS)
    	return 1;

    if (din_uri.empty())
    	din_uri = "data/r1.h5";
    if (dout_uri.empty())
    	dout_uri  = "out.h5";

    // Setup pulse design
    PulseDesign<float> pd (din_uri, dout_uri, verbose);

    // Design.
    pd.DesignOn(clp);
    
    return 0;

}

