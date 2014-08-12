/*
 * ocllorenz.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: kvahed
 */

#include "InputParser.hpp"
#include "Lorenz.hpp"

int main (int args, char** argv) {

	using namespace codeare::opencl;

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


}



