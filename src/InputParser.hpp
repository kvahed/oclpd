/*
 * InputParser.hpp
 *
 *  Created on: Nov 7, 2013
 *      Author: kvahed
 */

#ifndef INPUTPARSER_HPP_
#define INPUTPARSER_HPP_

#include "Options.hpp"
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include <vector>
#include <exception>

static const bool
ParseInput (int args, char** argv, bool& verbose, bool& query,
		cl_device_type& cldtype, std::vector<unsigned short>& devs,
		std::string& code_uri, std::string& din_uri, std::string& dout_uri) {

	char* tmp;
	Options opts;

	opts.addUsage  ("Copyright (C) 2013");
	opts.addUsage  ("Kaveh Vahedipour <kaveh@codeare.org>");
	opts.addUsage  ("Juelich Research Centre");
	opts.addUsage  ("Medical Imaging Physics");
	opts.addUsage  ("");
	opts.addUsage  ("Usage:");

	opts.setFlag   ("help"    ,'h');

	opts.addUsage  (" -v, --verbose     Verbose output");
	opts.addUsage  (" -q, --query-devs  Query devices");
	opts.addUsage  (" -u, --use-devs    List of devices to be used (-q first?)");
	opts.addUsage  (" -c, --code-file   Complete path (default: src/opencl/sim.cl)");
	opts.addUsage  (" -i  --data-in     Input data (default: data/r1.h5)");
	opts.addUsage  (" -o  --data-out    Output data (default: out.h5)");
	opts.addUsage  ("");
	opts.addUsage  (" -h, --help    Print this help screen");
	opts.addUsage  ("");
	opts.addUsage  ("Examples:");
	opts.addUsage  ("  oclpd -q");
	opts.addUsage  ("  oclpd -c src/opencl/sim.cl -i data/r1.h5");
	opts.addUsage  ("  oclpd -c src/opencl/sim.cl -i data/r1.h5 -o r1out.h5");

	opts.setFlag   ("help"       , 'h');
	opts.setFlag   ("verbose"    , 'v');
	opts.setFlag   ("query-devs" , 'q');
	opts.setOption ("code-file"  , 'c');
	opts.setOption ("data-in"    , 'i');
	opts.setOption ("data-out"   , 'o');
	opts.setOption ("user-devs"  , 'u');

	opts.processCommandArgs(args, argv);

	if (opts.getFlag("help")) {
		opts.printUsage();
		return false;
	}

	code_uri.assign ((tmp = opts.getValue("code-file")) ? tmp : "");
    din_uri.assign  ((tmp = opts.getValue("data-in"))   ? tmp : "");
    dout_uri.assign ((tmp = opts.getValue("data-out"))  ? tmp : "");
    verbose               = opts.getFlag("verbose");
    query                 = opts.getFlag("query-devs");
    tmp = opts.getValue("user-devs");
    if (tmp) {
		try {
			devs.push_back((unsigned short)atoi(std::string(tmp).c_str()));
		} catch (const std::exception& e) {
			fprintf (stderr, "oclpd: device list must be comma seperated list of positive integers. f.e. -u 0,1\n");
			return false;
		}
    }

	return true;

}
#endif /* INPUTPARSER_HPP_ */
