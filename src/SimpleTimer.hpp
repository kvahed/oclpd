#ifndef __SIMPLE_TIMER_HPP__
#define __SIMPLE_TIMER_HPP__

#include "config.h"
#ifdef HAVE_MACH_ABSOLUTE_TIME
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#include "cycle.h"            // FFTW cycle implementation

#include <fstream>


const std::string exec (char* cmd) {

  FILE* pipe = popen(cmd, "r");
  if (!pipe)
	  return "ERROR";

  char buffer[128];
  std::string result = "";

  while(!feof(pipe))
      if(fgets(buffer, 128, pipe) != NULL)
        result += buffer;

  pclose(pipe);
  return result;

}


static const double FishyClockRate () {
	double ret = 1e6;

#if defined(HAVE_MACH_ABSOLUTE_TIME)
	// OS X
	uint64_t freq = 0;
	size_t   size = sizeof(freq);
	if (sysctlbyname("hw.tbfrequency", &freq, &size, NULL, 0) < 0)
		perror("sysctl");
	ret = (double) freq * 2.3;
#elif defined(HAVE_HRTIME_T)
	// LINUX
	std::string cmd ("lscpu | grep \"CPU MHz\"|awk '{print $3}'");
	std::string mhzstr = exec(&cmd[0]);
	ret = 1.0e6 * atof(mhzstr.c_str());
#endif

	return ret;
}

class SimpleTimer {

public: 
    inline SimpleTimer (const std::string& identifier = "") :
        _start (getticks()), _identifier(""), _stopped(false) , _net(0) {
    	fprintf (stderr, "      Processing ... \n", identifier.c_str());
    }

    inline ~SimpleTimer() {
        if (!_stopped)
            Stop();
        fprintf (stderr, "        ... done. cycles %.4f\n", 1.0 / FishyClockRate() * _net);
    }

    inline void Stop () {
        _net += elapsed(getticks(), _start);
        _stopped = true;
    }

    inline void Resume () {
        _stopped = false;
        _start = getticks();
    }
    
    const std::string _identifier;

private: 
    ticks _start;
    ticks _net;
    bool _stopped;
    
};

#endif // __SIMPLE_TIMER_HPP__
