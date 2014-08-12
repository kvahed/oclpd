#ifndef __MR_SIM_DATA__
#define __MR_SIM_DATA__

#include "CLProcessor.hpp"
#include "HDF5File.hpp"
#include "SimpleTimer.hpp"

/**
 * @brief  Pulse design according to
 *         Vahedipour et al, "Time reversed Integration ...", ISMRM 2012, Melbourne, AUS
 */
template<class T>
class PulseDesign {

    typedef std::complex<T> cplx;
    typedef T               real;

    unsigned nr, nc, nk;

    bool _verbose;

    NDData<cplx> b1, rf;
    NDData<real>  r, b0, m0, gs, g, j, m, ic, tm0;     // MR data

    cl::Buffer rfbuf, b1buf, rbuf, m0buf, mbuf, b0buf, pbuf,
    	xbuf, gsbuf, gbuf, brfbuf, jbuf, icbuf, tm0buf;  // OpenCL representations

    std::string _out_file; // Output file

public:


    /**
     * @brief Default constructor
     */
    PulseDesign () : nr(0), nc(1), nk(0), _verbose(false) {}


    /**
     * @brief Construct with IO
     *
     * @param  in_file   Incoming (b0, b1, r, m0, gs, g, j)
     * @param  out_file  Outgoing (rf, ic, m)
     */
    PulseDesign (const std::string& in_file,
    		const std::string& out_file = "out.h5", const bool verbose = false) :
    			_out_file (out_file), _verbose(verbose) {

    	// Read data
        HDF5File f;
        f  = fopen (in_file);
        b1 = fread<cplx>(f, "b1"); // b1 Sensitivity maps      nr x nc
        r  = fread<real>(f,  "r"); //  r Spatial positions      3 x nr
        m0 = fread<real>(f, "m0"); // m0 Pattern                3 x nr
        b0 = fread<real>(f, "b0"); // b0 b0 map O(nr)          nr
        gs = fread<real>(f, "gs"); // gs Gradient sensitivity   3 x nr
        g  = fread<real>(f,  "g"); //  g Gradient trajectory    3 x nk
        j  = fread<real>(f,  "j"); //  j Jacobian determinant  nk
        tm0 = fread<real>(f,"tm0"); //  j Jacobian determinant nk
        fclose (f);

        // Sizes & stuff
        nr  = size(r,  1);
        nc  = size(b1, 1);
        nk  = size(g,  1);

        // Intermediate and outgoing.
        rf  = NDData<cplx> (nk,nc);    // rf RF pulses nk x nc
        m   = NDData<real> (size(m0)); // Excited magnetisation
        ic  = NDData<real> (nr);       // Intensity correction
    }

    /**
     * @brief Write output
     */
    ~PulseDesign () {
    	HDF5File f (_out_file, OUT);
		fwrite (f, rf);
		fwrite (f, m);
		fwrite (f, ic);
		fclose (f);
    }
    
    /**
     * @brief  Upload to GPU run design algorithm and download data
     *
     * @param cp  Assigned processor class
     */
    inline void DesignOn (codeare::opencl::CLProcessor& cp) {
    	GPUUpload (cp);
    	CGNR (cp);
    	GPUDownload (cp);
    }

protected:

    /**
     * @brief Upload data to GPU
     *
     * @param  cp Assigned processor class
     */
    inline void GPUUpload (codeare::opencl::CLProcessor& cp) {
    	cp.Copy (b1, b1buf);
    	cp.Copy ( r,  rbuf);
    	cp.Copy (m0, m0buf);
    	cp.Copy (b0, b0buf);
    	cp.Copy (gs, gsbuf);
    	cp.Copy ( g,  gbuf);
    	cp.Copy ( j,  jbuf);
    	cp.Copy (tm0, tm0buf);
    	mbuf  = cl::Buffer (cp.Context(), CL_MEM_READ_WRITE,
    			sizeof(real) * 3*nr);               // Excitation profile
    	rfbuf = cl::Buffer (cp.Context(), CL_MEM_READ_WRITE,
    			sizeof(cplx) * nc*nk); 			 // RF scratch buffer
    	brfbuf = cl::Buffer (cp.Context(), CL_MEM_READ_WRITE,
    			sizeof(cplx) * nc*nk*nr);  // RF buffer
    	icbuf = cl::Buffer (cp.Context(), CL_MEM_READ_WRITE,
    			sizeof(real) * nr);              // Intesity correction
    }

    /**
     * @brief Retrieve result from GPU
     */
    inline void GPUDownload (codeare::opencl::CLProcessor& cp) {
    	cp.Copy (rfbuf, rf);
    	cp.Copy ( mbuf,  m);
    	cp.Copy (icbuf, ic);
    }

    void CGNR (codeare::opencl::CLProcessor& cp) {

        cl::Kernel simacq = cp.MakeKernel("simacq"),
        		   simexc = cp.MakeKernel("simexc"),
		           redsig = cp.MakeKernel("redsig"),
		           intcor = cp.MakeKernel("intcor"),
		           zerorf = cp.MakeKernel("zerorf");

        float    dt = 1.0e-2;

        zerorf.setArg( 0, brfbuf);  

        intcor.setArg( 0,  b1buf); intcor.setArg( 1,  nc);
        intcor.setArg( 2,  nr);    intcor.setArg( 3,  icbuf);

        simacq.setArg( 0,  b1buf); simacq.setArg( 1,   gbuf);
        simacq.setArg( 2,   rbuf); simacq.setArg( 3,  b0buf);
        simacq.setArg( 4,  gsbuf); simacq.setArg( 5,  m0buf);
        simacq.setArg( 6,  icbuf); simacq.setArg( 7,  nr);
        simacq.setArg( 8,  nc);    simacq.setArg( 9,  nk);
        simacq.setArg(10,  dt);    simacq.setArg(11, brfbuf);

        redsig.setArg( 0, brfbuf); redsig.setArg( 1,  jbuf);
        redsig.setArg( 2,  nc);    redsig.setArg( 3,  nk);
        redsig.setArg( 4,  nr);    redsig.setArg( 5,  rfbuf);

        simexc.setArg( 0,  b1buf); simexc.setArg( 1,   gbuf);
        simexc.setArg( 2,  rfbuf); simexc.setArg( 3,   rbuf);
        simexc.setArg( 4,  b0buf); simexc.setArg( 5,  gsbuf);
        simexc.setArg( 6, tm0buf); simexc.setArg( 7,  nr);
        simexc.setArg( 8,  nc);    simexc.setArg( 9,  nk);
        simexc.setArg(10,  dt);    simexc.setArg(11,   mbuf);

        double wtime = 0.;
        wtime += cp.Run (zerorf, 2*nk*nc*nr, 16, _verbose); // Reset
		wtime += cp.Run (intcor,         nr,  8, _verbose); // Intensity correction
		wtime += cp.Run (simacq,         nr,  4, _verbose); // Acquire
		wtime += cp.Run (redsig,    2*nk*nc,  0, _verbose); // Reduce signals
		wtime += cp.Run (simexc,         nr,  4, _verbose); // Excite

        printf ("    Running    program ... done; wtime: %.3fs.\n", 1.0e-3*wtime);

    }

};


#endif //__MR_SIM_DATA__
