/*
 * MATFile.hpp
 *
 *  Created on: May 20, 2013
 *      Author: kvahed
 */

#ifndef MATFILE_HPP_
#define MATFILE_HPP_

#include "Demangle.hpp"
#include "File.hpp"

#include <mat.h>

#include <complex>

template <class T> inline static bool
MXValidate  (const NDData<T>& M, const mxArray* mxa) {

	T t;
	mxClassID     mcid = mxGetClassID(mxa);
	std::string cplx = (mxIsComplex(mxa)) ? "complex" : "real";

	const char* vname = demangle(typeid(T).name()).c_str();

	if (is_singlep(t) && mcid == 7)
		return true;
	if (is_doublep(t) && mcid == 6)
		return true;
	else {
		printf ("Matrix is %s, yet Matlab variable is %s %s\n", vname, mxGetClassName(mxa), cplx.c_str());
		return false;
	}

	return true;

}



namespace codeare {
namespace io {

template <class T>
struct MXTraits;

template <class T>
static void write_real (mxArray* mxa, const NDData<T>& M) {
	memcpy(mxGetData(mxa), M.Ptr(), numel(M) * sizeof(T));
}

template <class T>
static void read_real (NDData<T>& M, const mxArray* mxa) {
	memcpy (&M[0], mxGetPr(mxa), numel(M) * sizeof (T));
}

template <class T>
static void write_complex (mxArray* mxa, const NDData<std::complex<T> >& M) {
	T* re = (T*)mxGetPr(mxa);
	T* im = (T*)mxGetPi(mxa);
	for (size_t i = 0; i < numel(M); ++i) {
		re[i] = real(M[i]);
		im[i] = imag(M[i]);
	}
}

template <class T>
static void read_complex (NDData<std::complex<T> >& M, const mxArray* mxa) {
	T* re = (T*)mxGetPr(mxa);
	T* im = (T*)mxGetPi(mxa);
	bool cplx = (im != NULL);
	for (size_t i = 0; i < numel(M); ++i)
		M[i] = std::complex<T>(re[i], (cplx) ? im[i] : 0.0);
}

template <> struct MXTraits<float> {
	static const mxClassID prec = mxSINGLE_CLASS;
	static const mxComplexity cplx = mxREAL;
	typedef float T;
	static void Write (mxArray* mxa, const NDData<T>& M) {write_real(mxa,M);}
	static void Read (NDData<T>& M, const mxArray* mxa) {read_real(M,mxa);}
};
template <> struct MXTraits<double> {
	static const mxClassID prec = mxDOUBLE_CLASS;
	static const mxComplexity cplx = mxREAL;
	typedef double T;
	static void Write (mxArray* mxa, const NDData<T>& M) {write_real(mxa,M);}
	static void Read (NDData<T>& M, const mxArray* mxa) {read_real(M,mxa);}
};
template <> struct MXTraits<std::complex<float> > {
	static const mxClassID prec = mxSINGLE_CLASS;
	static const mxComplexity cplx = mxCOMPLEX;
	typedef std::complex<float>  T;
	typedef float T2;
	static void Write (mxArray* mxa, const NDData<T>& M) {write_complex(mxa,M);}
	static void Read (NDData<T>& M, const mxArray* mxa) {read_complex(M,mxa);}
};
template <> struct MXTraits<std::complex<double> > {
	static const mxClassID prec = mxDOUBLE_CLASS;
	static const mxComplexity cplx = mxCOMPLEX;
	typedef std::complex<double>  T;
	typedef double T2;
	static void Write (mxArray* mxa, const NDData<T>& M) {write_complex(mxa,M);}
	static void Read (NDData<T>& M, const mxArray* mxa) {read_complex(M,mxa);}
};
template <> struct MXTraits<short> {
	static const mxClassID prec = mxINT16_CLASS;
	static const mxComplexity cplx = mxREAL;
	typedef std::complex<double>  T;
	static void Write (mxArray* mxa, const NDData<T>& M) {write_complex(mxa,M);}
	static void Read (NDData<T>& M, const mxArray* mxa) {read_real(M,mxa);}
};

	class MXFile : public File {

	public:

		MXFile (const std::string& fname, const IOMode mode, const bool verbose = false) :
			File (fname, mode, verbose) {

			if (verbose)
				printf ("Opening %s (%s).\n", fname.c_str(), (mode == READ) ? "r" : "w");

			_file = matOpen (fname.c_str(), (mode == READ) ? "r" : "w" );

			if (_file == NULL) {
				printf ("Error opening MATLAB file %s\n", fname.c_str());
				assert (false);
			}

		}

		virtual ~MXFile () {

			if (_file)
				if (matClose(_file) != 0)
				printf ("Error closing file %s\n",this->_fname.c_str());

		}

		template<class T> NDData<T>
		Read (const std::string& uri) const {

			mxArray*      mxa = matGetVariable (_file, uri.c_str());
			NDData<T> M;

			if (!mxa) {
				printf ("**ERROR**: Failed to retrieve variable %s\n", uri.c_str());
				assert (false);
			}

			mxClassID     mcid = mxGetClassID(mxa);
			mwSize        ndim = mxGetNumberOfDimensions(mxa);
			const mwSize*  dim = mxGetDimensions(mxa);
			size_t i = 0;

			assert (MXValidate (M, mxa));

			std::vector<size_t> mdims(ndim,1);

			for (; i < ndim; ++i)
				mdims[i] = (size_t)dim[i];

			M = NDData<T> (mdims);

			// Copy from memory block ----------------------
			MXTraits<T>::Read(M, mxa);

			// Clean up and close file -------------------
			if (mxa != NULL)
				mxDestroyArray(mxa);

			return M;


		}

		template <class T> bool
		Write (const NDData<T>& M, const std::string& uri) {

			// Declare dimensions and allocate array
			size_t nd = M.NDim();
			std::vector<mwSize> dim(nd);
			for (size_t i = 0; i < nd; ++i)
				dim[i] = (mwSize)M.Dim(i);
			mxArray*  mxa = mxCreateNumericArray (nd, &dim[0], MXTraits<T>::prec, MXTraits<T>::cplx);

			// Assign data
			MXTraits<T>::Write (mxa, M);

			// Write data
			int status = matPutVariable(_file, uri.c_str(), mxa);
			if (status != 0) {
				printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
				return false;
			}

			// Clean up RAM
			if (mxa != NULL)
				mxDestroyArray(mxa);

			return true;


		}


	private:

		MXFile (const MXFile& mf) : _file(0) {}
		MXFile ()                 : _file(0) {}

		MATFile*                    _file;

	};


#define mxwrite(X,Y) _mxwrite (X,Y,#X)
	template<class T> inline static bool
	_mxwrite (const NDData<T>& M, const std::string& fname, const std::string& uri) {

		MXFile mf (fname, WRITE);
		mf.Write(M, uri);
		return true;

	}


	template<class T> inline static NDData<T>
	mxread (const std::string& fname, const std::string& uri) {

		MXFile mf (fname, READ);
		return mf.Read<T>(uri);

	}

}}


#endif /* MATFILE_HPP_ */
