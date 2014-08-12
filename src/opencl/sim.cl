__constant float GAMMA = 42.57748f;
__constant float TWOPI = 6.283185307179586476925286766559005768394338798750211641949889185f;


__kernel void zerorf (__global float* rf) {
    rf[get_global_id(0)] = 0.;
}

void rotmn (const float *n, float *m, float *r) {
    
	float phi, hp, cp, sp, ar, ai, br, bi, arar, aiai, arai2, brbr,
		bibi, brbi2, arbi2, aibr2, arbr2, aibi2, brmbi, brpbi,
		armai, arpai, tmp[3];

    phi = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    
    if (phi) { /* Any rotation? */
        
        /* Cayley-Klein parameters */
        hp     =  .5*phi;        
        cp     =  cos(hp);
        sp     =  sin(hp)/phi;
        ar     =  cp;
        ai     = -n[2]*sp;
        br     =  n[1]*sp;
        bi     = -n[0]*sp;
		
        /* Speed up */
        arar   =    ar*ar;
        aiai   =    ai*ai;
        arai2  = 2.*ar*ai;
        brbr   =    br*br;
        bibi   =    bi*bi;
        brbi2  = 2.*br*bi;
        arbi2  = 2.*ar*bi;
        aibr2  = 2.*ai*br;
        arbr2  = 2.*ar*br;
        aibi2  = 2.*ai*bi;
        brmbi  = brbr - bibi;
        brpbi  = brbr + bibi;
        armai  = arar - aiai;
        arpai  = arar + aiai;
        
        /* Rotation matrix */
        r[0]   =  armai - brmbi;
        r[1]   = -arai2 - brbi2;
        r[2]   = -arbr2 + aibi2;
        r[3]   =  arai2 - brbi2; 
        r[4]   =  armai + brmbi;
        r[5]   = -aibr2 - arbi2;
        r[6]   =  arbr2 + aibi2;
        r[7]   =  arbi2 - aibr2;
        r[8]   =  arpai - brpbi;
        
        tmp[0] = r[0]*m[0] + r[3]*m[1] + r[6]*m[2];
        tmp[1] = r[1]*m[0] + r[4]*m[1] + r[7]*m[2];
        tmp[2] = r[2]*m[0] + r[5]*m[1] + r[8]*m[2];

        m[0]   = tmp[0];
        m[1]   = tmp[1];
        m[2]   = tmp[2];
        
    }

}

__kernel void simacq (const __global float* b1, const __global float*  g, const __global float* r,
                      const __global float* b0, const __global float* gs, const __global float* m0,
                      const __global float* ic, const       unsigned  nr, const       unsigned  nc,
                      const       unsigned  nk, const          float  dt,       __global float* rf) {

    unsigned pos = get_global_id(0);
    unsigned  os = pos*3;
    float    nv[3];
    float    lm[3];
    float    ls[8][2]; /* Local sensitivity */
    float   rot[9];
    float    tm[3];

    float gdt = GAMMA * TWOPI* dt;
    float rdt = 1.0e-3 * dt * TWOPI;
    float tmp[2];
    float lr[3] = {r[os]*gs[os],r[os+1]*gs[os+1],r[os+2]*gs[os+2]};

    nv[0] = 0.0;
    nv[1] = 0.0;

    lm[0] = m0[os  ]*ic[pos];
    lm[1] = m0[os+1]*ic[pos];
    lm[2] = m0[os+2]*ic[pos];

    if (lm[0] + lm[1] + lm[2] > 0.0) { // Simulate only if non-zeros voxel

        unsigned t, c, t3;

        // Local sensitivities (conj) 
        for (c = 0; c < nc; ++c) {
            unsigned b1os = 2*(pos+c*nr);
            ls[c][0] =  b1[  b1os];
            ls[c][1] =  b1[1+b1os];
        }

        // Simulate Bloch on spin 
        for (t = 0, t3 = (nk-1-t)*3; t < nk; ++t) {
            
            // lmxy before rotation
            tmp[0] = lm[0];
            tmp[1] = lm[1];
            
            // Signal acquisition only; i.e. no rf. 
            nv[2] = - gdt * (-g[  t3]*lr[0] +
                             -g[1+t3]*lr[1] +
                             -g[2+t3]*lr[2] - t * rdt * b0[pos]);
            t3 -= 3;
            // Rotate lm around nv 
            rotmn (nv, lm, rot);

            // Momentum
            tmp[0] += lm[0];
            tmp[1] += lm[1];

            unsigned st = (nk-1-t)+pos*nk*nc;
            for (c = 0; c < nc; ++c) {
                unsigned stcnk = 2*(st+c*nk);
                rf[stcnk  ] = tmp[0]*ls[c][0]+tmp[1]*ls[c][1];
                rf[stcnk+1] = tmp[0]*ls[c][0]+tmp[1]*ls[c][1];
            }
        }
    } 
	
}


//__kernel double 

__kernel void redsig (const __global float* srep, const __global float* j, 
                      const unsigned nc, const unsigned nk, const unsigned nr, 
                      __global float* rf) {
    unsigned sample = get_global_id(0);
    unsigned slen   = 2*nc*nk;
    rf[sample]    = 0.;
    for (unsigned r = 0; r < nr*slen; r += slen)
        rf[sample] += srep[r + sample];
    rf[sample]   *= j[sample%nk];
    return;
}


__kernel void intcor (__global const float* b1, const unsigned nc,
                      const unsigned nr, __global float* ic) {

    unsigned 
		pos   = get_global_id(0), 
		pos2  = pos  + pos, 
		pos21 = pos2 + 1,
		nr2   = 2*nr, pos21nr, pos2nr;

	ic[pos] = 0.;
	for (unsigned r = 0; r < nr2*nc; r += nr2) {
		pos2nr  = pos2  + r; 
		pos21nr = pos21 + r;
		ic[pos] += b1[pos2nr]*b1[pos2nr] + b1[pos21nr]*b1[pos21nr];
	}
	ic[pos] = 1.0/ic[pos];
    
}

__kernel void simexc (const __global float* b1, const __global float*  g, const __global float* rf,
                      const __global float*  r, const __global float* b0, const __global float* gs,
                      const __global float* m0, const unsigned nr, const unsigned nc, const unsigned nk,
                      const float dt, __global float* m) {


    unsigned pos = get_global_id(0);
    int     os = 3 * pos;            /* offset */
    
    float   lm[3] = {0.,0.,1.};       /* Magnetisation */
    lm[0] = m0[os  ];
    lm[1] = m0[os+1];
    lm[2] = m0[os+2];

    if (lm[0] + lm[1] + lm[2] > 0.0) { // Simulate only if non-zeros voxel


    float   nv[3];       /* Rotation axis */
    float   ls[8][2]; /* Local sensitivity */
    float  rot[9];
    float   tm[3];
    float   lr[3] = {r[os]*gs[os],r[os+1]*gs[os+1],r[os+2]*gs[os+2]};

    float  gdt = GAMMA * TWOPI * dt;
    float  rdt = 1.0e-3 * dt * TWOPI;

    unsigned t, c, t3;
    
	// Local sensitivities
	for (c = 0; c < nc; ++c) {
		int cpos = 2*(pos+c*nr);
		ls[c][0] = b1[cpos];
		ls[c][1] = b1[cpos+1];
	}
    
	for (t = 0, t3 = 0; t < nk; ++t) {
		
		// Total rf at site
		float rfsr = 0., rfsi = 0.;
        
		for (c = 0; c < nc; c++) {
			unsigned rfos = 2*(t+c*nk);
			rfsr += rf[rfos]*ls[c][0]-rf[rfos+1]*ls[c][1];
			rfsi += rf[rfos]*ls[c][1]+rf[rfos+1]*ls[c][0];
		}
        
		// Rotation vector
		nv[0] = - rdt *  rfsi;
		nv[1] =   rdt *  rfsr;
		nv[2] = - gdt * (g[t3++]*lr[0] +
						 g[t3++]*lr[1] +
						 g[t3++]*lr[2] - t*rdt*b0[pos]);
		
		// Rotate lm around nv by abs(nv) 
		rotmn (nv, lm, rot);
		
	}
	
	// Store final magnetisation for every spin 
	m[os  ] = lm[0];
	m[os+1] = lm[1];
	m[os+2] = lm[2];
    } else {
        m[os  ] = 0.;
        m[os+1] = 0.;
        m[os+2] = 0.;
    }
    
	
}
    
