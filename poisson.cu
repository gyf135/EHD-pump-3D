/*
*   The Lattice Boltzmann Method with Electrokinetic convection
*   Yifei Guan
*   Rice University
*   Apr/01/2021
*
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include "LBM.h"
#include <device_functions.h>

__global__ void gpu_efield(double*, double*, double*, double*);
__global__ void odd_extension(double*, double*, cufftDoubleComplex*);
__global__ void gpu_derivative(double*, double*, double*, cufftDoubleComplex*);
__global__ void odd_extract(double*, cufftDoubleComplex*);
__global__ void gpu_bc(double*);

__global__ void	gpu_collide_phi(double*, double*, double*, double*, double*, double*, double);
__global__ void gpu_stream_phi(double*, double*, double*);
__global__ void gpu_bc_phi(double*, double*, double*);

// =========================================================================
// Electric field solver domain extension
// =========================================================================
__host__
void efield(double *phi_gpu, double *Ex_gpu, double *Ey_gpu, double *Ez_gpu) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NZ);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	gpu_efield << < grid, threads >> > (phi_gpu, Ex_gpu, Ey_gpu, Ez_gpu);
	gpu_bc << <grid, threads >> > (Ez_gpu);
	getLastCudaError("Efield kernel error");
}

__global__ void gpu_efield(double *fi, double *ex, double *ey, double *ez){

	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int zp1 = (z + 1) % NZ;

	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	unsigned int zm1 = (NZ + z - 1) % NZ;

	ex[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(xm1,y,z)] - fi[gpu_scalar_index(xp1, y, z)]) / dx;
	ey[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, ym1, z)] - fi[gpu_scalar_index(x, yp1, z)]) / dy;
	ez[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, y, zm1)] - fi[gpu_scalar_index(x, y, zp1)]) / dz;

	// Correct boundary for electric potential phi
	//if (z == 0) {
	//	ez[gpu_scalar_index(x, y, z)] = (voltage - fi[gpu_scalar_index(x, y, zp1)]) / dz;
	//}

	//if (z == NZ - 1) {
	//	ez[gpu_scalar_index(x, y, z)] = (fi[gpu_scalar_index(x, y, zm1)] - voltage) / dz;
	//}
}
__global__ void gpu_bc(double *ez) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
		
		ez[gpu_scalar_index(x, y, 0)] = ez[gpu_scalar_index(x, y, 1)];
		return;
	}
	if (z == NZ - 1) {
		ez[gpu_scalar_index(x, y, NZ - 1)] = ez[gpu_scalar_index(x, y, NZ - 2)];
		return;
	}
}

// =========================================================================
// LBM poisson solver domain extension
// =========================================================================
__host__ void LBM_Poisson(double *phi0_gpu, double *phi1_gpu, double *phi2_gpu, double *charge_gpu, double *chargen_gpu, double *phi_gpu, double t) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NZ);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	gpu_collide_phi << < grid, threads >> >(phi0_gpu, phi1_gpu, phi2_gpu, charge_gpu, chargen_gpu, phi_gpu, t);
	gpu_stream_phi << < grid, threads >> >(phi0_gpu, phi1_gpu, phi2_gpu);
	gpu_bc_phi << < grid, threads >> >(phi0_gpu, phi1_gpu, phi2_gpu);

	// Calculate electric field strength
	efield(phi_gpu, Ex_gpu, Ey_gpu, Ez_gpu);
}


__global__ void gpu_collide_phi(double *f0, double *f1, double *f2, double *c, double *cn, double *phi, double t)
{
	// useful constants
	double omega_minus_phi = 1.0 / (beta / cs_square_phi / dt_phi + 1.0 / 2.0) / dt_phi;
	double omega_plus_phi = 1.0 / (V_Phi / (beta / cs_square_phi / dt_phi) + 1.0 / 2.0) / dt_phi;

	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    double ft0 = f0[gpu_field0_index(x, y, z)];

	double ft1 = f1[gpu_fieldn_index(x, y, z, 1)];
	double ft2 = f1[gpu_fieldn_index(x, y, z, 2)];
	double ft3 = f1[gpu_fieldn_index(x, y, z, 3)];
	double ft4 = f1[gpu_fieldn_index(x, y, z, 4)];
	double ft5 = f1[gpu_fieldn_index(x, y, z, 5)];
	double ft6 = f1[gpu_fieldn_index(x, y, z, 6)];
	double ft7 = f1[gpu_fieldn_index(x, y, z, 7)];
	double ft8 = f1[gpu_fieldn_index(x, y, z, 8)];
	double ft9 = f1[gpu_fieldn_index(x, y, z, 9)];
	double ft10 = f1[gpu_fieldn_index(x, y, z, 10)];
	double ft11 = f1[gpu_fieldn_index(x, y, z, 11)];
	double ft12 = f1[gpu_fieldn_index(x, y, z, 12)];
	double ft13 = f1[gpu_fieldn_index(x, y, z, 13)];
	double ft14 = f1[gpu_fieldn_index(x, y, z, 14)];
	double ft15 = f1[gpu_fieldn_index(x, y, z, 15)];
	double ft16 = f1[gpu_fieldn_index(x, y, z, 16)];
	double ft17 = f1[gpu_fieldn_index(x, y, z, 17)];
	double ft18 = f1[gpu_fieldn_index(x, y, z, 18)];
	double ft19 = f1[gpu_fieldn_index(x, y, z, 19)];
	double ft20 = f1[gpu_fieldn_index(x, y, z, 20)];
	double ft21 = f1[gpu_fieldn_index(x, y, z, 21)];
	double ft22 = f1[gpu_fieldn_index(x, y, z, 22)];
	double ft23 = f1[gpu_fieldn_index(x, y, z, 23)];
	double ft24 = f1[gpu_fieldn_index(x, y, z, 24)];
	double ft25 = f1[gpu_fieldn_index(x, y, z, 25)];
	double ft26 = f1[gpu_fieldn_index(x, y, z, 26)];

	// compute macroscopic variables from microscopic variables
	double force = beta*convertCtoCharge*(c[gpu_scalar_index(x, y, z)] - cn[gpu_scalar_index(x, y, z)]) / eps;
	double p = ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8
		+ ft9 + ft10 + ft11 + ft12 + ft13 + ft14 + ft15 + ft16 + ft17 + ft18 + ft19 + ft20 + ft21 + ft22 + ft23 + ft24 + ft25 + ft26 + force*dt_phi*0.5;
	// write to memory (only when visualizing the data)

	phi[gpu_scalar_index(x, y, z)] = p;

	// collision step
	// now compute and relax to equilibrium

	// calculate equilibrium
	// temporary variables
	double w0p = w0*p;
	double wsp = ws*p;
	double wap = wa*p;
	double wdp = wd*p;

	// zero weight
	double fe0 = w0p;

	// adjacent weight
	// phi

	double fe1 = wsp;
	double fe2 = wsp;
	double fe3 = wsp;
	double fe4 = wsp;
	double fe5 = wsp;
	double fe6 = wsp;

	// diagonal weight
	// phi
	double fe7 = wap;
	double fe8 = wap;
	double fe9 = wap;
	double fe10 = wap;
	double fe11 = wap;
	double fe12 = wap;
	double fe13 = wap;
	double fe14 = wap;
	double fe15 = wap;
	double fe16 = wap;
	double fe17 = wap;
	double fe18 = wap;

	// 3d diagonal

	double fe19 = wdp;
	double fe20 = wdp;
	double fe21 = wdp;
	double fe22 = wdp;
	double fe23 = wdp;
	double fe24 = wdp;
	double fe25 = wdp;
	double fe26 = wdp;

	// calculate force population
	// temperory variables
	double coe0 = w0 * force;
	double coes = ws * force;
	double coea = wa * force;
	double coed = wd * force;

	double fpop0 = coe0;

	double fpop1 = coes; 
	double fpop2 = coes;
	double fpop3 = coes;
	double fpop4 = coes;
	double fpop5 = coes;
	double fpop6 = coes;

	double fpop7 = coea;
	double fpop8 = coea;
	double fpop9 = coea;
	double fpop10 = coea;
	double fpop11 = coea;
	double fpop12 = coea;

	double fpop13 = coea;
	double fpop14 = coea;
	double fpop15 = coea;
	double fpop16 = coea;
	double fpop17 = coea;
	double fpop18 = coea;

	double fpop19 = coed;
	double fpop20 = coed;
	double fpop21 = coed;
	double fpop22 = coed;
	double fpop23 = coed;
	double fpop24 = coed;
	double fpop25 = coed;
	double fpop26 = coed;

	// calculate f1 plus and minus
	double fp0 = ft0;
	double fp1 = 0.5*(ft1 + ft2);
	double fp2 = fp1;
	double fp3 = 0.5*(ft3 + ft4);
	double fp4 = fp3;
	double fp5 = 0.5*(ft5 + ft6);
	double fp6 = fp5;
	double fp7 = 0.5*(ft7 + ft8);
	double fp8 = fp7;
	double fp9 = 0.5*(ft9 + ft10);
	double fp10 = fp9;
	double fp11 = 0.5*(ft11 + ft12);
	double fp12 = fp11;
	double fp13 = 0.5*(ft13 + ft14);
	double fp14 = fp13;
	double fp15 = 0.5*(ft15 + ft16);
	double fp16 = fp15;
	double fp17 = 0.5*(ft17 + ft18);
	double fp18 = fp17;
	double fp19 = 0.5*(ft19 + ft20);
	double fp20 = fp19;
	double fp21 = 0.5*(ft21 + ft22);
	double fp22 = fp21;
	double fp23 = 0.5*(ft23 + ft24);
	double fp24 = fp23;
	double fp25 = 0.5*(ft25 + ft26);
	double fp26 = fp25;

	double fm0 = 0.0;
	double fm1 = 0.5*(ft1 - ft2);
	double fm2 = -fm1;
	double fm3 = 0.5*(ft3 - ft4);
	double fm4 = -fm3;
	double fm5 = 0.5*(ft5 - ft6);
	double fm6 = -fm5;
	double fm7 = 0.5*(ft7 - ft8);
	double fm8 = -fm7;
	double fm9 = 0.5*(ft9 - ft10);
	double fm10 = -fm9;
	double fm11 = 0.5*(ft11 - ft12);
	double fm12 = -fm11;
	double fm13 = 0.5*(ft13 - ft14);
	double fm14 = -fm13;
	double fm15 = 0.5*(ft15 - ft16);
	double fm16 = -fm15;
	double fm17 = 0.5*(ft17 - ft18);
	double fm18 = -fm17;
	double fm19 = 0.5*(ft19 - ft20);
	double fm20 = -fm19;
	double fm21 = 0.5*(ft21 - ft22);
	double fm22 = -fm21;
	double fm23 = 0.5*(ft23 - ft24);
	double fm24 = -fm23;
	double fm25 = 0.5*(ft25 - ft26);
	double fm26 = -fm25;

	// calculate feq plus and minus
	double fep0 = fe0;
	double fep1 = 0.5*(fe1 + fe2);
	double fep2 = fep1;
	double fep3 = 0.5*(fe3 + fe4);
	double fep4 = fep3;
	double fep5 = 0.5*(fe5 + fe6);
	double fep6 = fep5;
	double fep7 = 0.5*(fe7 + fe8);
	double fep8 = fep7;
	double fep9 = 0.5*(fe9 + fe10);
	double fep10 = fep9;
	double fep11 = 0.5*(fe11 + fe12);
	double fep12 = fep11;
	double fep13 = 0.5*(fe13 + fe14);
	double fep14 = fep13;
	double fep15 = 0.5*(fe15 + fe16);
	double fep16 = fep15;
	double fep17 = 0.5*(fe17 + fe18);
	double fep18 = fep17;
	double fep19 = 0.5*(fe19 + fe20);
	double fep20 = fep19;
	double fep21 = 0.5*(fe21 + fe22);
	double fep22 = fep21;
	double fep23 = 0.5*(fe23 + fe24);
	double fep24 = fep23;
	double fep25 = 0.5*(fe25 + fe26);
	double fep26 = fep25;

	double fem0 = 0.0;
	double fem1 = 0.5*(fe1 - fe2);
	double fem2 = -fem1;
	double fem3 = 0.5*(fe3 - fe4);
	double fem4 = -fem3;
	double fem5 = 0.5*(fe5 - fe6);
	double fem6 = -fem5;
	double fem7 = 0.5*(fe7 - fe8);
	double fem8 = -fem7;
	double fem9 = 0.5*(fe9 - fe10);
	double fem10 = -fem9;
	double fem11 = 0.5*(fe11 - fe12);
	double fem12 = -fem11;
	double fem13 = 0.5*(fe13 - fe14);
	double fem14 = -fem13;
	double fem15 = 0.5*(fe15 - fe16);
	double fem16 = -fem15;
	double fem17 = 0.5*(fe17 - fe18);
	double fem18 = -fem17;
	double fem19 = 0.5*(fe19 - fe20);
	double fem20 = -fem19;
	double fem21 = 0.5*(fe21 - fe22);
	double fem22 = -fem21;
	double fem23 = 0.5*(fe23 - fe24);
	double fem24 = -fem23;
	double fem25 = 0.5*(fe25 - fe26);
	double fem26 = -fem25;

	// calculate force_plus and force_minus
	double forcep0 = fpop0;
	double forcep1 = 0.5*(fpop1 + fpop2);
	double forcep2 = forcep1;
	double forcep3 = 0.5*(fpop3 + fpop4);
	double forcep4 = forcep3;
	double forcep5 = 0.5*(fpop5 + fpop6);
	double forcep6 = forcep5;
	double forcep7 = 0.5*(fpop7 + fpop8);
	double forcep8 = forcep7;
	double forcep9 = 0.5*(fpop9 + fpop10);
	double forcep10 = forcep9;
	double forcep11 = 0.5*(fpop11 + fpop12);
	double forcep12 = forcep11;
	double forcep13 = 0.5*(fpop13 + fpop14);
	double forcep14 = forcep13;
	double forcep15 = 0.5*(fpop15 + fpop16);
	double forcep16 = forcep15;
	double forcep17 = 0.5*(fpop17 + fpop18);
	double forcep18 = forcep17;
	double forcep19 = 0.5*(fpop19 + fpop20);
	double forcep20 = forcep19;
	double forcep21 = 0.5*(fpop21 + fpop22);
	double forcep22 = forcep21;
	double forcep23 = 0.5*(fpop23 + fpop24);
	double forcep24 = forcep23;
	double forcep25 = 0.5*(fpop25 + fpop26);
	double forcep26 = forcep25;

	double forcem0 = 0.0;
	double forcem1 = 0.5*(fpop1 - fpop2);
	double forcem2 = -forcem1;
	double forcem3 = 0.5*(fpop3 - fpop4);
	double forcem4 = -forcem3;
	double forcem5 = 0.5*(fpop5 - fpop6);
	double forcem6 = -forcem5;
	double forcem7 = 0.5*(fpop7 - fpop8);
	double forcem8 = -forcem7;
	double forcem9 = 0.5*(fpop9 - fpop10);
	double forcem10 = -forcem9;
	double forcem11 = 0.5*(fpop11 - fpop12);
	double forcem12 = -forcem11;
	double forcem13 = 0.5*(fpop13 - fpop14);
	double forcem14 = -forcem13;
	double forcem15 = 0.5*(fpop15 - fpop16);
	double forcem16 = -forcem15;
	double forcem17 = 0.5*(fpop17 - fpop18);
	double forcem18 = -forcem17;
	double forcem19 = 0.5*(fpop19 - fpop20);
	double forcem20 = -forcem19;
	double forcem21 = 0.5*(fpop21 - fpop22);
	double forcem22 = -forcem21;
	double forcem23 = 0.5*(fpop23 - fpop24);
	double forcem24 = -forcem23;
	double forcem25 = 0.5*(fpop25 - fpop26);
	double forcem26 = -forcem25;

	double sp = 1.0 - 0.5*dt_phi*omega_plus_phi;
	double sm = 1.0 - 0.5*dt_phi*omega_minus_phi;

	double source0 = sp*fpop0;
	double source1 = sp*forcep1 + sm*forcem1;
	double source2 = sp*forcep2 + sm*forcem2;
	double source3 = sp*forcep3 + sm*forcem3;
	double source4 = sp*forcep4 + sm*forcem4;
	double source5 = sp*forcep5 + sm*forcem5;
	double source6 = sp*forcep6 + sm*forcem6;
	double source7 = sp*forcep7 + sm*forcem7;
	double source8 = sp*forcep8 + sm*forcem8;
	double source9 = sp*forcep9 + sm*forcem9;
	double source10 = sp*forcep10 + sm*forcem10;
	double source11 = sp*forcep11 + sm*forcem11;
	double source12 = sp*forcep12 + sm*forcem12;
	double source13 = sp*forcep13 + sm*forcem13;
	double source14 = sp*forcep14 + sm*forcem14;
	double source15 = sp*forcep15 + sm*forcem15;
	double source16 = sp*forcep16 + sm*forcem16;
	double source17 = sp*forcep17 + sm*forcem17;
	double source18 = sp*forcep18 + sm*forcem18;
	double source19 = sp*forcep19 + sm*forcem19;
	double source20 = sp*forcep20 + sm*forcem20;
	double source21 = sp*forcep21 + sm*forcem21;
	double source22 = sp*forcep22 + sm*forcem22;
	double source23 = sp*forcep23 + sm*forcem23;
	double source24 = sp*forcep24 + sm*forcem24;
	double source25 = sp*forcep25 + sm*forcem25;
	double source26 = sp*forcep26 + sm*forcem26;
	// ===============================================================

	// ===============================================================
	// ===============================================================
	// temporary variables (relaxation times)
	double tw0rp = omega_plus_phi*dt_phi;  //   omega_plus*dt 
	double tw0rm = omega_minus_phi*dt_phi; //   omega_minus*dt 

	// TRT collision operations

	f0[gpu_field0_index(x, y, z)] = ft0 - (tw0rp * (fp0 - fep0) + tw0rm * (fm0 - fem0)) + dt_phi*source0;
	f2[gpu_fieldn_index(x, y, z, 1)] = ft1 - (tw0rp * (fp1 - fep1) + tw0rm * (fm1 - fem1)) + dt_phi*source1;
	f2[gpu_fieldn_index(x, y, z, 2)] = ft2 - (tw0rp * (fp2 - fep2) + tw0rm * (fm2 - fem2)) + dt_phi*source2;
	f2[gpu_fieldn_index(x, y, z, 3)] = ft3 - (tw0rp * (fp3 - fep3) + tw0rm * (fm3 - fem3)) + dt_phi*source3;
	f2[gpu_fieldn_index(x, y, z, 4)] = ft4 - (tw0rp * (fp4 - fep4) + tw0rm * (fm4 - fem4)) + dt_phi*source4;
	f2[gpu_fieldn_index(x, y, z, 5)] = ft5 - (tw0rp * (fp5 - fep5) + tw0rm * (fm5 - fem5)) + dt_phi*source5;
	f2[gpu_fieldn_index(x, y, z, 6)] = ft6 - (tw0rp * (fp6 - fep6) + tw0rm * (fm6 - fem6)) + dt_phi*source6;
	f2[gpu_fieldn_index(x, y, z, 7)] = ft7 - (tw0rp * (fp7 - fep7) + tw0rm * (fm7 - fem7)) + dt_phi*source7;
	f2[gpu_fieldn_index(x, y, z, 8)] = ft8 - (tw0rp * (fp8 - fep8) + tw0rm * (fm8 - fem8)) + dt_phi*source8;
	f2[gpu_fieldn_index(x, y, z, 9)] = ft9 - (tw0rp * (fp9 - fep9) + tw0rm * (fm9 - fem9)) + dt_phi*source9;
	f2[gpu_fieldn_index(x, y, z, 10)] = ft10 - (tw0rp * (fp10 - fep10) + tw0rm * (fm10 - fem10)) + dt_phi*source10;
	f2[gpu_fieldn_index(x, y, z, 11)] = ft11 - (tw0rp * (fp11 - fep11) + tw0rm * (fm11 - fem11)) + dt_phi*source11;
	f2[gpu_fieldn_index(x, y, z, 12)] = ft12 - (tw0rp * (fp12 - fep12) + tw0rm * (fm12 - fem12)) + dt_phi*source12;
	f2[gpu_fieldn_index(x, y, z, 13)] = ft13 - (tw0rp * (fp13 - fep13) + tw0rm * (fm13 - fem13)) + dt_phi*source13;
	f2[gpu_fieldn_index(x, y, z, 14)] = ft14 - (tw0rp * (fp14 - fep14) + tw0rm * (fm14 - fem14)) + dt_phi*source14;
	f2[gpu_fieldn_index(x, y, z, 15)] = ft15 - (tw0rp * (fp15 - fep15) + tw0rm * (fm15 - fem15)) + dt_phi*source15;
	f2[gpu_fieldn_index(x, y, z, 16)] = ft16 - (tw0rp * (fp16 - fep16) + tw0rm * (fm16 - fem16)) + dt_phi*source16;
	f2[gpu_fieldn_index(x, y, z, 17)] = ft17 - (tw0rp * (fp17 - fep17) + tw0rm * (fm17 - fem17)) + dt_phi*source17;
	f2[gpu_fieldn_index(x, y, z, 18)] = ft18 - (tw0rp * (fp18 - fep18) + tw0rm * (fm18 - fem18)) + dt_phi*source18;
	f2[gpu_fieldn_index(x, y, z, 19)] = ft19 - (tw0rp * (fp19 - fep19) + tw0rm * (fm19 - fem19)) + dt_phi*source19;
	f2[gpu_fieldn_index(x, y, z, 20)] = ft20 - (tw0rp * (fp20 - fep20) + tw0rm * (fm20 - fem20)) + dt_phi*source20;
	f2[gpu_fieldn_index(x, y, z, 21)] = ft21 - (tw0rp * (fp21 - fep21) + tw0rm * (fm21 - fem21)) + dt_phi*source21;
	f2[gpu_fieldn_index(x, y, z, 22)] = ft22 - (tw0rp * (fp22 - fep22) + tw0rm * (fm22 - fem22)) + dt_phi*source22;
	f2[gpu_fieldn_index(x, y, z, 23)] = ft23 - (tw0rp * (fp23 - fep23) + tw0rm * (fm23 - fem23)) + dt_phi*source23;
	f2[gpu_fieldn_index(x, y, z, 24)] = ft24 - (tw0rp * (fp24 - fep24) + tw0rm * (fm24 - fem24)) + dt_phi*source24;
	f2[gpu_fieldn_index(x, y, z, 25)] = ft25 - (tw0rp * (fp25 - fep25) + tw0rm * (fm25 - fem25)) + dt_phi*source25;
	f2[gpu_fieldn_index(x, y, z, 26)] = ft26 - (tw0rp * (fp26 - fep26) + tw0rm * (fm26 - fem26)) + dt_phi*source26;
}

__global__ void gpu_stream_phi(double *f0, double *f1, double *f2)
{
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// streaming step
	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int zp1 = (z + 1) % NZ;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	unsigned int zm1 = (NZ + z - 1) % NZ;
	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8

	// load populations from adjacent nodes (ft is post-streaming population of f1)
	// flows
	f1[gpu_fieldn_index(x, y, z, 1)] = f2[gpu_fieldn_index(xm1, y, z, 1)];
	f1[gpu_fieldn_index(x, y, z, 2)] = f2[gpu_fieldn_index(xp1, y, z, 2)];
	f1[gpu_fieldn_index(x, y, z, 3)] = f2[gpu_fieldn_index(x, ym1, z, 3)];
	f1[gpu_fieldn_index(x, y, z, 4)] = f2[gpu_fieldn_index(x, yp1, z, 4)];
	f1[gpu_fieldn_index(x, y, z, 5)] = f2[gpu_fieldn_index(x, y, zm1, 5)];
	f1[gpu_fieldn_index(x, y, z, 6)] = f2[gpu_fieldn_index(x, y, zp1, 6)];
	f1[gpu_fieldn_index(x, y, z, 7)] = f2[gpu_fieldn_index(xm1, ym1, z, 7)];
	f1[gpu_fieldn_index(x, y, z, 8)] = f2[gpu_fieldn_index(xp1, yp1, z, 8)];
	f1[gpu_fieldn_index(x, y, z, 9)] = f2[gpu_fieldn_index(xm1, y, zm1, 9)];
	f1[gpu_fieldn_index(x, y, z, 10)] = f2[gpu_fieldn_index(xp1, y, zp1, 10)];
	f1[gpu_fieldn_index(x, y, z, 11)] = f2[gpu_fieldn_index(x, ym1, zm1, 11)];
	f1[gpu_fieldn_index(x, y, z, 12)] = f2[gpu_fieldn_index(x, yp1, zp1, 12)];
	f1[gpu_fieldn_index(x, y, z, 13)] = f2[gpu_fieldn_index(xm1, yp1, z, 13)];
	f1[gpu_fieldn_index(x, y, z, 14)] = f2[gpu_fieldn_index(xp1, ym1, z, 14)];
	f1[gpu_fieldn_index(x, y, z, 15)] = f2[gpu_fieldn_index(xm1, y, zp1, 15)];
	f1[gpu_fieldn_index(x, y, z, 16)] = f2[gpu_fieldn_index(xp1, y, zm1, 16)];
	f1[gpu_fieldn_index(x, y, z, 17)] = f2[gpu_fieldn_index(x, ym1, zp1, 17)];
	f1[gpu_fieldn_index(x, y, z, 18)] = f2[gpu_fieldn_index(x, yp1, zm1, 18)];
	f1[gpu_fieldn_index(x, y, z, 19)] = f2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
	f1[gpu_fieldn_index(x, y, z, 20)] = f2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
	f1[gpu_fieldn_index(x, y, z, 21)] = f2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
	f1[gpu_fieldn_index(x, y, z, 22)] = f2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
	f1[gpu_fieldn_index(x, y, z, 23)] = f2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
	f1[gpu_fieldn_index(x, y, z, 24)] = f2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
	f1[gpu_fieldn_index(x, y, z, 25)] = f2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
	f1[gpu_fieldn_index(x, y, z, 26)] = f2[gpu_fieldn_index(xm1, yp1, zp1, 26)];
}


__global__ void gpu_bc_phi(double *temp0, double *temp1, double *temp2)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;

	// No-flux boundary conditions as in Yoshida - 2014 - Coupled lattice Boltzmann method for simulating electrokinetic flows - a localized scheme for the Nernst-Planck model
	/*if (z == 0 || z == NZ - 1) {
		// positive charge
		double ht1 = h2[gpu_fieldn_index(x, y, z, 1)];
		double ht2 = h2[gpu_fieldn_index(x, y, z, 2)];
		double ht3 = h2[gpu_fieldn_index(x, y, z, 3)];
		double ht4 = h2[gpu_fieldn_index(x, y, z, 4)];
		double ht5 = h2[gpu_fieldn_index(x, y, z, 5)];
		double ht6 = h2[gpu_fieldn_index(x, y, z, 6)];
		double ht7 = h2[gpu_fieldn_index(x, y, z, 7)];
		double ht8 = h2[gpu_fieldn_index(x, y, z, 8)];
		double ht9 = h2[gpu_fieldn_index(x, y, z, 9)];
		double ht10 = h2[gpu_fieldn_index(x, y, z, 10)];
		double ht11 = h2[gpu_fieldn_index(x, y, z, 11)];
		double ht12 = h2[gpu_fieldn_index(x, y, z, 12)];
		double ht13 = h2[gpu_fieldn_index(x, y, z, 13)];
		double ht14 = h2[gpu_fieldn_index(x, y, z, 14)];
		double ht15 = h2[gpu_fieldn_index(x, y, z, 15)];
		double ht16 = h2[gpu_fieldn_index(x, y, z, 16)];
		double ht17 = h2[gpu_fieldn_index(x, y, z, 17)];
		double ht18 = h2[gpu_fieldn_index(x, y, z, 18)];
		double ht19 = h2[gpu_fieldn_index(x, y, z, 19)];
		double ht20 = h2[gpu_fieldn_index(x, y, z, 20)];
		double ht21 = h2[gpu_fieldn_index(x, y, z, 21)];
		double ht22 = h2[gpu_fieldn_index(x, y, z, 22)];
		double ht23 = h2[gpu_fieldn_index(x, y, z, 23)];
		double ht24 = h2[gpu_fieldn_index(x, y, z, 24)];
		double ht25 = h2[gpu_fieldn_index(x, y, z, 25)];
		double ht26 = h2[gpu_fieldn_index(x, y, z, 26)];

		h0[gpu_field0_index(x, y, z)] = h0[gpu_field0_index(x, y, z)];
		h1[gpu_fieldn_index(x, y, z, 1)] = ht2;
		h1[gpu_fieldn_index(x, y, z, 2)] = ht1;
		h1[gpu_fieldn_index(x, y, z, 3)] = ht4;
		h1[gpu_fieldn_index(x, y, z, 4)] = ht3;
		h1[gpu_fieldn_index(x, y, z, 5)] = ht6;
		h1[gpu_fieldn_index(x, y, z, 6)] = ht5;

		h1[gpu_fieldn_index(x, y, z, 7)] = ht8;
		h1[gpu_fieldn_index(x, y, z, 8)] = ht7;
		h1[gpu_fieldn_index(x, y, z, 9)] = ht10;
		h1[gpu_fieldn_index(x, y, z, 10)] = ht9;
		h1[gpu_fieldn_index(x, y, z, 11)] = ht12;
		h1[gpu_fieldn_index(x, y, z, 12)] = ht11;
		h1[gpu_fieldn_index(x, y, z, 13)] = ht14;
		h1[gpu_fieldn_index(x, y, z, 14)] = ht13;
		h1[gpu_fieldn_index(x, y, z, 15)] = ht16;
		h1[gpu_fieldn_index(x, y, z, 16)] = ht15;
		h1[gpu_fieldn_index(x, y, z, 17)] = ht18;
		h1[gpu_fieldn_index(x, y, z, 18)] = ht17;

		h1[gpu_fieldn_index(x, y, z, 19)] = ht20;
		h1[gpu_fieldn_index(x, y, z, 20)] = ht19;
		h1[gpu_fieldn_index(x, y, z, 21)] = ht22;
		h1[gpu_fieldn_index(x, y, z, 22)] = ht21;
		h1[gpu_fieldn_index(x, y, z, 23)] = ht24;
		h1[gpu_fieldn_index(x, y, z, 24)] = ht23;
		h1[gpu_fieldn_index(x, y, z, 25)] = ht26;
		h1[gpu_fieldn_index(x, y, z, 26)] = ht25;
	}*/

	// Bounce-back boundary conditions for Dirichlet boundaries
	//double multi0T = 2.0*voltage*w0;
	//double multisT = 2.0*voltage*ws;
	//double multiaT = 2.0*voltage*wa;
	//double multidT = 2.0*voltage*wd;
	//if (z == 0) {
	//	// lower plate for constant phi
	//	double tempt1 = temp2[gpu_fieldn_index(x, y, 0, 1)];
	//	double tempt2 = temp2[gpu_fieldn_index(x, y, 0, 2)];
	//	double tempt3 = temp2[gpu_fieldn_index(x, y, 0, 3)];
	//	double tempt4 = temp2[gpu_fieldn_index(x, y, 0, 4)];
	//	double tempt5 = temp2[gpu_fieldn_index(x, y, 0, 5)];
	//	double tempt6 = temp2[gpu_fieldn_index(x, y, 0, 6)];
	//	double tempt7 = temp2[gpu_fieldn_index(x, y, 0, 7)];
	//	double tempt8 = temp2[gpu_fieldn_index(x, y, 0, 8)];
	//	double tempt9 = temp2[gpu_fieldn_index(x, y, 0, 9)];
	//	double tempt10 = temp2[gpu_fieldn_index(x, y, 0, 10)];
	//	double tempt11 = temp2[gpu_fieldn_index(x, y, 0, 11)];
	//	double tempt12 = temp2[gpu_fieldn_index(x, y, 0, 12)];
	//	double tempt13 = temp2[gpu_fieldn_index(x, y, 0, 13)];
	//	double tempt14 = temp2[gpu_fieldn_index(x, y, 0, 14)];
	//	double tempt15 = temp2[gpu_fieldn_index(x, y, 0, 15)];
	//	double tempt16 = temp2[gpu_fieldn_index(x, y, 0, 16)];
	//	double tempt17 = temp2[gpu_fieldn_index(x, y, 0, 17)];
	//	double tempt18 = temp2[gpu_fieldn_index(x, y, 0, 18)];
	//	double tempt19 = temp2[gpu_fieldn_index(x, y, 0, 19)];
	//	double tempt20 = temp2[gpu_fieldn_index(x, y, 0, 20)];
	//	double tempt21 = temp2[gpu_fieldn_index(x, y, 0, 21)];
	//	double tempt22 = temp2[gpu_fieldn_index(x, y, 0, 22)];
	//	double tempt23 = temp2[gpu_fieldn_index(x, y, 0, 23)];
	//	double tempt24 = temp2[gpu_fieldn_index(x, y, 0, 24)];
	//	double tempt25 = temp2[gpu_fieldn_index(x, y, 0, 25)];
	//	double tempt26 = temp2[gpu_fieldn_index(x, y, 0, 26)];

	//	temp0[gpu_field0_index(x, y, 0)] = -temp0[gpu_field0_index(x, y, 0)] + multi0T;
	//	temp1[gpu_fieldn_index(x, y, 0, 1)] = -tempt2 + multisT;
	//	temp1[gpu_fieldn_index(x, y, 0, 2)] = -tempt1 + multisT;
	//	temp1[gpu_fieldn_index(x, y, 0, 3)] = -tempt4 + multisT;
	//	temp1[gpu_fieldn_index(x, y, 0, 4)] = -tempt3 + multisT;
	//	temp1[gpu_fieldn_index(x, y, 0, 5)] = -tempt6 + multisT;
	//	temp1[gpu_fieldn_index(x, y, 0, 6)] = -tempt5 + multisT;

	//	temp1[gpu_fieldn_index(x, y, 0, 7)] = -tempt8 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 8)] = -tempt7 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 9)] = -tempt10 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 10)] = -tempt9 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 11)] = -tempt12 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 12)] = -tempt11 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 13)] = -tempt14 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 14)] = -tempt13 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 15)] = -tempt16 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 16)] = -tempt15 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 17)] = -tempt18 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, 0, 18)] = -tempt17 + multiaT;

	//	temp1[gpu_fieldn_index(x, y, 0, 19)] = -tempt20 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 20)] = -tempt19 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 21)] = -tempt22 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 22)] = -tempt21 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 23)] = -tempt24 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 24)] = -tempt23 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 25)] = -tempt26 + multidT;
	//	temp1[gpu_fieldn_index(x, y, 0, 26)] = -tempt25 + multidT;
	//}

	//if (z == NZ - 1) {

	//	// upper plate constant phi
	//	double tempt1 = temp2[gpu_fieldn_index(x, y, z, 1)];
	//	double tempt2 = temp2[gpu_fieldn_index(x, y, z, 2)];
	//	double tempt3 = temp2[gpu_fieldn_index(x, y, z, 3)];
	//	double tempt4 = temp2[gpu_fieldn_index(x, y, z, 4)];
	//	double tempt5 = temp2[gpu_fieldn_index(x, y, z, 5)];
	//	double tempt6 = temp2[gpu_fieldn_index(x, y, z, 6)];
	//	double tempt7 = temp2[gpu_fieldn_index(x, y, z, 7)];
	//	double tempt8 = temp2[gpu_fieldn_index(x, y, z, 8)];
	//	double tempt9 = temp2[gpu_fieldn_index(x, y, z, 9)];
	//	double tempt10 = temp2[gpu_fieldn_index(x, y, z, 10)];
	//	double tempt11 = temp2[gpu_fieldn_index(x, y, z, 11)];
	//	double tempt12 = temp2[gpu_fieldn_index(x, y, z, 12)];
	//	double tempt13 = temp2[gpu_fieldn_index(x, y, z, 13)];
	//	double tempt14 = temp2[gpu_fieldn_index(x, y, z, 14)];
	//	double tempt15 = temp2[gpu_fieldn_index(x, y, z, 15)];
	//	double tempt16 = temp2[gpu_fieldn_index(x, y, z, 16)];
	//	double tempt17 = temp2[gpu_fieldn_index(x, y, z, 17)];
	//	double tempt18 = temp2[gpu_fieldn_index(x, y, z, 18)];
	//	double tempt19 = temp2[gpu_fieldn_index(x, y, z, 19)];
	//	double tempt20 = temp2[gpu_fieldn_index(x, y, z, 20)];
	//	double tempt21 = temp2[gpu_fieldn_index(x, y, z, 21)];
	//	double tempt22 = temp2[gpu_fieldn_index(x, y, z, 22)];
	//	double tempt23 = temp2[gpu_fieldn_index(x, y, z, 23)];
	//	double tempt24 = temp2[gpu_fieldn_index(x, y, z, 24)];
	//	double tempt25 = temp2[gpu_fieldn_index(x, y, z, 25)];
	//	double tempt26 = temp2[gpu_fieldn_index(x, y, z, 26)];

	//	temp0[gpu_field0_index(x, y, z)] = -temp0[gpu_field0_index(x, y, z)] + multi0T;
	//	temp1[gpu_fieldn_index(x, y, z, 1)] = -tempt2 + multisT;
	//	temp1[gpu_fieldn_index(x, y, z, 2)] = -tempt1 + multisT;
	//	temp1[gpu_fieldn_index(x, y, z, 3)] = -tempt4 + multisT;
	//	temp1[gpu_fieldn_index(x, y, z, 4)] = -tempt3 + multisT;
	//	temp1[gpu_fieldn_index(x, y, z, 5)] = -tempt6 + multisT;
	//	temp1[gpu_fieldn_index(x, y, z, 6)] = -tempt5 + multisT;

	//	temp1[gpu_fieldn_index(x, y, z, 7)] = -tempt8 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 8)] = -tempt7 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 9)] = -tempt10 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 10)] = -tempt9 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 11)] = -tempt12 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 12)] = -tempt11 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 13)] = -tempt14 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 14)] = -tempt13 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 15)] = -tempt16 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 16)] = -tempt15 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 17)] = -tempt18 + multiaT;
	//	temp1[gpu_fieldn_index(x, y, z, 18)] = -tempt17 + multiaT;

	//	temp1[gpu_fieldn_index(x, y, z, 19)] = -tempt20 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 20)] = -tempt19 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 21)] = -tempt22 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 22)] = -tempt21 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 23)] = -tempt24 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 24)] = -tempt23 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 25)] = -tempt26 + multidT;
	//	temp1[gpu_fieldn_index(x, y, z, 26)] = -tempt25 + multidT;
	//}
	
	double tempt0 = temp0[gpu_field0_index(x, y, z)];
	double tempt1 = temp1[gpu_fieldn_index(x, y, z, 1)];
	double tempt2 = temp1[gpu_fieldn_index(x, y, z, 2)];
	double tempt3 = temp1[gpu_fieldn_index(x, y, z, 3)];
	double tempt4 = temp1[gpu_fieldn_index(x, y, z, 4)];
	double tempt5 = temp1[gpu_fieldn_index(x, y, z, 5)];
	double tempt6 = temp1[gpu_fieldn_index(x, y, z, 6)];
	double tempt7 = temp1[gpu_fieldn_index(x, y, z, 7)];
	double tempt8 = temp1[gpu_fieldn_index(x, y, z, 8)];
	double tempt9 = temp1[gpu_fieldn_index(x, y, z, 9)];
	double tempt10 = temp1[gpu_fieldn_index(x, y, z, 10)];
	double tempt11 = temp1[gpu_fieldn_index(x, y, z, 11)];
	double tempt12 = temp1[gpu_fieldn_index(x, y, z, 12)];
	double tempt13 = temp1[gpu_fieldn_index(x, y, z, 13)];
	double tempt14 = temp1[gpu_fieldn_index(x, y, z, 14)];
	double tempt15 = temp1[gpu_fieldn_index(x, y, z, 15)];
	double tempt16 = temp1[gpu_fieldn_index(x, y, z, 16)];
	double tempt17 = temp1[gpu_fieldn_index(x, y, z, 17)];
	double tempt18 = temp1[gpu_fieldn_index(x, y, z, 18)];
	double tempt19 = temp1[gpu_fieldn_index(x, y, z, 19)];
	double tempt20 = temp1[gpu_fieldn_index(x, y, z, 20)];
	double tempt21 = temp1[gpu_fieldn_index(x, y, z, 21)];
	double tempt22 = temp1[gpu_fieldn_index(x, y, z, 22)];
	double tempt23 = temp1[gpu_fieldn_index(x, y, z, 23)];
	double tempt24 = temp1[gpu_fieldn_index(x, y, z, 24)];
	double tempt25 = temp1[gpu_fieldn_index(x, y, z, 25)];
	double tempt26 = temp1[gpu_fieldn_index(x, y, z, 26)];

	//__constant__ double w0 = 8.0 / 27.0;  // zero weight for i=0
	//__constant__ double ws = 2.0 / 27.0;  // adjacent weight for i=1-6
	//__constant__ double wa = 1.0 / 54.0;  // adjacent weight for i=7-18
	//__constant__ double wd = 1.0 / 216.0; // diagonal weight for i=19-26

	// Wet-node boundary for Dirichlet
	//// z = 0 (lower boundary n = (0,0,1))
 //   // c \dot n = 0: 0, 1, 2, 3, 4; 7, 8, 13, 14
	//// c \dot n < 0: 6; 10, 12, 15, 17; 20, 21, 24, 26
	//// c \dot n > 0: 5; 9, 11, 16, 18; 19, 22, 23, 25
	if ((z == 0) && (x>=cathodeL && x<=cathodeR)) { // cathode grounded
		double cPrime = (0 - (tempt0 + tempt1 + tempt2 + tempt3 + tempt4 + tempt7 + tempt8 + tempt13 + tempt14 + tempt6 + tempt10 + tempt12
			+ tempt15 + tempt17 + tempt20 + tempt21 + tempt24 + tempt26)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 5)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 9)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 11)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 16)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 18)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 19)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 22)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 23)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 25)] = wd*cPrime;
	}

	if ((z == 0) && (x >= anodeL && x <= anodeR)) { // anode voltage
		double cPrime = (voltage - (tempt0 + tempt1 + tempt2 + tempt3 + tempt4 + tempt7 + tempt8 + tempt13 + tempt14 + tempt6 + tempt10 + tempt12
			+ tempt15 + tempt17 + tempt20 + tempt21 + tempt24 + tempt26)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 5)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 9)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 11)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 16)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 18)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 19)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 22)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 23)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 25)] = wd*cPrime;
	}


	//// z = NZ-1 (upper boundary n = (0,0,-1))
	//// c \dot n = 0: 0, 1, 2, 3, 4; 7, 8, 13, 14
	//// c \dot n > 0: 6; 10, 12, 15, 17; 20, 21, 24, 26
	//// c \dot n < 0: 5; 9, 11, 16, 18; 19, 22, 23, 25
	if ((z == NZ-1) && (x >= cathodeL && x <= cathodeR)) { // cathode grouned
		double cPrime = (0 - (tempt0 + tempt1 + tempt2 + tempt3 + tempt4 + tempt7 + tempt8 + tempt13 + tempt14 + tempt5 + tempt9 + tempt11
			+ tempt16 + tempt18 + tempt19 + tempt22 + tempt23 + tempt25)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 6)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 10)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 12)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 15)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 17)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 20)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 21)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 24)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 26)] = wd*cPrime;
	}

	if ((z == NZ - 1) && (x >= anodeL && x <= anodeR)) { // anode voltage
		double cPrime = (voltage - (tempt0 + tempt1 + tempt2 + tempt3 + tempt4 + tempt7 + tempt8 + tempt13 + tempt14 + tempt5 + tempt9 + tempt11
			+ tempt16 + tempt18 + tempt19 + tempt22 + tempt23 + tempt25)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 6)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 10)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 12)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 15)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 17)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 20)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 21)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 24)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 26)] = wd*cPrime;
	}

	// Wet-node boundary for Neumann
	// z = 0 (lower boundary n = (0,0,1))
	// c \dot n = 0: 0, 1, 2, 3, 4; 7, 8, 13, 14
	// c \dot n < 0: 6; 10, 12, 15, 17; 20, 21, 24, 26
	// c \dot n > 0: 5; 9, 11, 16, 18; 19, 22, 23, 25
	double omega_minus_phi = 1.0 / (beta / cs_square_phi / dt_phi + 1.0 / 2.0) / dt_phi;
	if ((z == 0) && ((x<cathodeL) || (x>cathodeR && x<anodeL) || (x>anodeR))) { // Neumann boundaries on substrate
		double flux0 = flux*CFL_phi / (omega_minus_phi / cs_square_phi);
		double cPrime = (flux0 + (tempt6 + tempt10 + tempt12
			+ tempt15 + tempt17 + tempt20 + tempt21 + tempt24 + tempt26)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 5)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 9)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 11)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 16)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 18)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 19)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 22)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 23)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 25)] = wd*cPrime;
	}

	// z = NZ-1 (upper boundary n = (0,0,-1))
	// c \dot n = 0: 0, 1, 2, 3, 4; 7, 8, 13, 14
	// c \dot n > 0: 6; 10, 12, 15, 17; 20, 21, 24, 26
	// c \dot n < 0: 5; 9, 11, 16, 18; 19, 22, 23, 25
	if ((z == NZ - 1) && ((x<cathodeL) || (x>cathodeR && x<anodeL) || (x>anodeR))) { // Neumann boundaries on substrate
		double flux1 = flux*CFL_phi / (omega_minus_phi / cs_square_phi);
		double cPrime = (flux1 + (tempt5 + tempt9 + tempt11
			+ tempt16 + tempt18 + tempt19 + tempt22 + tempt23 + tempt25)) / (ws + 4 * wa + 4 * wd);
		temp1[gpu_fieldn_index(x, y, z, 6)] = ws*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 10)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 12)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 15)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 17)] = wa*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 20)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 21)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 24)] = wd*cPrime;
		temp1[gpu_fieldn_index(x, y, z, 26)] = wd*cPrime;
	}


}



// =========================================================================
// Fast poisson solver domain extension
// =========================================================================

__host__ void fast_Poisson(double *charge_gpu, double *chargen_gpu, double *kx, double *ky, double *kz, cufftHandle plan) {

	checkCudaErrors(cudaMalloc((void**)&freq_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));
	checkCudaErrors(cudaMalloc((void**)&charge_gpu_ext, sizeof(cufftDoubleComplex)*mem_size_ext_scalar));


	// Extend the domain
	extension(charge_gpu, chargen_gpu, charge_gpu_ext);

	// Execute a real-to-complex 2D FFT
	CHECK_CUFFT(cufftExecZ2Z(plan, charge_gpu_ext, freq_gpu_ext, CUFFT_FORWARD));

	// Execute the derivatives in frequency domain
	derivative(kx, ky, kz, freq_gpu_ext);

	// Execute a complex-to-complex 2D IFFT
	CHECK_CUFFT(cufftExecZ2Z(plan, freq_gpu_ext, phi_gpu_ext, CUFFT_INVERSE));

	// Extraction of phi from extended domain phi_gpu_ext
	extract(phi_gpu, phi_gpu_ext);

	// Calculate electric field strength
	efield(phi_gpu, Ex_gpu, Ey_gpu, Ez_gpu);

	checkCudaErrors(cudaFree(charge_gpu_ext));
	checkCudaErrors(cudaFree(phi_gpu_ext));
	checkCudaErrors(cudaFree(freq_gpu_ext));
}

__host__ void extension(double *c, double *cn, cufftDoubleComplex *c_ext) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NE);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extension << < grid, threads >> > (c, cn, c_ext);
	getLastCudaError("Odd Extension error");
}

__global__ void odd_extension(double *charge, double *chargen, cufftDoubleComplex *charge_ext) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
		charge_ext[gpu_scalar_index(x, y, z)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == 1) {
                if (x < NX/2){
		    charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)])/ eps - voltage / dz / dz;
		}else{
                    charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)])/ eps - voltage2 / dz / dz;
                }

                charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z > 1 && z < NZ - 2) {
		charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)]) / eps;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ - 2) {
                if (x < NX/2){
		    charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)]) / eps - voltage / dz / dz;
                }else{
                    charge_ext[gpu_scalar_index(x, y, z)].x = -convertCtoCharge*(charge[gpu_scalar_index(x, y, z)] - chargen[gpu_scalar_index(x, y, z)]) / eps - voltage2 / dz / dz;
                } 

		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ - 1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = 0.0;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NZ) {
                if (x < NX/2){
                    charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, NE-z)] - chargen[gpu_scalar_index(x, y, NE-z)]) / eps + voltage / dz / dz;;
                }else{
                    charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, NE-z)] - chargen[gpu_scalar_index(x, y, NE-z)]) / eps + voltage2 / dz / dz;;
                }
		
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z > NZ && z<NE-1) {
		charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, NE-z)] - chargen[gpu_scalar_index(x, y, NE-z)]) / eps;
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
	if (z == NE - 1) {
                if (x < NX/2){
                    charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, 1)] - chargen[gpu_scalar_index(x, y, 1)]) / eps + voltage / dz / dz;
                }else{
                    charge_ext[gpu_scalar_index(x, y, z)].x = convertCtoCharge*(charge[gpu_scalar_index(x, y, 1)] - chargen[gpu_scalar_index(x, y, 1)]) / eps + voltage2 / dz / dz;
                }
		
		charge_ext[gpu_scalar_index(x, y, z)].y = 0.0;
		return;
	}
}

__host__ void derivative(double *kx, double *ky, double *kz, cufftDoubleComplex *source) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NE);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	gpu_derivative << < grid, threads >> > (kx, ky, kz, source);
	getLastCudaError("Gpu derivative error");
}
 
__global__ void gpu_derivative(double *kx, double *ky, double *kz, cufftDoubleComplex *source) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	double I = kx[x];
	double J = ky[y];
	double K = kz[z];
	double mu = (4.0 / dz / dz)*(sin(K*dz*0.5)*sin(K*dz*0.5)) + I*I + J*J;
	if (y == 0 && x == 0 && z == 0) mu = 1.0;
	source[gpu_scalar_index(x, y, z)].x = -source[gpu_scalar_index(x, y, z)].x / mu;
	source[gpu_scalar_index(x, y, z)].y = -source[gpu_scalar_index(x, y, z)].y / mu;
}

__host__ void extract(double *fi, cufftDoubleComplex *fi_ext) {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, NZ);
	// threads in block
	dim3  threads(nThreads, 1, 1);
	odd_extract << < grid, threads >> > (fi, fi_ext);
	getLastCudaError("Extraction error");
}

__global__ void odd_extract(double *phi, cufftDoubleComplex *phi_ext) {
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (z == 0) {
                if (x < NX/2){
    		    phi[gpu_scalar_index(x, y, z)] = voltage;
                }else{
		    phi[gpu_scalar_index(x, y, z)] = voltage2;
                }
		return;
	}
	if (z == NZ-1) {
		if (x < NX/2){
    		    phi[gpu_scalar_index(x, y, z)] = voltage;
                }else{
		    phi[gpu_scalar_index(x, y, z)] = voltage2;
                }

		return;
	}
	phi[gpu_scalar_index(x, y, z)] = phi_ext[gpu_scalar_index(x, y, z)].x/size;
}
