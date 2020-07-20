/*
Compiler options:
-O3 --std=c++14 --ptxas-options=-v --gpu-architecture=sm_35 -lineinfo -maxrregcount=128 -w --resource-usage
*/
#include <iostream>
#include <fstream>
#include <iomanip>
#include "DDEInit.h"
#include "GPUTimer.h"
#include "LorenzSystem.cuh"
#include "GPUDDESolver.cuh"

int main(int argc, char const *argv[])
{
		// get device information
		int dev = 0;
		cudaDeviceProp deviceProp;
		CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		printf("Using Device %d: %s\n", dev, deviceProp.name);
		CHECK(cudaSetDevice(dev));

		//constants
    const unsigned int nrOfInitialPoints = 50;
    const unsigned int nrOfSteps = 10000;
		const unsigned int nrOfPoints = nrOfSteps + 2 * nrOfInitialPoints;
		const unsigned int nrOfParameters = 32768;
		const unsigned int batchSize = 8192;
		const unsigned int nrOfBatches = (nrOfParameters + batchSize - 1)/batchSize;

		//memory sizes
		size_t tValsInitLen = nrOfInitialPoints;
		size_t xValsInitLen = nrOfInitialPoints * batchSize / 2;
		size_t tValsLen = nrOfPoints;
		size_t xValsLen = nrOfPoints * batchSize / 2;

		//fill integration settings struct
		integrationSettings intSettings;
		intSettings.nrOfInitialPoints = nrOfInitialPoints;
		intSettings.nrOfParameters = batchSize/2;
		intSettings.nrOfPoints = nrOfPoints;
		intSettings.nrOfSteps = nrOfSteps;
		intSettings.t0 = 13.0/28.0;
		intSettings.dt = intSettings.t0 / 400.0;

		//kernel configuration
		const unsigned int blocksize = 128;
		const unsigned int gridsize = (batchSize/2 + blocksize - 1) / blocksize;
		dim3 block(blocksize);
		dim3 grid(gridsize);

		//parameter stuff, initial CPU and GPU memory
		double * parameterListHost = linspace(47, 50, nrOfParameters);
		double * parameterListDevice;
		cudaMalloc((void**)&parameterListDevice,batchSize * sizeof(double));

		//discretize initial functions
		double * tInit = linspaceDisc(-13.0/28.0, 0.0, nrOfInitialPoints);
		double * y0Init = discretize(y0, tInit, nrOfInitialPoints);
		double * yd0Init = discretize(yd0, tInit, nrOfInitialPoints);

		//copy initial conditions to new bigger arrays
		double * y0 = new double[xValsInitLen];
		double * yd0 = new double[xValsInitLen];
		for (size_t i = 0; i < nrOfInitialPoints; i++)
		{
			for (size_t j = 0; j < batchSize/2; j++)
			{
				unsigned int idx = i*batchSize/2 + j;
				y0[idx] = y0Init[i];
				yd0[idx] = yd0Init[i];
			}
		}

		//allocate GPU memory
		cudaMalloc((void**)&intSettings.tVals, tValsLen*sizeof(double));
		cudaMalloc((void**)&intSettings.y0Vals, xValsLen*sizeof(double));
		cudaMalloc((void**)&intSettings.yd0Vals, xValsLen*sizeof(double));
		cudaMalloc((void**)&intSettings.y1Vals, xValsLen*sizeof(double));
		cudaMalloc((void**)&intSettings.yd1Vals, xValsLen*sizeof(double));

		//copy the initial values to gpu memory
		cudaMemcpy(intSettings.tVals, tInit,tValsInitLen*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(intSettings.y0Vals, y0,xValsInitLen*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(intSettings.yd0Vals, yd0,xValsInitLen*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(intSettings.y1Vals, y0,xValsInitLen*sizeof(double),cudaMemcpyHostToDevice);
		CHECK(cudaMemcpy(intSettings.yd1Vals, yd0,xValsInitLen*sizeof(double),cudaMemcpyHostToDevice));

		//information about the run
		printf("Memory size: %zd MB \n",(xValsLen*4+tValsLen)*sizeof(double)/1024/1024);
		printf("Launching kernel with <<<%d,%d>>> in %d batches\n",gridsize,blocksize,nrOfBatches);

		//save to file
		std::ofstream ofs("GPU_endvalues_new.txt");
		int id = nrOfInitialPoints + nrOfSteps - 1;

		//execution in batches
		double tStart = seconds();

		//execute in batches
		for (size_t k = 0; k < nrOfBatches; k++)
		{
			CHECK(cudaMemcpy(parameterListDevice,parameterListHost + k*batchSize,batchSize * sizeof(double),cudaMemcpyHostToDevice));

			//launch kernel
			solver<<<grid,block>>>(intSettings, parameterListDevice);
			CHECK(cudaDeviceSynchronize());

			//copy back to global memory
			double * y0Ref = new double[xValsLen];
			double * y1Ref = new double[xValsLen];
			CHECK(cudaMemcpy(y0Ref,intSettings.y0Vals,xValsLen*sizeof(double),cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(y1Ref,intSettings.y1Vals,xValsLen*sizeof(double),cudaMemcpyDeviceToHost));

			for (size_t i = 0; i < batchSize / 2; i++)
			{
					double y0 = y0Ref[id*batchSize/2 + i];
					double y1 = y1Ref[id*batchSize/2 + i];
					ofs << parameterListHost[k*batchSize + 2*i] <<"\t" << y0 << "\n";
					ofs << parameterListHost[k*batchSize + 2*i + 1] <<"\t" << y1 << "\n";
			}


			delete y0Ref, y1Ref;
		}
		double tEnd = seconds();
		printf("Execution finished for p = %d parameters in t = %lf s \n", nrOfParameters, (tEnd - tStart) );
		ofs.flush();
		ofs.close();

		//free gpu memomory
		cudaFree(parameterListDevice);

		//delete cpu memory
		delete parameterListHost;
		delete tInit, y0Init, yd0Init;
}
