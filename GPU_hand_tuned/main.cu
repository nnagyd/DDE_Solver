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

		//system settings
		const unsigned int nrOfVars = 3;
		const unsigned int nrOfDelays = 1;

		//constants
    const unsigned int nrOfInitialPoints = 100;
    const unsigned int nrOfSteps = 10000;
		const unsigned int nrOfPoints = nrOfSteps + 2 * nrOfInitialPoints;
		const unsigned int nrOfParameters = 32768;
		const unsigned int batchSize = 512;
		const unsigned int interpolationMemorySize = sizeof(integrationMemory<nrOfVars,nrOfDelays,nrOfPoints>)*batchSize;
		const unsigned int parameterMemorySize = batchSize * sizeof(double);
		const unsigned int nrOfBatches = (nrOfParameters + batchSize - 1)/batchSize;

		//fill integration settings struct
		integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> intSettings;
		intSettings.tStart = 0.;
		intSettings.tEnd = 10.0;
		intSettings.nrOfInitialPoints = nrOfInitialPoints;
		intSettings.nrOfSteps = nrOfSteps;
		intSettings.varId[0] = 1;
		intSettings.t0[0] = 13.0/28.0;

		//kernel configuration
		const unsigned int blocksize = 64;
		const unsigned int gridsize = (batchSize + blocksize - 1) / blocksize;
		dim3 block(blocksize);
		dim3 grid(gridsize);

		//parameter stuff, initial CPU and GPU memory
		double * parameterListHost = linspace(47, 50, nrOfParameters);
		double * parameterListDevice;
		cudaMalloc((void**)&parameterListDevice,parameterMemorySize);

		//allocate CPU memory
		integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * hostMem;
		hostMem = new integrationMemory<nrOfVars,nrOfDelays,nrOfPoints>[batchSize];

		//discretize initial functions
		double * tInit = linspaceDisc(-1.0, 0.0, nrOfInitialPoints);
		double * x0Init = discretize(x0, tInit, nrOfInitialPoints);
		double * xd0Init = discretize(xd0, tInit, nrOfInitialPoints);
		double * y0Init = discretize(y0, tInit, nrOfInitialPoints);
		double * yd0Init = discretize(yd0, tInit, nrOfInitialPoints);
		double * z0Init = discretize(z0, tInit, nrOfInitialPoints);
		double * zd0Init = discretize(zd0, tInit, nrOfInitialPoints);

		//load host memory with initial points
		for (size_t i = 0; i < batchSize; i++) //loop through parameters
		{
			for (size_t j = 0; j < nrOfPoints; j++) //loop through initial points
			{
				if(j < nrOfInitialPoints)
				{
					hostMem[i].tVals[j] 								= tInit[j];
					hostMem[i].xVals[j] 								= x0Init[j];
					hostMem[i].xVals[j + nrOfPoints] 		= y0Init[j];
					hostMem[i].xVals[j + 2*nrOfPoints] 	= z0Init[j];
					hostMem[i].xdVals[j] 								= xd0Init[j];
					hostMem[i].xdVals[j + nrOfPoints]		= yd0Init[j];
					hostMem[i].xdVals[j + 2*nrOfPoints]	= zd0Init[j];
					//printf("i = %zd j = %zd t = %lf x = %lf\n",i,j,tInit[j],y0Init[j]);
				}
				else
				{
					hostMem[i].tVals[j] 								= 0;
					hostMem[i].xVals[j] 								= 0;
					hostMem[i].xVals[j + nrOfPoints] 		= 0;
					hostMem[i].xVals[j + 2*nrOfPoints] 	= 0;
					hostMem[i].xdVals[j] 								= 0;
					hostMem[i].xdVals[j + nrOfPoints]		= 0;
					hostMem[i].xdVals[j + 2*nrOfPoints]	= 0;
				}
			}
		}
		//allocate GPU memory
		integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * devMem;
		cudaMalloc((void**)&devMem, interpolationMemorySize);
		CHECK(cudaMemcpy(devMem,hostMem,interpolationMemorySize,cudaMemcpyHostToDevice));

		//information about the run
		printf("Memory size: %d MB \n",interpolationMemorySize/1024/1024);
		printf("Launching kernel with <<<%d,%d>>> \n",gridsize,blocksize);

		//analyze initial conditions
		analyzeInit<nrOfVars,nrOfDelays,nrOfPoints>(true,tInit,x0Init,xd0Init,&intSettings);

		//integration setting copy to device
		integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> deviceIntSettings = intSettings;
		unsigned int meshSizeDouble = intSettings.meshLen * sizeof(double);
		unsigned int meshSizeInt = intSettings.meshLen * sizeof(int);
		cudaMalloc((void**)&deviceIntSettings.mesh,meshSizeDouble);
		cudaMalloc((void**)&deviceIntSettings.meshType,meshSizeInt);
		cudaMemcpy(deviceIntSettings.mesh,intSettings.mesh,meshSizeDouble,cudaMemcpyHostToDevice);
		cudaMemcpy(deviceIntSettings.meshType,intSettings.meshType,meshSizeInt,cudaMemcpyHostToDevice);

		//save end values
		std::ofstream ofs("GPU_endvalues.txt");
		int id = nrOfInitialPoints + 1 + nrOfSteps;

		//execution in batches
		double tStart = seconds();
		for (size_t k = 0; k < nrOfBatches; k++)
		{
			//copy new batch to memory
			CHECK(cudaMemcpy((void**)parameterListDevice,parameterListHost + k*batchSize,parameterMemorySize,cudaMemcpyHostToDevice));

			//launch kernel
			solver<nrOfVars,nrOfDelays,nrOfPoints><<<grid,block>>>(deviceIntSettings, parameterListDevice, batchSize, devMem);
			CHECK(cudaDeviceSynchronize());

			//copy back to global memory
			cudaMemcpy(hostMem,devMem,interpolationMemorySize,cudaMemcpyDeviceToHost);

			for (size_t i = 0; i < batchSize; i++)
			{
				double t = hostMem[i].tVals[id];
				double x = hostMem[i].xVals[id];
				double y = hostMem[i].xVals[id+nrOfPoints];
				double z = hostMem[i].xVals[id+2*nrOfPoints];
				ofs << parameterListHost[i+k*batchSize] << "\t" << t << "\t"<< x << "\t"<< y << "\t"<< z << "\n";
			}
		}
		double tEnd = seconds();
		ofs.flush();
		ofs.close();
		printf("Execution finished for %d parameters in t = %lf s \n", nrOfParameters, (tEnd - tStart) );

		//free gpu memomory
		cudaFree(devMem);
		cudaFree(parameterListDevice);

		//delete cpu memory
		delete hostMem,parameterListHost;
		delete tInit, x0Init, xd0Init, y0Init, yd0Init, z0Init, zd0Init;
}
