#ifndef GPU_DDE_SOLVER_
#define GPU_DDE_SOLVER_


//------------------------------ GPU Functions and Stuff --------------------------------
struct integrationSettings
{
	//delays
	double t0;

	//counters
	unsigned int nrOfSteps, nrOfInitialPoints, nrOfPoints, nrOfParameters;

	//mesh
	unsigned int meshId;
	double mesh[3];

	//memory
	double * xVals, * yVals, * ydVals, *zVals, * tVals;
};

struct threadVariables
{
	unsigned int threadId;
	unsigned int lastIndex; //index prediction

	//integration variables:    4*3 + 6 = 18 double
	double x[3];
	double xTmp[3];
	double kAct[3];
	double kSum[3];
	double p, xDelay;
	double t, tTmp, dt, dtAct;


};


__forceinline__ __device__ unsigned int findIndex(double t, threadVariables *vars, integrationSettings intSettings)
{
	for (unsigned int i = (*vars).lastIndex; i < intSettings.nrOfPoints; i++)
	{
		if (t < intSettings.tVals[i])
		{
			if ((unsigned int)(i - 2) <(unsigned int)1 << 31) (*vars).lastIndex = i - 2;
			else (*vars).lastIndex = 0;
			return i;
		}
	}
	return (unsigned int)0;
}


__forceinline__ __device__ double denseOutput(double t, threadVariables *vars, integrationSettings intSettings)
{
	unsigned int step = findIndex(t, vars, intSettings) - 1;
	double tb = intSettings.tVals[step];
	double tn = intSettings.tVals[step+1];
	double xb = intSettings.yVals[step * intSettings.nrOfParameters + (*vars).threadId];
	double xn = intSettings.yVals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
	double xdb = intSettings.ydVals[step * intSettings.nrOfParameters + (*vars).threadId];
	double xdn = intSettings.ydVals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];

	double deltat = tn - tb;
	double theta = (t - tb) / (tn - tb);
	double res = (1 - theta)*xb + theta * xn + theta * (theta - 1)*((1 - 2 * theta)*(xn - xb) + (theta - 1)*deltat*xdb + theta * deltat*xdn);
	return res;
}

//----------------------------------- integration  -------------------------------------
__global__ void solver(integrationSettings intSettings, double * parameters)
{
	//Starting thread
	threadVariables vars;
	intSettings.meshId = 0;

	//calculate thread id
	vars.threadId = blockDim.x*blockIdx.x + threadIdx.x;

	//read parameter from global memory
	if(vars.threadId < intSettings.nrOfParameters) vars.p = parameters[vars.threadId];
	else printf("INDEX OUT OF MEMORY");

	//initialize thread variables
	vars.dt = 10.0/double(intSettings.nrOfSteps);
	vars.t = 0;
	vars.lastIndex = 0;

	//read initial values
	__shared__ unsigned int idx,idxLin;
	idx = intSettings.nrOfInitialPoints-1;
	idxLin = (intSettings.nrOfInitialPoints-1) * intSettings.nrOfParameters + vars.threadId;
	vars.x[0] = intSettings.xVals[idxLin];
	vars.x[1] = intSettings.yVals[idxLin];
	vars.x[2] = intSettings.zVals[idxLin];
	idxLin += intSettings.nrOfParameters;
	idx++;

	//set initial values to save derivative from positive direction
	intSettings.tVals[idx] = vars.t;
	intSettings.xVals[idxLin] = vars.x[0];
	intSettings.yVals[idxLin] = vars.x[1];
	intSettings.zVals[idxLin] = vars.x[2];

	//integrate
	while(vars.t <= 10.0)
	{
		//----------------------------- Modify dt --------------------------------------------
		//assuming a simple step
		vars.dtAct = vars.dt;

		//detecting meshpoinr
		if (intSettings.meshId < 3 && intSettings.mesh[intSettings.meshId] < vars.t + vars.dt + 1e-15) //dt should be modified to reach a simple mesh point
		{
			vars.dtAct = intSettings.mesh[intSettings.meshId] - vars.t;
			intSettings.meshId++;
		}

		//----------------------------- START OF RK4 STEP ------------------------------------
		//k1
		vars.xDelay = denseOutput(vars.t-intSettings.t0, &vars, intSettings);
		f(vars.kAct, vars.t, vars.x, vars.xDelay, vars.p);

		//saving the derivative
		intSettings.ydVals[idxLin] = vars.kAct[1];

		//k2
		vars.tTmp = vars.t + 0.5*vars.dtAct;
		vars.xDelay = denseOutput(vars.tTmp-intSettings.t0, &vars, intSettings);

		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] = vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k3
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k4
		vars.tTmp = vars.t + vars.dtAct;
		vars.xDelay = denseOutput(vars.tTmp-intSettings.t0, &vars, intSettings);
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//result of step
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += vars.kAct[i];
			vars.x[i] += 1. / 6. * vars.dtAct * vars.kSum[i];
		}
		vars.t += vars.dtAct;
		//-----------------------------  END  OF RK4 STEP ------------------------------------

		//----------------------------- SAVE T AND X TO GLOBAL MEMORY ------------------------------------
		idxLin += intSettings.nrOfParameters;
		idx++;
		intSettings.tVals[idx] = vars.t;
		intSettings.xVals[idxLin] = vars.x[0];
		intSettings.yVals[idxLin] = vars.x[1];
		intSettings.zVals[idxLin] = vars.x[2];
	}
}

#endif
