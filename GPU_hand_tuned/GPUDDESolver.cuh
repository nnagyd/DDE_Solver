#ifndef GPU_DDE_SOLVER_
#define GPU_DDE_SOLVER_


//------------------------------ GPU Functions and Stuff --------------------------------
struct integrationSettings
{
	//delays
	double t0, dt;

	//counters
	unsigned int nrOfSteps, nrOfInitialPoints, nrOfPoints, nrOfParameters;

	//memory
	double * yVals, * ydVals, * tVals;
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
	double t, tTmp, dt;

	//memory loads
	double tb,xb,xn,xdb,xdn,deltat;
};


__forceinline__ __device__ unsigned int findIndex(double t, threadVariables *vars, integrationSettings intSettings)
{
	for (unsigned int i = (*vars).lastIndex; i < intSettings.nrOfPoints; i++)
	{
		if (t < intSettings.tVals[i])
		{
			if (i >= 1) (*vars).lastIndex = i;
			else (*vars).lastIndex = 0;
			return i;
		}
	}
	return (unsigned int)0;
}

__forceinline__ __device__ void loadValues(double t, threadVariables *vars, integrationSettings intSettings)
{
	unsigned int step = findIndex(t, vars, intSettings) - 1;
	(*vars).tb = intSettings.tVals[step];
	(*vars).xb = intSettings.yVals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xn = intSettings.yVals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdb = intSettings.ydVals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdn = intSettings.ydVals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).deltat = intSettings.tVals[step+1] - (*vars).tb;
}


__forceinline__ __device__ double denseOutput(double t, threadVariables *vars, integrationSettings intSettings)
{
	double theta = (t - (*vars).tb) / (*vars).deltat;
	double res = (1 - theta)*(*vars).xb + theta * (*vars).xn + theta * (theta - 1)*((1 - 2 * theta)*((*vars).xn - (*vars).xb) + (theta - 1)*(*vars).deltat*(*vars).xdb + theta * (*vars).deltat*(*vars).xdn);
	return res;
}

//----------------------------------- integration  -------------------------------------
__global__ void solver(integrationSettings intSettings, double * parameters)
{
	//Starting thread
	threadVariables vars;

	//calculate thread id
	vars.threadId = blockDim.x*blockIdx.x + threadIdx.x;

	//read parameter from global memory
	if(vars.threadId < intSettings.nrOfParameters) vars.p = parameters[vars.threadId];
	else printf("INDEX OUT OF MEMORY");

	//initialize thread variables
	vars.dt = intSettings.dt;
	vars.t = 0;
	vars.lastIndex = 0;

	//read initial values
	unsigned int idx,idxLin;
	idx = intSettings.nrOfInitialPoints-1;
	idxLin = (intSettings.nrOfInitialPoints-1) * intSettings.nrOfParameters + vars.threadId;
	vars.x[0] = -8.0;
	vars.x[1] = intSettings.yVals[idxLin];
	vars.x[2] = -8.0;
	idxLin += intSettings.nrOfParameters;
	idx++;

	//set initial values to save derivative from positive direction
	intSettings.tVals[idx] = vars.t;
	intSettings.yVals[idxLin] = vars.x[1];

	//integrate
	for(size_t stepNumber = 0; stepNumber < intSettings.nrOfSteps; stepNumber++)
	{
		//------------------------ LOAD DENSE denseOutput VALUES -----------------------------
		vars.tTmp = vars.t + 0.5*vars.dt;
		loadValues(vars.tTmp-intSettings.t0,&vars,intSettings);

		//----------------------------- START OF RK4 STEP ------------------------------------
		//k1
		vars.xDelay = denseOutput(vars.t-intSettings.t0, &vars, intSettings);
		f(vars.kAct, vars.t, vars.x, vars.xDelay, vars.p);

		//saving the derivative
		intSettings.ydVals[idxLin] = vars.kAct[1];

		//k2
		vars.xDelay = denseOutput(vars.tTmp-intSettings.t0, &vars, intSettings);
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] = vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k3
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k4
		vars.tTmp = vars.t + vars.dt;
		vars.xDelay = denseOutput(vars.tTmp-intSettings.t0, &vars, intSettings);
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//result of step
		#pragma unroll 3
		for (size_t i = 0; i < 3; i++)
		{
			vars.kSum[i] += vars.kAct[i];
			vars.x[i] += 1. / 6. * vars.dt * vars.kSum[i];
		}
		vars.t += vars.dt;
		//-----------------------------  END  OF RK4 STEP ------------------------------------

		//----------------------------- SAVE T AND X TO GLOBAL MEMORY ------------------------------------
		idxLin += intSettings.nrOfParameters;
		idx++;
		intSettings.tVals[idx] = vars.t;
		intSettings.yVals[idxLin] = vars.x[1];
	}
}

#endif
