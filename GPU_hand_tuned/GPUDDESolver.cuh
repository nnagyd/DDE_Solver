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
	double * y1Vals, * yd1Vals,* y0Vals, * yd0Vals, * tVals;
};

struct threadVariables
{
	// 2 unsigned int -> 8 byte 
	unsigned int threadId;
	unsigned int lastIndex; //index prediction

	//integration variables:    4*6 + 2*2 + 3 = 31 double -> 31*4 = 124 byte
	double x[6];
	double xTmp[6];
	double kAct[6];
	double kSum[6];
	double p[2], xDelay[2];
	double t, tTmp, dt;

	//memory loads
	//4*2 + 2 = 10 double -> 8 * 10 = 80 byte
	double tb,xb[2],xn[2],xdb[2],xdn[2],deltat;
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
	(*vars).deltat = intSettings.tVals[step+1] - (*vars).tb;

	//to 1. equation
	(*vars).xb[0] = intSettings.y0Vals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xn[0] = intSettings.y0Vals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdb[0] = intSettings.yd0Vals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdn[0] = intSettings.yd0Vals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];

	//to 2. equation
	(*vars).xb[1] = intSettings.y1Vals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xn[1] = intSettings.y1Vals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdb[1] = intSettings.yd1Vals[step * intSettings.nrOfParameters + (*vars).threadId];
	(*vars).xdn[1] = intSettings.yd1Vals[(step + 1) * intSettings.nrOfParameters + (*vars).threadId];
}


__forceinline__ __device__ double denseOutput(double t, unsigned int id, threadVariables *vars, integrationSettings intSettings)
{
	double theta = (t - (*vars).tb) / (*vars).deltat;
	double res = (1 - theta)*(*vars).xb[id] + theta * (*vars).xn[id] + theta * (theta - 1)*((1 - 2 * theta)*((*vars).xn[id] - (*vars).xb[id]) + (theta - 1)*(*vars).deltat*(*vars).xdb[id] + theta * (*vars).deltat*(*vars).xdn[id]);
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
	if(vars.threadId < intSettings.nrOfParameters)
	{
		unsigned int idTmp = 2*vars.threadId;
		vars.p[0] = parameters[idTmp];
		vars.p[1] = parameters[idTmp + 1];
	}
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
	vars.x[1] = intSettings.y0Vals[idxLin];
	vars.x[2] = -8.0;
	vars.x[3] = -8.0;
	vars.x[4] = intSettings.y1Vals[idxLin];
	vars.x[5] = -8.0;
	idxLin += intSettings.nrOfParameters;
	idx++;

	//set initial values to save derivative from positive direction
	intSettings.tVals[idx] = vars.t;
	intSettings.y0Vals[idxLin] = vars.x[1];
	intSettings.y1Vals[idxLin] = vars.x[4];

	//integrate
	for(size_t stepNumber = 0; stepNumber < intSettings.nrOfSteps; stepNumber++)
	{
		//------------------------ LOAD DENSE denseOutput VALUES -----------------------------
		vars.tTmp = vars.t + 0.5*vars.dt;
		loadValues(vars.tTmp-intSettings.t0,&vars,intSettings);

		//----------------------------- START OF RK4 STEP ------------------------------------
		//k1
		vars.xDelay[0] = denseOutput(vars.t-intSettings.t0, 0, &vars, intSettings);
		vars.xDelay[1] = denseOutput(vars.t-intSettings.t0, 1, &vars, intSettings);
		f(vars.kAct, vars.t, vars.x, vars.xDelay, vars.p);

		//saving the derivative
		intSettings.yd0Vals[idxLin] = vars.kAct[1];
		intSettings.yd1Vals[idxLin] = vars.kAct[4];

		//k2
		vars.xDelay[0] = denseOutput(vars.tTmp-intSettings.t0,0, &vars, intSettings);
		vars.xDelay[1] = denseOutput(vars.tTmp-intSettings.t0,1, &vars, intSettings);
		#pragma unroll 6
		for (size_t i = 0; i < 6; i++)
		{
			vars.kSum[i] = vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k3
		#pragma unroll 6
		for (size_t i = 0; i < 6; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//k4
		vars.tTmp = vars.t + vars.dt;
		vars.xDelay[0] = denseOutput(vars.tTmp-intSettings.t0,0, &vars, intSettings);
		vars.xDelay[1] = denseOutput(vars.tTmp-intSettings.t0,1, &vars, intSettings);
		#pragma unroll 6
		for (size_t i = 0; i < 6; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + vars.dt*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);

		//result of step
		#pragma unroll 6
		for (size_t i = 0; i < 6; i++)
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
		intSettings.y0Vals[idxLin] = vars.x[1];
		intSettings.y1Vals[idxLin] = vars.x[4];

		//debug
		if(vars.threadId == 0)
		{
			//printf("t=%6.3lf \t y0=%6.3lf \t y1=%6.3lf\n",intSettings.tVals[idx],intSettings.y0Vals[idxLin] = vars.x[1],	intSettings.y1Vals[idxLin] = vars.x[4]);
		}
	}
}

#endif
