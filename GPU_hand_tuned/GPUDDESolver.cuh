#ifndef GPU_DDE_SOLVER_
#define GPU_DDE_SOLVER_



//------------------------------ GPU Functions and Stuff --------------------------------
template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
struct integrationMemory
{
	double xVals[nrOfPoints*nrOfVars];
	double xdVals[nrOfPoints*nrOfVars];
	double tVals[nrOfPoints];
};

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
struct integrationSettings
{
	//range
	double tStart, tEnd;

	//delays
	double t0[nrOfDelays];
	double varId[nrOfDelays];

	//counters
	unsigned int nrOfSteps, nrOfInitialPoints;

	//mesh
	unsigned int meshLen, meshId;
	double * mesh;
	int * meshType;
};

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
struct threadVariables
{
	unsigned int threadId;

	//integration variables
	double x[nrOfVars];
	double xTmp[nrOfVars];
	double xDelay[nrOfVars];
	double kAct[nrOfVars];
	double kSum[nrOfVars];
	double t,tTmp, dt, p, dtAct;

	//index prediction
	unsigned int lastIndex[nrOfDelays];
};

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
__device__ unsigned int findIndex(double t, unsigned int delayID, threadVariables<nrOfVars,nrOfDelays,nrOfPoints> *vars, integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> intSettings, integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * memory)
{
	for (unsigned int i = (*vars).lastIndex[delayID]; i < nrOfPoints; i++)
	{
		if (t < memory[(*vars).threadId].tVals[i])
		{
			if ((unsigned int)(i - 2) <(unsigned int)1 << 31) (*vars).lastIndex[delayID] = i - 2;
			else (*vars).lastIndex[delayID] = 0;
			return i;
		}
	}
	return (unsigned int)0;
}


template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
__device__ double denseOutput(double t,unsigned int var, unsigned int delayID, threadVariables<nrOfVars,nrOfDelays,nrOfPoints> *vars, integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> intSettings, integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * memory)
{
	unsigned int step = findIndex<nrOfVars,nrOfDelays,nrOfPoints>(t, delayID, vars, intSettings, memory) - 1;
	double tb = memory[(*vars).threadId].tVals[step];
	double tn = memory[(*vars).threadId].tVals[step+1];
	double xb = memory[(*vars).threadId].xVals[step + var * nrOfPoints];
	double xn = memory[(*vars).threadId].xVals[step + 1 + var * nrOfPoints];
	double xdb = memory[(*vars).threadId].xdVals[step + var * nrOfPoints];
	double xdn = memory[(*vars).threadId].xdVals[step + 1 + var * nrOfPoints];
	double deltat = tn - tb;
	double theta = (t - tb) / (tn - tb);
	double res = (1 - theta)*xb + theta * xn + theta * (theta - 1)*((1 - 2 * theta)*(xn - xb) + (theta - 1)*deltat*xdb + theta * deltat*xdn);
	return res;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
__device__ void calculateDelay(double t,threadVariables<nrOfVars,nrOfDelays,nrOfPoints> *vars, integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> intSettings, integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * memory)
{
	for (size_t i = 0; i < nrOfDelays; i++)
	{
		(*vars).xDelay[i] = denseOutput<nrOfVars,nrOfDelays,nrOfPoints>(t-intSettings.t0[i], intSettings.varId[i], i, vars, intSettings, memory);
	}
}

//----------------------------------- integration  -------------------------------------
template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
__global__ void solver(integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> intSettings, double * parameters, const unsigned int nrOfParameters, integrationMemory<nrOfVars,nrOfDelays,nrOfPoints> * memory)
{
	//Starting thread
	threadVariables<nrOfVars,nrOfDelays,nrOfPoints> vars;
	intSettings.meshId = 0;

	//calculate thread id
	vars.threadId = blockDim.x*blockIdx.x + threadIdx.x;

	//read parameter from global memory
	if(vars.threadId < nrOfParameters) vars.p = parameters[vars.threadId];
	else printf("INDEX OUT OF MEMORY");

	//thread variables
	vars.dt = (intSettings.tEnd-intSettings.tStart)/double(intSettings.nrOfSteps);
	vars.t = intSettings.tStart;
	unsigned int memoryId = intSettings.nrOfInitialPoints;
	unsigned int offset = nrOfPoints;
	for (size_t i = 0; i < nrOfDelays; i++)
	{
		vars.lastIndex[i] = 0;
	}

	//read initial values
	for (size_t i = 0; i < nrOfVars; i++)
	{
		vars.x[i] = memory[vars.threadId].xVals[memoryId - 1 + i*offset];
		memory[vars.threadId].xVals[memoryId + i*offset] = vars.x[i];
	}

	//integrate
	while( vars.t <= intSettings.tEnd)
	{
		//----------------------------- Modify dt --------------------------------------------
		//assuming a simple step
		vars.dtAct = vars.dt;
		int stepType = 0;

		//detecting simple or double mesh
		if (intSettings.meshId < intSettings.meshLen && intSettings.mesh[intSettings.meshId] < vars.t + vars.dt + 1e-15) //dt should be modified to reach a simple mesh point
		{
			stepType = intSettings.meshType[intSettings.meshId];
			if (stepType == 1)
			{
				vars.dtAct = intSettings.mesh[intSettings.meshId] - vars.t;
			}
			intSettings.meshId++;
		}

		//----------------------------- START OF RK4 STEP ------------------------------------
		//k1
		calculateDelay<nrOfVars,nrOfDelays,nrOfPoints>(vars.t,&vars,intSettings,memory);
		f(vars.kAct, vars.t, vars.x, vars.xDelay, vars.p);
		//if(vars.threadId == 5) printf("xdelay: %16.13lf\t", vars.xDelay[0]);
		//if(vars.threadId == 5) printf("t: %6.3lf k1: %16.13lf\t",vars.t,vars.kAct[0]);

		//k2
		vars.tTmp = vars.t + 0.5*vars.dtAct;
		calculateDelay<nrOfVars,nrOfDelays,nrOfPoints>(vars.tTmp,&vars,intSettings,memory);
		for (size_t i = 0; i < nrOfVars; i++)
		{
			memory[vars.threadId].xdVals[memoryId + i*offset] = vars.kAct[i]; //saving to global memory
			vars.kSum[i] = vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);
		//if(vars.threadId == 5) printf("k2: %16.13lf\t",vars.kAct[0]);

		//k3
		for (size_t i = 0; i < nrOfVars; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + 0.5*vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);
		//if(vars.threadId == 5) 	printf("k3: %16.13lf\t",vars.kAct[0]);

		//k4
		vars.tTmp = vars.t + vars.dtAct;
		calculateDelay<nrOfVars,nrOfDelays,nrOfPoints>(vars.tTmp,&vars,intSettings,memory);
		for (size_t i = 0; i < nrOfVars; i++)
		{
			vars.kSum[i] += 2*vars.kAct[i];
			vars.xTmp[i] = vars.x[i] + vars.dtAct*vars.kAct[i];
		}
		f(vars.kAct, vars.tTmp, vars.xTmp, vars.xDelay, vars.p);
	//	if(vars.threadId == 5) printf("k4: %16.13lf\t",vars.kAct[0]);

		//result of step
		for (size_t i = 0; i < nrOfVars; i++)
		{
			vars.kSum[i] += vars.kAct[i];
			vars.x[i] += 1. / 6. * vars.dtAct * vars.kSum[i];
		}
		vars.t += vars.dtAct;
		//-----------------------------  END  OF RK4 STEP ------------------------------------

		//----------------------------- SAVE T AND X TO GLOBAL MEMORY ------------------------------------
		memoryId++;
		memory[vars.threadId].tVals[memoryId] = vars.t;
		for (size_t i = 0; i < nrOfVars; i++)
		{
			memory[vars.threadId].xVals[memoryId + i*offset] = vars.x[i];
		}

	}
}


//----------------------------------- Mesh calculation -----------------------------------
double * calculateMesh(double * a, double * b, unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth)
{
	double * newA = new double[sizeA*sizeB + sizeA];
	for (size_t i = 0; i < sizeA; i++)
	{
		newA[i] = a[i];
	}
	for (size_t i = 0; i < sizeA; i++)
	{
		for (size_t j = 0; j < sizeB; j++)
		{
			int id = i * sizeB + j + sizeA;
			newA[id] = a[i] + b[j];
		}
	}
	if (recursionDepth == 1) return newA;
	return calculateMesh(newA, b, sizeA*sizeB + sizeA, sizeB, recursionDepth - 1);
}

unsigned int calculateLength(unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth)
{
	if (recursionDepth == 1) return sizeA * sizeB + sizeA;
	return calculateLength(sizeA, sizeB, recursionDepth - 1) * (sizeB + 1);
}

double * filter(double * original, unsigned int * nr, double min, double * toRemove = NULL, unsigned int nrToRemove = 0, double tol = 1e-12)
{
	unsigned int count = *nr;
	double * unique = new double[count];
	unsigned int uniqueNr = 0;
	for (size_t i = 0; i < count; i++)
	{
		bool set = true;
		for (size_t j = 0; j < uniqueNr; j++)
		{
			if (abs(original[i] - unique[j]) < tol) //already in new list
			{
				set = false;
			}
		}

		for (size_t j = 0; j < nrToRemove; j++)
		{
			if (abs(original[i] - toRemove[j]) < tol) //already in concurrent list
			{
				set = false;
			}
		}

		if (set && original[i] > min) //original[i] not in new list yet
		{
			unique[uniqueNr] = original[i];
			uniqueNr++;
		}
	}

	double * filtered = new double[uniqueNr];
	for (size_t i = 0; i < uniqueNr; i++)
	{
		filtered[i] = unique[i];
	}
	*nr = uniqueNr;
	delete unique;
	return filtered;
}

void sort(double * lst, unsigned int len)
{
	//bubblesort
	for (size_t i = 1; i < len; i++)
	{
		for (size_t j = 1; j < len; j++)
		{
			if (lst[j] < lst[j - 1]) //swap
			{
				double tmp = lst[j];
				lst[j] = lst[j - 1];
				lst[j - 1] = tmp;
			}
		}
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfPoints>
void analyzeInit(bool print, double * tVals, double * xVals, double * xdVals, integrationSettings<nrOfVars,nrOfDelays,nrOfPoints> * intSettings)
{
	//phase 1: counting
	unsigned int nrOfC0 = 0;
	unsigned int nrOfC1 = 1;
	for (size_t i = 1; i < (*intSettings).nrOfInitialPoints; i++)
	{
		if (tVals[i] == tVals[i - 1]) //double initial point found
		{
			//categorization of this discontinous point
			for (size_t k = 0; k < nrOfVars; k++)
			{
				if (xVals[i + k*nrOfPoints] == xVals[i - 1 + k*nrOfPoints] && xdVals[i+k*nrOfPoints] != xdVals[i - 1 + k*nrOfPoints]) //C1 discontinouity
				{
					nrOfC1++;
					break;
				}

				if (xVals[i + k * nrOfPoints] != xVals[i - 1 + k *nrOfPoints]) //C0 discontinouity
				{
					nrOfC0++;
					break;
				}
			}
		}
	}


	//phase 2: saving points
	double * C0disc = new double[nrOfC0];
	double * C1disc = new double[nrOfC1];
	C1disc[nrOfC1-1] = 0.0;
	unsigned int c0 = 0, c1 = 0;
	for (size_t i = 1; i < (*intSettings).nrOfInitialPoints; i++)
	{
		if (tVals[i] == tVals[i - 1]) //double initial point found
		{
			//categorization of this discontinous point
			for (size_t k = 0; k < nrOfVars; k++)
			{
				if (xVals[i+ k*nrOfPoints] == xVals[i - 1+k*nrOfPoints] && xdVals[i+k*nrOfPoints] != xdVals[i - 1+k*nrOfPoints]) //C1 discontinouity
				{
					C1disc[c1] = tVals[i];
					c1++;
					break;
				}

				if (xVals[i + k*nrOfPoints] != xVals[i - 1 + k*nrOfPoints]) //C0 discontinouity
				{
					C0disc[c0] = tVals[i];
					c0++;
					break;
				}
			}
		}
	}

	//phase 3: calculating double points mesh
	double * meshDouble = calculateMesh(C0disc, (*intSettings).t0, nrOfC0, nrOfDelays, 1);
	unsigned int meshDoubleLen = calculateLength(nrOfC0, nrOfDelays, 1);
	meshDouble = filter(meshDouble, &meshDoubleLen, (*intSettings).tStart);

	//phase 4: concatenating double mesh points and c1 discontinouties lists
	unsigned int newLen = meshDoubleLen + nrOfC1;
	double * newList = new double[newLen];
	for (size_t i = 0; i < nrOfC1; i++)
	{
		newList[i] = C1disc[i];
	}
	for (size_t i = 0; i < meshDoubleLen; i++)
	{
		newList[i+nrOfC1] = meshDouble[i];
	}

	//phase 6: calculate simple mesh
	(*intSettings).mesh = calculateMesh(newList, (*intSettings).t0, newLen, nrOfDelays, 3);
	(*intSettings).meshLen = calculateLength(newLen, nrOfDelays, 3);
	(*intSettings).mesh = filter((*intSettings).mesh, &((*intSettings).meshLen), (*intSettings).tStart);
	sort((*intSettings).mesh, (*intSettings).meshLen);

	(*intSettings).meshType = new int[(*intSettings).meshLen];
	for (size_t i = 0; i < (*intSettings).meshLen; i++)
	{
		(*intSettings).meshType[i] = 1;
		for (size_t j = 0; j < meshDoubleLen; j++)
		{
			if ((*intSettings).mesh[i] == meshDouble[j])
			{
				(*intSettings).meshType[i] = 2;
			}
		}
	}

	if (print)
	{
		//print discontinouties
		printf("%d + %d discontinouities are found in the initial function:\nC0: ",nrOfC0,nrOfC1);
		for (size_t i = 0; i < nrOfC0; i++)
		{
			printf("%5.3lf\t", C0disc[i]);
		}
		printf("\nC1: ");
		for (size_t i = 0; i < nrOfC1; i++)
		{
			printf("%5.3lf\t", C1disc[i]);
		}
		printf("\n");
	}

	delete C0disc, C1disc, meshDouble, newList;
}


#endif
