#include "DDESolver.h"
#include <string>
#include <fstream>
#include <iomanip>

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::DDESolver()
{
	//allocate memory - vars
	kAct = new double[nrOfVars];
	kSum = new double[nrOfVars];
	xTmp = new double[nrOfVars];
	x = new double[nrOfVars];
	xd = new double[nrOfVars];

	//allocate memory - delays
	xDelayed = new double[nrOfDelays];
	lastIndex = new unsigned int[nrOfDelays];
	t0 = new double[nrOfDelays];
	varId = new unsigned int[nrOfDelays];

	//allocate memory - parameters
	p = new double[nrOfParameters];
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::save(std::string fn, bool saveStart)
{
	int startIndex = 0;
	if (!saveStart)
	{
		startIndex = nrOfInitialPoints;
	}
	std::ofstream file(fn);
	file << std::setprecision(17);
	for (size_t i = startIndex; i < memoryId; i++)
	{
		file << tVals[i] << "\t";
		for (size_t j = 0; j < nrOfVars; j++)
		{
			file << xVals[i][j] << "\t";
		}
		file << "\n";
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
double * DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::saveEnd(bool includePar)
{
	double * tmp;
	if (includePar)
	{
		tmp = new double[nrOfParameters + 1 + nrOfVars];

		for (size_t i = 0; i < nrOfParameters; i++)
		{
			tmp[i] = p[i];
		}
		tmp[nrOfParameters] = tVals[memorySize - 1];
		for (size_t i = nrOfParameters + 1; i < nrOfParameters + 1 + nrOfVars; i++)
		{
			tmp[i] = xVals[memorySize - 1][i - 1 - nrOfParameters];
		}
	}
	else
	{
		tmp = new double[1 + nrOfVars];

		tmp[0] = tVals[memorySize - 1];
		for (size_t i = 1; i < 1 + nrOfVars; i++)
		{
			tmp[i] = xVals[memorySize - 1][i - 1];
		}
	}

	return tmp;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::~DDESolver()
{
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
double * DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::calculateMesh(double * a, double * b, unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth)
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

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
int DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::calculateLength(unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth)
{
	if (recursionDepth == 1) return sizeA * sizeB + sizeA;
	return calculateLength(sizeA, sizeB, recursionDepth - 1) * (sizeB + 1);
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
double * DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::filter(double * original, unsigned int * nr, double min, double * toRemove , unsigned int nrToRemove, double tol)
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
	return filtered;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::sort(double * lst, unsigned int len)
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

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::allocate()
{
	memorySize = nrOfSteps + nrOfDelays * nrOfInitialPoints * 5;
	tVals = new double[memorySize];
	xVals = new double*[memorySize];
	xdVals = new double*[memorySize];

	for (size_t i = 0; i < memorySize; i++)
	{
		xVals[i] = new double[nrOfVars];
		xdVals[i] = new double[nrOfVars];
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::analyzeInit(bool print)
{
	//phase 1: counting
	unsigned int nrOfC0 = 0; 
	unsigned int nrOfC1 = 1;
	for (size_t i = 1; i < nrOfInitialPoints; i++)
	{
		if (tVals[i] == tVals[i - 1]) //double initial point found
		{
			//categorization of this discontinous point
			for (size_t k = 0; k < nrOfVars; k++)
			{
				if (xVals[i][k] == xVals[i - 1][k] && xdVals[i][k] != xdVals[i - 1][k]) //C1 discontinouity
				{
					nrOfC1++;
					break;
				}

				if (xVals[i][k] != xVals[i - 1][k]) //C0 discontinouity
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
	for (size_t i = 1; i < nrOfInitialPoints; i++)
	{
		if (tVals[i] == tVals[i - 1]) //double initial point found
		{
			//categorization of this discontinous point
			for (size_t k = 0; k < nrOfVars; k++)
			{
				if (xVals[i][k] == xVals[i - 1][k] && xdVals[i][k] != xdVals[i - 1][k]) //C1 discontinouity
				{
					C1disc[c1] = tVals[i];
					c1++;
					break;
				}

				if (xVals[i][k] != xVals[i - 1][k]) //C0 discontinouity
				{
					C0disc[c0] = tVals[i];
					c0++;
					break;
				}
			}
		}
	}

	//phase 3: calculating double points mesh
	double * meshDouble = calculateMesh(C0disc, t0, nrOfC0, nrOfDelays, 1);
	unsigned int meshDoubleLen = calculateLength(nrOfC0, nrOfDelays, 1);
	meshDouble = filter(meshDouble, &meshDoubleLen, tStart);

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
	mesh = calculateMesh(newList, t0, newLen, nrOfDelays, 3);
	meshLen = calculateLength(newLen, nrOfDelays, 3);
	mesh = filter(mesh, &meshLen, tStart);
	sort(mesh, meshLen);

	meshType = new int[meshLen];
	for (size_t i = 0; i < meshLen; i++)
	{
		meshType[i] = 1;
		for (size_t j = 0; j < meshDoubleLen; j++)
		{
			if (mesh[i] == meshDouble[j])
			{
				meshType[i] = 2;
			}
		}
	}

	if (print)
	{
		//print discontinouties
		printf("%zd + %zd discontinouities are found in the initial function:\nC0: ",nrOfC0,nrOfC1);
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
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
unsigned int DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::findIndex(double t, unsigned int delayID)
{
	for (unsigned int i = lastIndex[delayID]; i < memorySize; i++)
	{
		if (t < tVals[i])
		{
			if (i - 2 < 1 << 31) lastIndex[delayID] = i - 2;
			else lastIndex[delayID] = 0;
			return i;
		}
	}
	return 0;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::calculateDelay(double t)
{
	for (size_t i = 0; i < nrOfDelays; i++)
	{
		xDelayed[i] = denseOutput(t-t0[i], varId[i], i);
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::RK4Step(void f(double *, double, double *, double *, double *))
{
	//k1
	calculateDelay(t);
	f(xd, t, x, xDelayed, p);

	//k2
	tTmp = t + 0.5*dtAct;
	calculateDelay(tTmp);
	for (size_t i = 0; i < nrOfVars; i++)
	{
		kSum[i] = xd[i];
		xTmp[i] = x[i] + 0.5*dtAct*xd[i];
	}
	f(kAct, tTmp, xTmp, xDelayed, p);

	//k3
	for (size_t i = 0; i < nrOfVars; i++)
	{
		kSum[i] += 2*kAct[i];
		xTmp[i] = x[i] + 0.5*dtAct*kAct[i];
	}
	f(kAct, tTmp, xTmp, xDelayed, p);

	//k4
	tTmp = t + dtAct;
	calculateDelay(tTmp);
	for (size_t i = 0; i < nrOfVars; i++)
	{
		kSum[i] += 2*kAct[i];
		xTmp[i] = x[i] + dtAct*kAct[i];
	}
	f(kAct, tTmp, xTmp, xDelayed, p);

	//result of step
	for (size_t i = 0; i < nrOfVars; i++)
	{
		kSum[i] += kAct[i];
		x[i] += 1. / 6. * dtAct * kSum[i];
	}
	t += dtAct;

}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::integrate(void f(double *, double, double *, double *, double *))
{
	//initialize dynamic variables
	t = tStart;
	memoryId = nrOfInitialPoints; //memory index
	tVals[memoryId] = t; //save t = 0
	for (size_t i = 0; i < nrOfVars; i++)
	{
		x[i] = xVals[memoryId - 1][i];
		xVals[memoryId][i] = x[i];
	}

	//index prediction initalization
	for (size_t i = 0; i < nrOfDelays; i++)
	{
		lastIndex[i] = 0;
	}
	meshId = 0;
	double minidt;

	while (t <= tEnd)
	{
		memoryId++;

		//assuming a simple step
		int stepType = 0; // 0: simple step, 1: step to reach simple mesh, 2: step to reach double mesh
		dtAct = dt;

		//detecting simple or double mesh
		if (meshId < meshLen && mesh[meshId] < t + dt + 1e-15) //dt should be modified to reach a simple mesh point
		{
			stepType = meshType[meshId];
			if (stepType == 1)
			{
				dtAct = mesh[meshId] - t;
			}			
			if (stepType == 2)
			{
				minidt = pow(10, -(14 - log10(t)));
				//printf("t=%lf\tminidt=%lf\n", t, minidt);
				dtAct = mesh[meshId] - t - minidt;
			}
			meshId++;
		}

		RK4Step(f);

		//save values
		tVals[memoryId] = t;
		for (size_t i = 0; i < nrOfVars; i++)
		{
			xVals[memoryId][i] = x[i];
			xdVals[memoryId-1][i] = xd[i];
		}

		if (stepType == 2) //double mesh points, two values should be saved, here xDelayed comes from the initial function
		{
			memoryId++;
			dtAct = 2*minidt;
			RK4Step(f);
			
			tVals[memoryId] = t;
			for (size_t i = 0; i < nrOfVars; i++)
			{
				xVals[memoryId][i] = x[i];
				xdVals[memoryId - 1][i] = xd[i];
				//f(x',t,x,xdelay,p)
			}
		}


	}

}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
double DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::denseOutput(double t, unsigned int var, unsigned int delayID)
{
	unsigned int step = findIndex(t, delayID) - 1;
	double tb = tVals[step];
	double tn = tVals[step + 1];
	double xb = xVals[step][var];
	double xn = xVals[step + 1][var];
	double xdb = xdVals[step][var];
	double xdn = xdVals[step + 1][var];
	double deltat = tn - tb;
	double theta = (t - tb) / (tn - tb);

	double res = (1 - theta)*xb + theta * xn + theta * (theta - 1)*((1 - 2 * theta)*(xn - xb) + (theta - 1)*deltat*xdb + theta * deltat*xdn);
	return res;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setDt(double dt)
{
	this->dt = dt;
	nrOfSteps = int((tEnd - tStart) / dt);
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setNrOfSteps(unsigned int nrOfSteps)
{
	this->nrOfSteps = nrOfSteps;
	dt = (tEnd - tStart) / double(nrOfSteps);
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setNrOfInitialPoints(unsigned int nr)
{
	this->nrOfInitialPoints = nr;
	allocate();
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setInitialTValues(double * t)
{
	for (size_t i = 0; i < nrOfInitialPoints; i++)
	{
		tVals[i] = t[i];
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setInitialXValues(unsigned int var, double * x, double  *xd)
{
	for (size_t i = 0; i < nrOfInitialPoints; i++)
	{
		xVals[i][var] = x[i];
		xdVals[i][var] = xd[i];
	}
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setDelay(unsigned int nr, unsigned int var, double t0)
{
	this->t0[nr] = t0;
	this->varId[nr] = var;
}

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
void DDESolver<nrOfVars, nrOfDelays, nrOfParameters>::setParameters(double * p)
{
	for (size_t i = 0; i < nrOfParameters; i++)
	{
		this->p[i] = p[i];
	}
}

