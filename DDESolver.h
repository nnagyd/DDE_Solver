#pragma once
#include <string>

template<unsigned int nrOfVars, unsigned int nrOfDelays, unsigned int nrOfParameters>
class DDESolver
{
private:
	//user given inside variables
	unsigned int  nrOfInitialPoints, nrOfSteps;

	//calculated inside variables
	unsigned int memorySize, nrOfDisc;

	//inside counter
	unsigned int memoryId;

	//variable for index prediction to every delay
	unsigned int * lastIndex;

	//integration temporary variables
	double *kAct, *kSum, *xTmp, *xDelayed, tTmp;

	//dynamic integration variable
	double *x, *xd, t, dtAct;

	//integration settings
	double tStart, tEnd, dt;

	//delays
	double * t0;
	unsigned int * varId;

	//parameters
	double *p;

	//mesh points and stepsize correction
	unsigned int meshLen, meshId;
	double * mesh;
	int * meshType;

	//memory
	double **xVals, **xdVals, *tVals;
public:
	DDESolver();

	//initial functions
	void setRange(double tStart, double tEnd) { this->tStart = tStart; this->tEnd = tEnd; }
	void setDt(double dt);
	void setNrOfSteps(unsigned int nrOfSteps);
	void setNrOfInitialPoints(unsigned int nr);
	void setInitialTValues(double *t);
	void setInitialXValues(unsigned int var, double * x, double *xd);
	void setDelay(unsigned int nr, unsigned int var, double t0);
	void setParameters(double * p);
	void analyzeInit(bool print);

	//dense output
	double denseOutput(double t, unsigned int var, unsigned int delayID);
	unsigned int findIndex(double t, unsigned int delayID);
	
	//integrate
	void calculateDelay(double t);
	void RK4Step(void f(double*, double, double*, double*, double*));
	void integrate(void f(double*, double, double*, double*, double*)); //f(x',t,x,xdelay,p)

	//save function
	void save(std::string fn, bool saveStart = true);
	double * saveEnd(bool includePar = true);

	~DDESolver();


private: //inside functions
	//dealing with discontinouities
	double * calculateMesh(double * a, double * b, unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth = 2);
	int calculateLength(unsigned int sizeA, unsigned int sizeB, unsigned int recursionDepth = 2);
	double * filter(double * original, unsigned int * nr, double min = 0, double * toRemove = NULL, unsigned int nrToRemove = 0, double tol = 1e-12);
	void sort(double *lst, unsigned int len);
	void allocate();
};

