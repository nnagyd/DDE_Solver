#include <iostream>
#include "DDESolver.h"
#include "DDESolver.cpp"
#include "DDEInit.h"

//initial functions and its derivatives
double x0(double t, int dir)
{
	if (t < -1 || (t == -1 && dir == -1)) return cos(t);
	if (t < -0.5 || (t == -0.5 && dir == -1)) return sin(t);
	return -t;
}

double xd0(double t, int dir)
{
	if (t < -1 || (t == -1 && dir == -1)) return -sin(t);
	if (t < -0.5 || (t == -0.5 && dir == -1)) return cos(t);
	return -1;
}

double y0(double t, int dir)
{
	return 1;
}

double yd0(double t, int dir)
{
	return 0;
}

void f(double * xd, double t, double *x, double *xdelay, double * p)
{
	xd[0] = -0.2*(xdelay[0]+xdelay[1]*x[1])+sin(t);
	xd[1] = -0.8*(xdelay[2] + xdelay[3]) + 0.04*x[0];
}

int main()
{
	//looping through these steps
	unsigned int * stepList = new unsigned int[10]{32,50,100,110,179,347,783,1000,1934,4124};

	for (size_t i = 0; i < 10; i++)
	{
		unsigned int steps = stepList[i];
		DDESolver<2, 4, 0> solver;

		//set delay
		solver.setDelay(0, 0, 2);
		solver.setDelay(1, 1, 1.3);
		solver.setDelay(2, 0, 1);
		solver.setDelay(3, 1, 0.7);

		//set time interval
		solver.setRange(0.0, 10.0);

		//set number of steps steps
		solver.setNrOfSteps(steps);

		//initial condition generation
		unsigned int nr = 640;
		unsigned int nrOfDisc = 2;
		unsigned int nrOfPoints = nr + 2 * nrOfDisc;
		double * disc = new double[nrOfDisc] {-1, -0.5};

		//calculate initial points
		double * tInit = linspaceDisc(-2.0, 0.0,nr,disc,nrOfDisc);
		double * x0Init = discretize(x0, tInit, nrOfPoints);
		double * xd0Init = discretize(xd0, tInit, nrOfPoints);
		double * y0Init = discretize(y0, tInit, nrOfPoints);
		double * yd0Init = discretize(yd0, tInit, nrOfPoints);

		//set initial points 
		solver.setNrOfInitialPoints(nrOfPoints);
		solver.setInitialTValues(tInit);
		solver.setInitialXValues(0, x0Init, xd0Init);
		solver.setInitialXValues(1, y0Init, yd0Init);

		//create mesh
		solver.analyzeInit(true);
		
		//integrate and save
		solver.integrate(f);
		solver.save("final_t7_" + std::to_string(steps) + ".txt", false);
		delete tInit, x0Init, xd0Init;
	}

}