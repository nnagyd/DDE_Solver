#include <iostream>
#include "DDESolver.h"
#include "DDESolver.cpp"


double * discretize(double f(double t, int dir), double * tVals, unsigned int nrOfPoints);
double * linspace(double t0, double t1, unsigned int nr);
double * linspaceDisc(double t0, double t1, unsigned int nr, double * tDisc, unsigned int nrOfDisc, double eps = 0);
void printArray(double *t, unsigned int count)
{
	for (size_t i = 0; i < count; i++)
	{
		printf("i=%d\t%4.2lf\n", i, t[i]);
	}
}


double x00(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -1.1 || (t == -1.1 && direction == -1)) return cos(t);
	if (t < -0.5 || (t == -0.5 && direction == -1)) return sin(t);
	return -t;
}

double xd00(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -1.1 || (t == -1.1 && direction == -1)) return -sin(t);
	if (t < -0.5 || (t == -0.5 && direction == -1)) return cos(t);
	return -1;
}

double x10(double t, int)
{
	return 1;
}

double xd10(double t, int)
{
	return 0;
}

double x20(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -0.55 || (t == -0.55 && direction == -1)) return exp(t);
	return -t;
}

double xd20(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -0.55 || (t == -0.55 && direction == -1)) return exp(t);
	return -1;
}

double x30(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -1.5 || (t == -1.5 && direction == -1)) return cos(t-1.2);
	if (t < -0.9 || (t == -0.9 && direction == -1)) return 2*sin(4*t);
	if (t < -0.5 || (t == -0.5 && direction == -1)) return 0.5;
	return -t;
}

double xd30(double t, int direction) //-1: negativ, 1: positiv
{
	if (t < -1.5 || (t == -1.5 && direction == -1)) return -sin(t-1.2);
	if (t < -0.9 || (t == -0.9 && direction == -1)) return 8*cos(4*t);
	if (t < -0.5 || (t == -0.5 && direction == -1)) return 0;
	return -1;
}

//f(x',t,x,xdelay,p)
void f(double * xd, double t, double * x, double * xdelay, double *)
{
	xd[0] = -0.2*(xdelay[0] + xdelay[1]*x[1])+sin(t)-x[3]*x[3];
	xd[1] = -0.8*(xdelay[2] + xdelay[3])+0.04*x[0]+x[3];
	xd[2] = -0.1*(xdelay[4] + xdelay[5]) + 0.04*xdelay[6];
	xd[3] = -0.2*(xdelay[2] + xdelay[7])*(xdelay[2] + xdelay[7]) + 0.01*x[0];
}


int main()
{
	unsigned int * stepList = new unsigned int[7]{ 110,179,347,783,1000,1934,4124 };
	
	for (size_t i = 0; i < 7; i++)
	{
		unsigned int steps = stepList[i];
		DDESolver<4, 8, 0> solver;

		solver.setDelay(0, 0, 2);
		solver.setDelay(1, 1, 1.3);
		solver.setDelay(2, 0, 1);
		solver.setDelay(3, 1, 0.7);
		solver.setDelay(4, 0, 0.37);
		solver.setDelay(5, 3, 0.7);
		solver.setDelay(6, 3, 2.2);
		solver.setDelay(7, 2, 0.678);
		solver.setRange(0.0, 5.5);
		solver.setNrOfSteps(steps);

		unsigned int nr = 100;
		unsigned int nrOfDisc = 6;
		unsigned int nrOfInit = nr + 2 * nrOfDisc;
		double * disc = new double[nrOfDisc] {-0.5,-1.0,-1.5,-0.9,-1.1,-0.55};



		double * tInit = linspaceDisc(-2.3, 0.0, nr, disc, nrOfDisc);
		double * x0Init = discretize(x00, tInit, nrOfInit);
		double * xd0Init = discretize(xd00, tInit, nrOfInit);		
		double * x1Init = discretize(x10, tInit, nrOfInit);
		double * xd1Init = discretize(xd10, tInit, nrOfInit);		
		double * x2Init = discretize(x20, tInit, nrOfInit);
		double * xd2Init = discretize(xd20, tInit, nrOfInit);		
		double * x3Init = discretize(x30, tInit, nrOfInit);
		double * xd3Init = discretize(xd30, tInit, nrOfInit);

		solver.setNrOfInitialPoints(nrOfInit);
		solver.setInitialTValues(tInit);
		solver.setInitialXValues(0, x0Init, xd0Init);
		solver.setInitialXValues(1, x1Init, xd1Init);
		solver.setInitialXValues(2, x2Init, xd2Init);
		solver.setInitialXValues(3, x3Init, xd3Init);



		solver.analyzeInit();

		solver.integrate(f);
		solver.save("C:\\Users\\nnagy\\Documents\\ODE HDS\\DDE\\Adatok\\ver4_t9_" + std::to_string(steps) + ".txt", false);
	}
}


double * discretize(double f(double t, int dir), double * tVals, unsigned int nrOfPoints)
{
	double * lst = new double[nrOfPoints];
	
	for (size_t i = 0; i < nrOfPoints; i++)
	{
		int dir = -1;
		if (i >= 1 && tVals[i] == tVals[i - 1])
		{
			dir = 1;
		}
		lst[i] = f(tVals[i], dir);
		//printf("i=%3zd\tt=%5.3lf\tlst=%5.3lf\tdir=%d\n", i, tVals[i], lst[i],dir);
	}
	return lst;
}


double * linspace(double t0, double t1, unsigned int nr)
{
	double * lst = new double[nr];
	double dt = (t1 - t0) / (nr - 1);
	double t = t0;
	for (size_t i = 0; i < nr; i++)
	{
		lst[i] = t;
		t += dt;
	}
	return lst;
}


double * linspaceDisc(double t0, double t1, unsigned int nr, double * tDisc, unsigned int nrOfDisc, double eps)
{
	double * lst = new double[nr + 2 * nrOfDisc];
	int * discMask = new int[nrOfDisc];
	//set all element to 0, set to 1 if the i. discontinouity is included 
	for (size_t i = 0; i < nrOfDisc; i++)
	{
		discMask[i] = 0;
	}

	double dt = (t1 - t0) / (nr - 1);
	double t = t0;
	for (size_t i = 0; i < nr + 2 * nrOfDisc; i++)
	{
		bool set = true;
		for (size_t j = 0; j < nrOfDisc; j++)
		{

			if (!discMask[j] && fabs(tDisc[j] - t) < eps) //discontinuity happens at a point
			{
				lst[i] = tDisc[j] - eps;
				lst[i + 1] = tDisc[j];
				lst[i + 2] = tDisc[j] + eps;
				set = false;
				discMask[j] = 1;
				i += 2;
			}
			else if (!discMask[j] && tDisc[j] < t) //discontinuity far from point
			{
				lst[i] = tDisc[j] - eps;
				lst[i + 1] = tDisc[j] + eps;
				discMask[j] = 1;
				i += 2;
			}
		}
		if (set) lst[i] = t;
		t += dt;
	}
	return lst;
}