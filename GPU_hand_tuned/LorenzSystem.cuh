#ifndef LORENZ_SYSTEM_
#define LORENZ_SYSTEM_

#include <cmath>

//------------------------------Global constants----------------------------------------
const double PI = 3.14159265358979323846264338327950;

//------------------------------ ODE Function --------------------------------
__forceinline__ __device__ void f(double * __restrict__ xd, const double t, const double * __restrict__ x, const double xDelay, const double p)
{
	xd[0] = 10.0*(xDelay - x[1]);
	xd[1] = p * x[0] - x[1] - x[2] * x[0];
	xd[2] = x[0] * x[1] - 8.0/3.0 * x[2];
}

//------------------------------ Initial functions --------------------------------
double x0(double t, int dir)
{
	return -8 ;
}
double xd0(double t, int dir)
{
	return 0;
}
double y0(double t, int dir)
{
	return -8 + sin(2* PI * t);
}
double yd0(double t, int dir)
{
	return 2*PI*cos(2 * PI * t);
}
double z0(double t, int dir)
{
	return -8;
}
double zd0(double t, int dir)
{
	return 0;
}



#endif
