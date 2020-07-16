#ifndef DDE_INIT_
#define DDE_INIT_

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

double * linspaceDisc(double t0, double t1, unsigned int nr, double * tDisc = NULL, unsigned int nrOfDisc = 0, double eps = 0.0)
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

#endif
