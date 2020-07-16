#ifndef GPU_TIMER_
#define GPU_TIMER_

#include <chrono>

//------------------------------CUDA CHECK MACRO----------------------------------------
#define CHECK(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
  printf("Error: %s:%d, ", __FILE__, __LINE__); \
  printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
  exit(-10*error); \
  } \
}


//------------------------------ Timer stuff -------------------------------------------
uint64_t micros()
{
	uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::
		now().time_since_epoch()).count();
	return us;
}

inline double seconds()
{
	return double(micros()) / 1000. / 1000.;
}


#endif
