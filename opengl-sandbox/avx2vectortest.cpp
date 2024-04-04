#include <immintrin.h>
#include <iostream>
#include <assert.h>
#include <chrono>

constexpr int FLOAT_OPS = 256 / (sizeof(float) * 8); // 256-bit wide register / sizeof float in bits
const int TEST_ARR_SIZE = 65536;

void addVectorTest(const float*, const float*, float*);
void addScalarTest(const float*, const float*, float*);

void fillArrays(float*, float*);

int main()
{
	float a[TEST_ARR_SIZE];
	float b[TEST_ARR_SIZE];
	float result[TEST_ARR_SIZE];

	// Vector Add ------------------------
	fillArrays(a, b);

	auto startTime = std::chrono::high_resolution_clock::now();
	addVectorTest(a, b, result);
	auto endTime = std::chrono::high_resolution_clock::now();
	std::cout << "Vector add timing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() << "ns" << std::endl;

	// Scalar Add ------------------------
	fillArrays(a, b); // Reset to produce similar results
	
	startTime = std::chrono::high_resolution_clock::now();
	addScalarTest(a, b, result);
	endTime = std::chrono::high_resolution_clock::now();

	std::cout << "Scalar add timing " << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() << "ns" << std::endl;
	return 0;
}

void fillArrays(float* arrA, float* arrB)
{
	for (int i = 0; i < TEST_ARR_SIZE; ++i)
	{
		arrA[i] = i;
		arrB[i] = 1.1f;
	}
}

void addScalarTest(const float* a, const float* b, float* result)
{
	for (int i = 0; i < TEST_ARR_SIZE; i++) // scalar operation, add 1 pair at a time
	{
		result[i] = a[i] + b[i];
	}
}

void addVectorTest(const float* a, const float* b, float* result)
{
	for (int i = 0; i < TEST_ARR_SIZE; i += FLOAT_OPS)
	{
		__m256 avx_a = _mm256_loadu_ps(&a[i]); // loadu will load 8*32-bit floats at a time
		__m256 avx_b = _mm256_loadu_ps(&b[i]);

		_mm256_storeu_ps(&result[i], _mm256_add_ps(avx_a, avx_b)); //SIMD add operation (8*32-bit floats simultaneously) + store in results arr
	}
}