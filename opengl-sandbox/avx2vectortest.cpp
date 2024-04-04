#include <immintrin.h>
#include <iostream>
#include <assert.h>

constexpr int FLOAT_OPS = 256 / (sizeof(float) * 8); // 256-bit wide register / sizeof float in bits
const int TEST_ARR_SIZE = 65536;

void addVectorTest(const float*, const float*, float*);
void fillVectorArray(float*, float*);

int main()
{
	float a[TEST_ARR_SIZE];
	float b[TEST_ARR_SIZE];
	float result[TEST_ARR_SIZE];

	fillVectorArray(a, b);
	addVectorTest(a, b, result);

	return 0;
}

void fillVectorArray(float* arrA, float* arrB)
{
	for (int i = 0; i < TEST_ARR_SIZE; ++i)
	{
		arrA[i] = i;
		arrB[i] = 1.1f;
	}
}

void addVectorTest(const float* a, const float* b, float* result)
{
	for (int i = 0; i < TEST_ARR_SIZE; i += FLOAT_OPS)
	{
		__m256 avx_a = _mm256_loadu_ps(&a[i]); // loadu will load 8 32-bit floats at a time
		__m256 avx_b = _mm256_loadu_ps(&b[i]);

		_mm256_storeu_ps(&result[i], _mm256_add_ps(avx_a, avx_b));
	}
}