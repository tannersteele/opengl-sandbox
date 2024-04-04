#include <immintrin.h>
#include <iostream>
#include <assert.h>


void addVectorTest(const float* a, const float* b, float* result, const int n);

int main()
{
	const int n = 16;
	float a[n] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
	float b[n] = { 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f};

	assert((sizeof(a) / sizeof(a[0])) == n && (sizeof(b) / sizeof(b[0])) == n, "Vector arrays must be the same size");

	float result[n];
	addVectorTest(a, b, result, n);

	for (int i = 0; i < n; ++i)
	{
		std::cout << result[i] << std::endl;
	}

	return 0;
}

void addVectorTest(const float* a, const float* b, float* result, const int n)
{
	for (int i = 0; i < n; i += 8) // 8 * 32-bit float = 256 bit wide instructions
	{
		__m256 avx_a = _mm256_loadu_ps(&a[i]); // loadu will load 8 32-bit floats at a time
		__m256 avx_b = _mm256_loadu_ps(&b[i]);

		__m256 avx_result = _mm256_add_ps(avx_a, avx_b);

		_mm256_storeu_ps(&result[i], avx_result);
	}
}