#include <immintrin.h>
#include <iostream>
#include <assert.h>
#include <chrono>

constexpr uint32_t INT_OPS = 256 / (sizeof(int) * 8); // 256-bit wide register / sizeof float in bits
const uint32_t TEST_ARR_SIZE = 65536;

void addVectorTest(const int32_t*, const int32_t*, int32_t*);
void addScalarTest(const int32_t*, const int32_t*, int32_t*);

void fillArrays(int32_t*, int32_t*);

int main()
{
	int a[TEST_ARR_SIZE];
	int b[TEST_ARR_SIZE];
	int result[TEST_ARR_SIZE];

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

void fillArrays(int32_t* arrA, int32_t* arrB)
{
	for (int i = 0; i < TEST_ARR_SIZE; ++i)
	{
		arrA[i] = i;
		arrB[i] = 1.1f;
	}
}

void addScalarTest(const int32_t* a, const int32_t* b, int32_t* result)
{
	for (int i = 0; i < TEST_ARR_SIZE; i++) // scalar operation, add 1 pair at a time
	{
		result[i] = a[i] + b[i];
	}
}

void addVectorTest(const int32_t* a, const int32_t* b, int32_t* result)
{
	__m256i c, d;
	for (int i = 0; i < TEST_ARR_SIZE; i += INT_OPS) // Improvement, do scalar addition if we don't have enough elements for vector in some other conditional
	{
		c = _mm256_loadu_si256((__m256i*) & a[i]); // loadu will load 8*32-bit floats at a time
		d = _mm256_loadu_si256((__m256i*) & b[i]);

		_mm256_storeu_si256((__m256i*) & result[i], _mm256_add_epi32(c, d)); //SIMD add operation (8*32-bit floats simultaneously) + store in results arr
	}
}