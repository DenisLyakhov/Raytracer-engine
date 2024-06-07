#pragma once
#include <cmath>
#include "Random.h"

#define EPSILON 1e-8

class Vector {
public:
	float x;
	float y;
	float z;

	__host__ __device__ Vector() {
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}

	__host__ __device__ Vector(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__host__ __device__ inline Vector copy() {
		return Vector(this->x, this->y, this->z);
	}

	__host__ __device__ inline Vector operator*(float t) {
		return Vector(this->x * t, this->y * t, this->z * t);
	}

	__host__ __device__ inline Vector operator*(const Vector& vector) {
		return Vector(this->x * vector.x,
			this->y * vector.y,
			this->z * vector.z);
	}

	__host__ __device__ inline Vector operator/(const Vector& vector) {
		return Vector(this->x / vector.x,
			this->y / vector.y,
			this->z / vector.z);
	}

	__host__ __device__ inline Vector operator/(float t) {
		return this->copy() * (1 / t);
	}

	__host__ __device__ inline Vector operator+(const Vector& vector) {
		return Vector(this->x + vector.x,
			this->y + vector.y,
			this->z + vector.z);
	}

	__host__ __device__ inline Vector operator+(float t) const {
		return Vector(this->x + t,
			this->y + t,
			this->z + t);
	}

	__host__ __device__ inline Vector operator-(float t) const {
		return Vector(this->x - t,
			this->y - t,
			this->z - t);
	}

	__host__ __device__ inline Vector operator-(const Vector& vector) {
		return Vector(this->x - vector.x,
			this->y - vector.y,
			this->z - vector.z);
	}

	__host__ __device__ inline Vector operator-() {
		return Vector(-this->x, -this->y, -this->z);
	}

	__host__ __device__ inline float getLength() const {
		return std::sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}

	__host__ __device__ inline float scalarProduct(Vector vector) {
		return this->x * vector.x + this->y * vector.y + this->z * vector.z;
	}

	__host__ __device__ inline Vector getNormalizedVector() {
		return this->copy() / getLength();
	}

	__host__ __device__ bool isZero() {
		return (fabs(x) < EPSILON) && (fabs(y) < EPSILON) && (fabs(z) < EPSILON);
	}

	__host__ __device__ inline Vector crossProduct(Vector other) {
		return Vector(
			this->y * other.z - this->z * other.y,
			this->z * other.x - this->x * other.z,
			this->x * other.y - this->y * other.x
		);
	}

	__host__ __device__ void print() {
		printf("[%f, %f, %f]", x, y, z);
	}

	__host__ __device__ float getMaxValue() {
		float result = this->x;

		if (this->y > result) {
			result = this->y;
		}

		if (this->z > result) {
			result = this->z;
		}

		return result;
	}

	__host__ __device__ float getMinValue() {
		float result = this->x;

		if (this->y < result) {
			result = this->y;
		}

		if (this->z < result) {
			result = this->z;
		}

		return result;
	}
};

__host__ __device__ inline Vector operator+(float t, const Vector& vector) {
	return vector + t;
}




