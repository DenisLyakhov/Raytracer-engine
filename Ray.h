#pragma once
#include "Vector.h"

// Class Ray representing a 3D line in space
// location = origin + t*vector
class Ray {
public:
	Vector origin;
	Vector vector;

	__device__ Ray(Vector origin, Vector vector) {
		this->origin = origin;
		this->vector = vector;
	}

	__device__ inline Vector getLocation(float t) {
		return origin + (vector * t);
	}

	__device__ Ray reflect(Vector normal, Vector newOrigin) {
		Vector reflectedVector = vector - normal * vector.scalarProduct(normal) * 2;
		return Ray(
			newOrigin,
			reflectedVector
		);
	}

	__device__ Ray refract(Vector normal, Vector newOrigin, float refractiveIndex, curandState* randState) {
		Vector normalizedVector = vector.getNormalizedVector();

		float angleCos = (-normalizedVector).scalarProduct(normal);
		if (angleCos > 1.0) {
			angleCos = 1.0;
		}

		bool canRefract = refractiveIndex * sqrt(1.0 - angleCos * angleCos) <= 1.0;

		// If outside the object (or on the border (depending on reflectance index)), then refract
		if (canRefract && getReflectanceIndex(angleCos, refractiveIndex) <= randomFloat(randState)) {
			Vector refractedVectorPerpendicular = (normalizedVector + normal * angleCos) * refractiveIndex;
			Vector refractedVectorParallel = normal * (-sqrt(fabs(1.0 - refractedVectorPerpendicular.scalarProduct(refractedVectorPerpendicular))));

			return Ray(newOrigin, refractedVectorPerpendicular + refractedVectorParallel);
		}

		return reflect(normal, newOrigin);
	}

	// Schlick's approximation for determining reflectance index on the border of two media
	__device__ float getReflectanceIndex(float angleCos, float refractionIndex) {
		double R0 = (1 - refractionIndex) / (1 + refractionIndex);
		R0 = R0 * R0;
		// TODO: Remove pow function
		float a = (1 - angleCos);
		return R0 + (1 - R0) * a*a*a*a*a;
	}
};
