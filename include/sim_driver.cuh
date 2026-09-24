#ifndef SIM_DRIVER_CUH
#define SIM_DRIVER_CUH

#include "config.cuh"
#include "physics.cuh"
#include "math_utils.cuh"
#include "engine.cuh"

__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup, float base_mu);
__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments);
__host__ __device__ float calculate_mguk_deployment(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments);
__host__ __device__ void update_ers(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt);
__host__ __device__ void update_driver_pedals(F1Car* car, F1CarDynamics& dynamics, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt);

#endif