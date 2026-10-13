#ifndef SIM_DRIVER_CUH
#define SIM_DRIVER_CUH

#include "config.cuh"
#include "physics.cuh"
#include "chassis.cuh"
#include "engine.cuh"
#include "ers.cuh"

// getting max deceleration
__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup, float base_mu, float total_mass, float future_radius);

// getting allowed speed
__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments);

// updating driver pedals
__host__ __device__ void update_driver_pedals(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float target_speed, float dt);

#endif