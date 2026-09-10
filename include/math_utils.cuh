#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <math.h>
#include "config.cuh"
#include "physics.cuh"

__host__ __device__ inline bool is_straight(float radius_m, const CarSetup* setup) {
    if (radius_m >= 10000.0f) { return true; } // its a perfect straight

    // V^2 * (mass/R - 0.5 * rho * Cl * A * mu) = mass * g * mu
    float cl_a = (setup->drag_coef * 3.0f) * Config::FRONTAL_AREA;
    float aero_term = 0.5f * Config::AIR_DENSITY * cl_a * Config::BASE_MECH_GRIP;
    float mechanical_term = setup->mass_kg / radius_m;

    if (mechanical_term <= aero_term) { return true; } // if the aero part is faster than centrifugal force the car sticks

    float max_v_sq = (setup->mass_kg * Config::GRAVITY * Config::BASE_MECH_GRIP) / (mechanical_term - aero_term);

    // 97.5m/s -> 351km/h
    return max_v_sq > (97.5f * 97.5f);
}

__host__ __device__ inline float get_track_pitch_angle(const TrackSegment* track, int current_seg, int num_segments) {
    int next_seg = (current_seg + 1) % num_segments;
    float delta_z = track[next_seg].z - track[current_seg].z;
    float seg_length = track[current_seg].length_m; // this is always 1.0f;

    if (seg_length<=0.0f) return 0.0f;

    float ratio = delta_z / seg_length;
    if (ratio>1.0f) ratio = 1.0f;
    if (ratio<-1.0f)ratio = -1.0f;

    // asinf is a CUDA functon to calculate arcsen of a float number
    return asinf(ratio);
}

#endif