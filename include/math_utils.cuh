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

__host__ __device__ inline F1CarDynamics calculate_car_dynamics(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    F1CarDynamics dynamics;

    float pitch_angle = get_track_pitch_angle(track, car->current_seg, num_segments);

    dynamics.drag_force = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * setup->drag_coef * Config::FRONTAL_AREA;

    float downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;

    float normal_force = (setup->mass_kg * Config::GRAVITY * cosf(pitch_angle)) + downforce;
    if (normal_force < 0.0f) normal_force = 0.0f;

    dynamics.max_grip = normal_force * Config::BASE_MECH_GRIP;

    // Kamm Circle: Lateral Force: F * v² / R
    dynamics.lateral_force = (setup->mass_kg * car->v * car->v) / track[car->current_seg].radius_m;
    // pythagorean theorem: sqrt(F² + L²) = max_grip
    if (dynamics.max_grip > dynamics.lateral_force) {
        dynamics.long_grip = sqrtf(dynamics.max_grip * dynamics.max_grip - dynamics.lateral_force * dynamics.lateral_force);
    }
    dynamics.gravity_longitudinal = -setup->mass_kg * Config::GRAVITY * sinf(pitch_angle);

    dynamics.max_traction_force = dynamics.long_grip * 0.55f;

    dynamics.engine_breaking_force = (car->rpm / Config::RPM_REDLINE) * Config::MAX_ENGINE_BRAKING;

    return dynamics;
}

__host__ __device__ inline float compute_net_force(const F1Car* car, const CarSetup* setup, const F1CarDynamics& dynamics) {
    float net_force = 0.0f;
    float engine_braking_force = (car->rpm / Config::RPM_REDLINE) * Config::MAX_ENGINE_BRAKING;

    switch (car->action) {
        case DriverAction::BRAKE: {
            float applied_brake_force = car->brake_pedal * dynamics.desired_braking_force;
            
            net_force = -applied_brake_force - dynamics.drag_force - engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::COAST: {
            net_force = -dynamics.drag_force - engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::ACCELERATE: {
            float applied_engine_force = car->throttle_pedal * dynamics.desired_engine_force;
            
            net_force = applied_engine_force - dynamics.drag_force + dynamics.gravity_longitudinal;
            break;
        }
    }
    
    return net_force;
}
#endif