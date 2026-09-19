#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <math.h>
#include "config.cuh"
#include "physics.cuh"
#include "tyres.cuh"

__host__ __device__ inline bool is_straight(float radius_m, const CarSetup* setup) {
    /*
    if (radius_m >= 10000.0f) { return true; } // its a perfect straight

    // V^2 * (mass/R - 0.5 * rho * Cl * A * mu) = mass * g * mu
    float cl_a = (setup->drag_coef * 3.0f) * Config::FRONTAL_AREA;
    float aero_term = 0.5f * Config::AIR_DENSITY * cl_a * Config::BASE_MECH_GRIP;
    float mechanical_term = setup->mass_kg / radius_m;

    if (mechanical_term <= aero_term) { return true; } // if the aero part is faster than centrifugal force the car sticks
    
    float denom = mechanical_term - aero_term; if (denom < 1e-5f) {return true}

    float max_v_sq = (setup->mass_kg * Config::GRAVITY * Config::BASE_MECH_GRIP) / denom;

    // 97.5m/s -> 351km/h
    return max_v_sq > (97.5f * 97.5f);
    */
    return radius_m >= 800.f;               // any corner radius above 800m is effectively a straight
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

    // Track pitch angle theta = arcsin(delta_z / delta_s)
    float pitch_angle = get_track_pitch_angle(track, car->current_seg, num_segments);

    // Dynamic drag reduction when rear wing slot gap is open
    // F_drag = 0.5 * rho * v^2 * Cd * A
    float current_drag_coef = setup->drag_coef;
    if (car->drs_open) {
        current_drag_coef *= 0.65f;
    }
    dynamics.drag_force = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * current_drag_coef * Config::FRONTAL_AREA;
    
    // F_downforce = 0.5 * rho * v^2 * Cl * A (where Cl * A ~= 3.0 * Cd * A)
    float cl_a = (setup->drag_coef * 3.0f) * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * cl_a;

    // Base static gravity load perpendicular to the road: F_normal,static = m * g * cos(theta)
    float static_normal = setup->mass_kg * Config::GRAVITY * cosf(pitch_angle);
    float normal_force = static_normal + downforce;
    if (normal_force < 0.0f) normal_force = 0.0f;

    // Longitudinal Weight Transfer: Delta_Fz = (m * a * h_cg) / L
    // car->a > 0 (acceleration) shifts weight to rear tires (+Delta_Fz)
    // car->a < 0 (braking) shifts weight to front tires (-Delta_Fz)
    float weight_transfer = (setup->mass_kg * car->a * Config::GRAVITY_CENTER_HEIGHT) / Config::WHEEL_BASE;

    // F1 static distribution (~45% front / ~55% rear) + aero balance (~40% front / ~60% rear)
    float front_force = (0.45f * static_normal) + (0.40f * downforce) - weight_transfer;
    float rear_force  = (0.55f * static_normal) + (0.60f * downforce) + weight_transfer;

    if (front_force < 0.0f) front_force = 0.0f;
    if (rear_force < 0.0f) rear_force = 0.0f;

    // Dynamic rear load ratio replacing static 0.55 constant
    float rear_grip_ratio = (normal_force > 0.0f) ? (rear_force / normal_force) : 0.55f;
    if (rear_grip_ratio > 0.85f) rear_grip_ratio = 0.85f;
    if (rear_grip_ratio < 0.15f) rear_grip_ratio = 0.15f;

    // Maximum friction circle radius: F_grip,max = F_normal * the grip of the car according to tyres grip
    dynamics.max_grip = normal_force * calculate_effective_grip(car);

    // Centrifugal cornering load: F_lat = (m * v^2) / R
    dynamics.lateral_force = (setup->mass_kg * car->v * car->v) / track[car->current_seg].radius_m;

    // Kamm Circle: F_long,grip = sqrt(F_grip,max^2 - F_lat^2)
    if (dynamics.max_grip > dynamics.lateral_force) {
        dynamics.long_grip = sqrtf(dynamics.max_grip * dynamics.max_grip - dynamics.lateral_force * dynamics.lateral_force);
    } else {
        dynamics.long_grip = 0.0f;
    }

    // Maximum traction force limited by dynamic rear vertical load
    dynamics.max_traction_force = dynamics.long_grip * rear_grip_ratio;

    // Longitudinal gravity: F_gravity,long = -m * g * sin(theta)
    dynamics.gravity_longitudinal = -setup->mass_kg * Config::GRAVITY * sinf(pitch_angle);
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