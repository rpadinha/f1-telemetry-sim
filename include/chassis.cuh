#ifndef CHASSIS_CUH
#define CHASSIS_CUH

#include <math.h>
#include "config.cuh"
#include "physics.cuh"
#include "engine.cuh"
#include "tyres.cuh"
#include "aero.cuh"

__host__ __device__ inline bool is_straight(float radius_m) {
    return radius_m >= 1800.f;
}

__host__ __device__ inline F1CarDynamics calculate_car_dynamics(const F1Car* car, const CarSetup* setup, const TrackSegment* track_seg) {
    F1CarDynamics dynamics;

    // Full weight
    dynamics.total_mass = car->fuel_kg + setup->mass_kg;

    dynamics.pitch_angle = track_seg->pitch_angle;
    
    // Calcula Aero (Drag e Downforce preenchem a struct)
    calculate_aerodynamics(&dynamics, car->v, setup, car->drs_open);
    
    // Base static gravity load perpendicular to the road: F_normal,static = m * g * cos(theta)
    float static_normal = dynamics.total_mass * Config::GRAVITY * cosf(dynamics.pitch_angle);
    dynamics.normal_force = static_normal + dynamics.downforce;
    if (dynamics.normal_force < 0.0f) dynamics.normal_force = 0.0f;

    // Longitudinal Weight Transfer: Delta_Fz = (m * a * h_cg) / L
    float weight_transfer_long = (dynamics.total_mass * car->a * Config::GRAVITY_CENTER_HEIGHT) / Config::WHEEL_BASE;

    // F1 static distribution (~45% front / ~55% rear) + aero balance (~40% front / ~60% rear)
    float front_force = (0.45f * static_normal) + (0.40f * dynamics.downforce) - weight_transfer_long;
    float rear_force  = (0.55f * static_normal) + (0.60f * dynamics.downforce) + weight_transfer_long;
    if (front_force < 0.0f) front_force = 0.0f;
    if (rear_force < 0.0f) rear_force = 0.0f;

    float radius = track_seg->radius_m;
    float lat_dir = track_seg->turn_direction;

    // Centrifugal cornering load: F_lat = (m * v^2) / R
    // radius might be making bad calculations
    dynamics.lateral_force = (dynamics.total_mass * (car->v * car->v)) / radius;

    // assuming track width of 1.6m
    float weight_transfer_lat = (dynamics.lateral_force * Config::GRAVITY_CENTER_HEIGHT) / 1.6f;
    float lat_transfer_front = 0.5f * weight_transfer_lat;
    float lat_transfer_rear  = 0.5f * weight_transfer_lat;

    dynamics.load_fl = (front_force * 0.5f) + (lat_transfer_front * lat_dir);
    dynamics.load_fr = (front_force * 0.5f) - (lat_transfer_front * lat_dir);
    dynamics.load_rl = (rear_force * 0.5f)  + (lat_transfer_rear * lat_dir);
    dynamics.load_rr = (rear_force * 0.5f)  - (lat_transfer_rear * lat_dir);
    if (dynamics.load_fl < 0.0f) dynamics.load_fl = 0.0f;
    if (dynamics.load_fr < 0.0f) dynamics.load_fr = 0.0f;
    if (dynamics.load_rl < 0.0f) dynamics.load_rl = 0.0f;
    if (dynamics.load_rr < 0.0f) dynamics.load_rr = 0.0f;

    float nominal_load_per_tyre = (dynamics.total_mass * Config::GRAVITY) * 0.25f;

    float mu_fl = apply_load_sensitivity(calculate_tyre_grip(car->current_compound, car->tyre_temp_fl, car->tyre_wear_fl), dynamics.load_fl, nominal_load_per_tyre);
    float mu_fr = apply_load_sensitivity(calculate_tyre_grip(car->current_compound, car->tyre_temp_fr, car->tyre_wear_fr), dynamics.load_fr, nominal_load_per_tyre);
    float mu_rl = apply_load_sensitivity(calculate_tyre_grip(car->current_compound, car->tyre_temp_rl, car->tyre_wear_rl), dynamics.load_rl, nominal_load_per_tyre);
    float mu_rr = apply_load_sensitivity(calculate_tyre_grip(car->current_compound, car->tyre_temp_rr, car->tyre_wear_rr), dynamics.load_rr, nominal_load_per_tyre);

    float grip_fl = dynamics.load_fl * mu_fl;
    float grip_fr = dynamics.load_fr * mu_fr;
    float grip_rl = dynamics.load_rl * mu_rl;
    float grip_rr = dynamics.load_rr * mu_rr;

    dynamics.max_grip = grip_fl + grip_fr + grip_rl + grip_rr;

    /* protection against the physics limit, sbins and lateral force hits the limit
    if (dynamics.lateral_force > dynamics.max_grip) {
        dynamics.lateral_force = dynamics.max_grip;
    }
    */
   
    dynamics.slip_angle_rad = calculate_slip_angle(dynamics.lateral_force, dynamics.max_grip);
    dynamics.cornering_drag = dynamics.lateral_force * sinf(dynamics.slip_angle_rad);

    // Kamm Circle: Flong_grip = sqrtf(F_gripmax² - Flat²)
    float long_grip_diff = (dynamics.max_grip * dynamics.max_grip) - (dynamics.lateral_force * dynamics.lateral_force);
    dynamics.long_grip = sqrtf(fmaxf(0.0f, long_grip_diff));

    // For F1 only Rear tyres accel so f1 is rwd so grip all rear lmao
    float rear_max_grip = grip_rl + grip_rr;
    float rear_lat_ratio = ((dynamics.load_rl + dynamics.load_rr) / dynamics.normal_force);
    float rear_lateral_force = dynamics.lateral_force * rear_lat_ratio;

    // Maximum traction force limited by dynamic rear vertical load
    float max_traction_diff = (rear_max_grip * rear_max_grip) - (rear_lateral_force * rear_lateral_force);
    dynamics.max_traction_force = sqrtf(fmaxf(0.0f, max_traction_diff));

    // Longitudinal gravity: F_gravity,long = -m * g * sin(theta)
    dynamics.gravity_longitudinal = -dynamics.total_mass * Config::GRAVITY * sinf(dynamics.pitch_angle);

    // Initialize action-dependent forces to zero or to the correct value before driver step
    dynamics.engine_braking_force = get_engine_braking_force(car->rpm);
    dynamics.applied_long_force = 0.0f;
    dynamics.desired_engine_force = 0.0f;
    
    return dynamics;
}

__host__ __device__ inline float compute_net_force(const F1Car* car) {
    const F1CarDynamics& dynamics = car->dynamics;
    float net_force = 0.0f;

    switch (car->action) {
        case DriverAction::BRAKE: {
            float applied_brake_force = car->brake_pedal * Config::MAX_BRAKE_SYSTEM_FORCE;

            net_force = -applied_brake_force - dynamics.drag_force - dynamics.cornering_drag - dynamics.engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::COAST: {
            net_force = -dynamics.drag_force - dynamics.cornering_drag - dynamics.engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::ACCELERATE: {
            float applied_engine_force = car->throttle_pedal * dynamics.desired_engine_force;
            net_force = applied_engine_force - dynamics.drag_force - dynamics.cornering_drag + dynamics.gravity_longitudinal;
            break;
        }
    }
    return net_force;
}

#endif