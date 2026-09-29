#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include <math.h>
#include "config.cuh"
#include "physics.cuh"
#include "engine.cuh"
#include "tyres.cuh"

__host__ __device__ inline bool is_straight(float radius_m, const CarSetup* setup) {
    return radius_m >= 1000.f;
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

// Returns +1.0f for right turn (weight shifts left), -1.0f for left turn (weight shifts right)
__host__ __device__ inline float get_turn_direction(const TrackSegment* track, int current_seg, int num_segments) {
    int prev_seg = (current_seg - 2 + num_segments) % num_segments;
    int next_seg = (current_seg + 2 + num_segments) % num_segments;

    float dx1 = track[current_seg].x - track[prev_seg].x;
    float dy1 = track[current_seg].y - track[prev_seg].y;
    float dx2 = track[next_seg].x - track[current_seg].x;
    float dy2 = track[next_seg].y - track[current_seg].y;

    float cross_z = (dx1 * dy2) - (dy1 * dx2);
    return (cross_z < 0.0f) ? 1.0f : -1.0f;
}

__host__ __device__ inline F1CarDynamics calculate_car_dynamics(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    F1CarDynamics dynamics;

    // Full weight
    dynamics.total_mass = car->fuel_kg + setup->mass_kg;

    // Track pitch angle theta = arcsin(delta_z / delta_s)
    dynamics.pitch_angle = get_track_pitch_angle(track, car->current_seg, num_segments);

    // Dynamic drag reduction when rear wing slot gap is open
    float current_drag_coef = setup->drag_coef;
    if (car->drs_open) { current_drag_coef *= 0.65f; }
    dynamics.drag_force = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * current_drag_coef * Config::FRONTAL_AREA;
    
    // F_downforce = 0.5 * rho * v^2 * Cl * A (where Cl * A ~= 3.0 * Cd * A)
    float cl_a = (setup->drag_coef * 3.0f) * Config::FRONTAL_AREA;
    dynamics.downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * cl_a;

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

    // Direct radius lookup from the track segment
    float radius = track[car->current_seg].radius_m;
    float lat_dir = get_turn_direction(track, car->current_seg, num_segments);

    // Centrifugal cornering load: F_lat = (m * v^2) / R
    dynamics.lateral_force = (dynamics.total_mass * car->v * car->v) / radius;

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

    // pretection against the physics limit, sbins and lateral force hits the limit
    if (dynamics.lateral_force > dynamics.max_grip) {
        dynamics.lateral_force = dynamics.max_grip;
    }

    dynamics.slip_angle_rad = calculate_slip_angle(dynamics.lateral_force, dynamics.max_grip);
    dynamics.cornering_drag = dynamics.lateral_force * sinf(dynamics.slip_angle_rad);

    // Kamm Circle: Flong_grip = sqrtf(F_gripmax² - Flat²)
    float diff = (dynamics.max_grip * dynamics.max_grip) - (dynamics.lateral_force * dynamics.lateral_force);
    dynamics.long_grip = sqrtf(fmaxf(0.0f, diff));

    // For F1 only Rear tyres accel so f1 is rwd so grip all rear lmao
    float rear_max_grip = grip_rl + grip_rr;
    float rear_lat_ratio = (dynamics.normal_force > 0.0f) ? ((dynamics.load_rl + dynamics.load_rr) / dynamics.normal_force) : 0.55f;
    float rear_lateral_force = dynamics.lateral_force * rear_lat_ratio;

    // Maximum traction force limited by dynamic rear vertical load
    if (rear_max_grip > rear_lateral_force) {
        dynamics.max_traction_force = sqrtf((rear_max_grip * rear_max_grip) - (rear_lateral_force * rear_lateral_force));
    } else {
        dynamics.max_traction_force = 0.0f;
    }

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
            float requested_brake = car->brake_pedal * Config::MAX_BRAKE_SYSTEM_FORCE;
            float applied_brake_force = (requested_brake > dynamics.long_grip) ? dynamics.long_grip : requested_brake;

            net_force = -applied_brake_force - dynamics.drag_force - dynamics.cornering_drag - dynamics.engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::COAST: {
            net_force = -dynamics.drag_force - dynamics.cornering_drag - dynamics.engine_braking_force + dynamics.gravity_longitudinal;
            break;
        }
        case DriverAction::ACCELERATE: {
            float requested_engine = car->throttle_pedal * dynamics.desired_engine_force;
            float applied_engine_force = (requested_engine > dynamics.max_traction_force) 
                                       ? dynamics.max_traction_force 
                                       : requested_engine;

            net_force = applied_engine_force - dynamics.drag_force - dynamics.cornering_drag + dynamics.gravity_longitudinal;
            break;
        }
    }
    return net_force;
}

#endif