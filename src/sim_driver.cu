#include "sim_driver.cuh"
#include <math.h>

// getting max deceleration
__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup, float base_mu) {
    // first we calculate the aerodynamics at current speed v_ms
    float drag = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * setup->drag_coef * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;
    float nominal_load = (setup->mass_kg) * Config::GRAVITY;

    // mechanical grip with curr tyres
    float normal_force = (setup->mass_kg * Config::GRAVITY * cosf(pitch_angle)) + downforce;
    if (normal_force < 0.0f) { normal_force = 0.0f; }
    
    float final_mu = apply_load_sensitivity(base_mu, normal_force, nominal_load);
    float max_mech_brake = normal_force * final_mu;

    // total braking = brakes(tyres) + drag force
    float total_brake_force = max_mech_brake + drag;

    // gravity impacts if its going up / should also affect when its going down
    return total_brake_force / setup->mass_kg + (Config::GRAVITY * sinf(pitch_angle));
}

// getting allowed speed
__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    float current_pitch = get_track_pitch_angle(track, car->current_seg, num_segments);
    float base_mu = calculate_effective_grip(car);

    float cl_a = (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * cl_a;
    float nominal_load = (setup->mass_kg + car->fuel_kg) * Config::GRAVITY;
    float current_normal = (setup->mass_kg * Config::GRAVITY * cosf(current_pitch)) + downforce;
    if (current_normal < 0.0f) current_normal = 0.0f;

    float current_mu = apply_load_sensitivity(base_mu, current_normal, nominal_load);
    float speed = sqrtf((current_mu * current_normal * track[car->current_seg].radius_m) / setup->mass_kg);

    float dist_to_curve = track[car->current_seg].length_m - car->current_m;
    for (int i = 1; dist_to_curve < Config::LOOKAHEAD_METERS; ++i) {
        int lookahead = (car->current_seg + i) % num_segments;
        
        if (!is_straight(track[lookahead].radius_m, setup)) {
            float radius = track[lookahead].radius_m;
            float future_pitch = get_track_pitch_angle(track, lookahead, num_segments);
            
            // Decoupled 2-step Aero Prediction (Safe & stable)
            float v_guess_sq = base_mu * Config::GRAVITY * radius;
            float aero_df = 0.5f * Config::AIR_DENSITY * v_guess_sq * cl_a;
            float future_normal = (setup->mass_kg * Config::GRAVITY * cosf(future_pitch)) + aero_df;
            
            // Realistic Tire Grip Calculation
            float future_mu = apply_load_sensitivity(base_mu, future_normal, nominal_load);
            
            // Absolute Physics-Safe Speed Limit
            float corner_v_sq = (future_normal * future_mu * radius) / setup->mass_kg;
            float corner_v = sqrtf(corner_v_sq);

            // Balanced Deceleration & Braking Profile
            // Using car->v keeps the deceleration profile tied to physical state, not lookahead iteration
            float avg_speed_during_braking = (car->v + corner_v) * 0.5f;
            float effective_decel = get_max_deceleration(avg_speed_during_braking, current_pitch, setup, base_mu);
            
            // Tuned safety margin (0.88f = 12% margin) to bridge early/late discrepancies
            effective_decel *= 0.88f; 
            if (effective_decel < 1.0f) effective_decel = 1.0f;

            // Torricelli Threat Evaluation
            float v_critical = sqrtf(corner_v_sq + (2.0f * effective_decel * dist_to_curve));
            
            if (v_critical < speed) {
                speed = v_critical;
            }
        }
        dist_to_curve += track[lookahead].length_m; 
    }
    return speed;
}

// for now deploying eletric energy based on battery soc, speed and upcoming straight length
__host__ __device__ float calculate_mguk_deployment(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    if (car->action != DriverAction::ACCELERATE || car->v < 16.6f || car->battery_mj <= 0.f || car->current_gear <= 3) {
        return 0.0f;
    }

    if (car->battery_mj < 0.5f) {
        return 0.02f;
    }

    float upcoming_straight_m = track[car->current_seg].length_m - car->current_m;
    int lookahead = (car->current_seg + 1) % num_segments;
    
    while (is_straight(track[lookahead].radius_m, setup) && upcoming_straight_m < 2500.0f) {
        upcoming_straight_m += track[lookahead].length_m;
        lookahead = (lookahead + 1) % num_segments;
    }

    float ratio = 0.0f;
    if (upcoming_straight_m > 800.f) {
        ratio = 1.0f;
    } else if (upcoming_straight_m > 400.f) {
        ratio = 0.5f;
    } else {
        ratio = 0.2f;
    }

    float battery_ratio = car->battery_mj / Config::MAX_BATTERY_MJ;
    ratio *= battery_ratio;

    if (car->v*3.6f > 320.0f) {
        ratio *= 0.7f;
    }
    if (car->v*3.6f > 340.0f) {
        ratio *= 0.2f;
    }
    return ratio;
}

// ers fucntion independent from other code
__host__ __device__ void update_ers(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {
    if (car->throttle_pedal > 0.0f) {
        float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);
        float mguk_power = (mguk_ratio > 0.0f) ? (setup->mguk_power_kw * mguk_ratio) : 0.0f;
        
        car->battery_mj -= (mguk_power * dt) / 1000.0f;
        if (car->battery_mj < 0.0f) car->battery_mj = 0.0f;
    }

    if (car->brake_pedal > 0.0f) {
        car->battery_mj += (Config::MGUK_REGEN_KW * car->brake_pedal * dt) / 1000.0f;
    } 
    else if (car->throttle_pedal == 0.0f) {
        car->battery_mj += (Config::MGUK_REGEN_KW * 0.25f * dt) / 1000.0f;
    }

    if (car->battery_mj > Config::MAX_BATTERY_MJ) {
        car->battery_mj = Config::MAX_BATTERY_MJ;
    }
}

// updating driver pedals
__host__ __device__ void update_driver_pedals(F1Car* car, F1CarDynamics& dynamics, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {
    float target_throttle = 0.0f;
    float target_brake = 0.0f;

    switch (car->action) {
        case DriverAction::BRAKE:
            dynamics.desired_braking_force = dynamics.total_mass * Config::DECEL_RATE;
            target_brake = 1.0f;
            break;
            
        case DriverAction::COAST:
            break;
            
        case DriverAction::ACCELERATE: {
            float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);
            float mguk_power = (mguk_ratio > 0.0f) ? (setup->mguk_power_kw * mguk_ratio) : 0.0f;
            
            dynamics.desired_engine_force = compute_drive_force(car, setup, mguk_power);

            float lateral_ratio = 0.0f;
            if (track[car->current_seg].radius_m < 5000.0f && dynamics.max_grip > 1e-3f) {
                lateral_ratio = dynamics.lateral_force / dynamics.max_grip;
            }

            target_throttle = 1.0f - lateral_ratio;
            if (target_throttle > 1.0f) target_throttle = 1.0f;
            if (target_throttle < 0.0f) target_throttle = 0.0f;
            break;
        }
    }

    float pedal_rate = (1.0f / 0.15f) * dt;

    if (car->throttle_pedal < target_throttle) {
        car->throttle_pedal += pedal_rate;
        if (car->throttle_pedal > target_throttle) car->throttle_pedal = target_throttle;
    } else {
        car->throttle_pedal -= pedal_rate;
        if (car->throttle_pedal < target_throttle) car->throttle_pedal = target_throttle;
    }

    if (target_brake > 0.0f) car->throttle_pedal = 0.0f;

    if (car->brake_pedal < target_brake) {
        car->brake_pedal += pedal_rate;
        if (car->brake_pedal > target_brake) car->brake_pedal = target_brake;
    } else {
        car->brake_pedal -= pedal_rate;
        if (car->brake_pedal < target_brake) car->brake_pedal = target_brake;
    }
}