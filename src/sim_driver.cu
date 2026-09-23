#include "sim_driver.cuh"
#include <math.h>

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

__host__ __device__ void update_driver_pedals(F1Car* car, F1CarDynamics& dynamics, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {
    car->throttle_pedal = 0.0f;
    car->brake_pedal = 0.0f;
    car->drs_open = false;

    switch (car->action) {
        case DriverAction::BRAKE: {
            float engine_braking_force = (car->rpm / Config::RPM_REDLINE) * Config::MAX_ENGINE_BRAKING;

            // max brake force the car can do ~5G
            dynamics.desired_braking_force = setup->mass_kg * Config::DECEL_RATE;

            // what can we brake rn?
            float actual_breaking_force = (dynamics.desired_braking_force > dynamics.long_grip) ? dynamics.long_grip : dynamics.desired_braking_force;
            // adding engine breaking force to our actual breaking force
            actual_breaking_force += engine_braking_force;

            // pedal force we take the desired -> bf we can do rn / bf max 
            car->brake_pedal = actual_breaking_force / dynamics.desired_braking_force;
            if (car->brake_pedal > 1.0f) { car->brake_pedal = 1.0f; }

            if (car->battery_mj < Config::MAX_BATTERY_MJ) {
                car->battery_mj += (Config::MGUK_REGEN_KW * dt) / 1000.0f;
            }
            break;
        }
        case DriverAction::COAST: {
            if (car->battery_mj < Config::MAX_BATTERY_MJ) {
                car->battery_mj += (Config::MGUK_REGEN_KW * dt) / 1000.0f;
            }
            break;
        }
        case DriverAction::ACCELERATE: {
            
            if (track[car->current_seg].drs_zone && car->brake_pedal == 0.0f) {
                car->drs_open = true;
            }

            float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);
            float mguk_power = (mguk_ratio > 0.0f) ? (setup->mguk_power_kw * mguk_ratio) : 0.0f;
            if (mguk_ratio > 0.0f) {
                car->battery_mj -= (mguk_power * dt) / 1000.0f;
            }
            
            dynamics.desired_engine_force = compute_drive_force(car, setup, 1.0f, mguk_power);

            // if radius is to big in this case > 5000.f the lateral ratio stays at 0
            float lateral_ratio = 0.0f;
            if (track[car->current_seg].radius_m < 5000.0f) {
                lateral_ratio = dynamics.lateral_force / dynamics.max_grip;
            }

            float throttle_allowed = 1.0f - lateral_ratio;
            if (throttle_allowed > 1.0f) throttle_allowed = 1.0f;
            if (throttle_allowed < 0.0f) throttle_allowed = 0.0f;

            if (dynamics.desired_engine_force > 1e-3) {
                float actual_engine_force = (dynamics.desired_engine_force > dynamics.max_traction_force) 
                                        ? dynamics.max_traction_force 
                                        : dynamics.desired_engine_force;
                float ideal_pedal = actual_engine_force / dynamics.desired_engine_force;

                car->throttle_pedal = (ideal_pedal > throttle_allowed) ? throttle_allowed : ideal_pedal;
            } else {
                car->throttle_pedal = 0.0f;
            }
            break;
        }
    }
}