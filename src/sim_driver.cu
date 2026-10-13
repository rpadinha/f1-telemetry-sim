#include "sim_driver.cuh"
#include <math.h>


// get_max_deceleration: no lateral term during straight-line braking,
// per-tyre load sensitivity applied ONCE on the raw mu
__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup,
                                               float mu_raw, float total_mass, float future_radius) {
    float q         = 0.5f * Config::AIR_DENSITY * v_ms * v_ms;
    float drag      = q * setup->drag_coef * Config::FRONTAL_AREA;
    float downforce = q * setup->drag_coef * 4.5f * Config::FRONTAL_AREA;
    float static_n  = total_mass * Config::GRAVITY * cosf(pitch_angle);
    float normal    = fmaxf(0.0f, static_n + downforce);

    float mu = apply_load_sensitivity(mu_raw, normal * 0.25f, static_n * 0.25f);
    float max_mech_brake = fminf(normal * mu, Config::MAX_BRAKE_SYSTEM_FORCE);

    float total = max_mech_brake + drag + Config::MAX_ENGINE_BRAKING * 0.8f;
    return total / total_mass + Config::GRAVITY * sinf(pitch_angle);
}

// could this be removed using max verstappen's velocity instead?
__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    float current_radius = track[car->current_seg].radius_m;

    // Safe base mechanical traction speed definition
    float speed = sqrtf((car->dynamics.max_grip * current_radius) / car->dynamics.total_mass);

    float mu_raw = 0.25f * (calculate_tyre_grip(car->current_compound, car->tyre_temp_fl, car->tyre_wear_fl)
                      + calculate_tyre_grip(car->current_compound, car->tyre_temp_fr, car->tyre_wear_fr)
                      + calculate_tyre_grip(car->current_compound, car->tyre_temp_rl, car->tyre_wear_rl)
                      + calculate_tyre_grip(car->current_compound, car->tyre_temp_rr, car->tyre_wear_rr));
    
    float cl_a = (setup->drag_coef * 4.5f) * Config::FRONTAL_AREA;
    
    // Scaled dynamic look-ahead window matching vehicle kinetic energy state (with braking distance, so it can never be the limiting factor)
    float dynamic_lookahead = Config::LOOKAHEAD_METERS + (car->v * car->v) / (2.0f * 15.0f);

    float dist_to_curve = track[car->current_seg].length_m - car->current_m;

    for (int i = 1; dist_to_curve < dynamic_lookahead; ++i) {
        int lookahead = (car->current_seg + i) % num_segments;
        float future_radius = track[lookahead].radius_m;

        // FIXED: Expanded structural bounds checks to safely capture high-speed sweeps
        if (future_radius < 2000.0f) {
            float future_pitch = track[lookahead].pitch_angle;
            
            // FIXED: Quadratic Aero Prediction formulation to balance high downforce properties
            float aero_factor = 0.5f * Config::AIR_DENSITY * cl_a / car->dynamics.total_mass;
            float denominator = 1.0f - (mu_raw * aero_factor * future_radius);
            
            float corner_v_sq;
            if (denominator > 0.05f) {
                corner_v_sq = (mu_raw * Config::GRAVITY * future_radius) / denominator;
            } else {
                corner_v_sq = (car->dynamics.max_grip * future_radius) / car->dynamics.total_mass;
            }
            
            // Limit velocity ceiling bounds to absolute tire properties
            float corner_v = sqrtf(corner_v_sq);
            
            // Root-Mean-Square calculation models physical aerodynamic braking drag forces
            float rms_speed_during_braking = sqrtf((car->v * car->v + corner_v_sq) * 0.5f);
            float effective_decel = get_max_deceleration(rms_speed_during_braking, future_pitch, setup, mu_raw, car->dynamics.total_mass, future_radius);
            
            // 8% performance engineering buffer scale
            effective_decel *= 0.92f;
            if (effective_decel < 1.0f) effective_decel = 1.0f;

            // Torricelli classical kinematic trajectory projection step
            float v_critical = sqrtf(corner_v_sq + (2.0f * effective_decel * dist_to_curve));
            
            if (v_critical < speed) {
                speed = v_critical;
            }
        }
        dist_to_curve += track[lookahead].length_m;
    }
    return speed;
}

__host__ __device__ void update_driver_pedals(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments, float target_speed, float dt) {
    F1CarDynamics& dynamics = car->dynamics;
    float target_throttle = 0.0f;
    float target_brake = 0.0f;

    switch (car->action) {
        case DriverAction::BRAKE: {
            float ideal_brake = dynamics.long_grip / Config::MAX_BRAKE_SYSTEM_FORCE;
            target_brake = ideal_brake;
            if (target_brake > 1.0f) target_brake = 1.0f;
            if (target_brake < 0.0f) target_brake = 0.0f;

            car->throttle_pedal = target_throttle;
            break;
        }
        case DriverAction::COAST: {
            break;
        }

        case DriverAction::ACCELERATE: {
            float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);
            float mguk_power = (mguk_ratio > 0.0f) ? (setup->mguk_power_kw * mguk_ratio) : 0.0f;
            
            // calculate max raw force the engine can produce
            dynamics.desired_engine_force = compute_drive_force(car, setup, mguk_power);

            // compute target throttle only based on speed error
            float speed_error = target_speed - car->v;
            float speed_throttle = (speed_error <= 0.0f) ? 0.0f : fminf(1.0f, speed_error / 5.0f);
            // 3. Compute Grip-Limited Throttle (Traction Control Cap)
            // Available longitudinal force based on the friction circle (accounting for lateral forces)
            float grip_throttle_cap = 1.0f;
            
            if (dynamics.desired_engine_force > 1e-3f) {
                // If max_traction_force is already pre-subtracted by lateral components via Pacejka,
                // we can map the ratio directly. Otherwise, evaluate the cap:
                float traction_ratio = dynamics.max_traction_force / dynamics.desired_engine_force;
                
                // Cap the throttle command to match the physical traction threshold
                grip_throttle_cap = fminf(1.0f, traction_ratio);
            }

            // 4. Lateral Derating Factor (Optional but recommended for stability out of corners)
            // If the car is pulling massive lateral Gs, smoothly lower the throttle ceiling 
            // to prevent sudden oversteer snap snaps during corner exit phases.
            float lateral_g = dynamics.lateral_force / (dynamics.total_mass * Config::GRAVITY);
            float corner_exit_smoothing = fmaxf(0.1f, 1.0f - (lateral_g * 0.15f)); // Scales down grip ceiling slightly under high lateral loads
            grip_throttle_cap *= corner_exit_smoothing;

            // 5. Final Arbitrated Target Throttle
            // The driver wants speed_demanded_throttle, but physics clamps it to grip_throttle_cap!
            target_throttle = fminf(speed_throttle, grip_throttle_cap);
            
            // Absolute sanity constraints
            if (target_throttle < 0.0f) target_throttle = 0.0f;
            if (target_throttle > 1.0f) target_throttle = 1.0f;
            
            break;
        }
    }

    // 15ms rate
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