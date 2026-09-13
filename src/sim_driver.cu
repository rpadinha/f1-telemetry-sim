#include "sim_driver.cuh"
#include <math.h>

__host__ __device__ float get_max_deceleration(float v_ms, float pitch_angle, const CarSetup* setup) {
    // first we calculate the aerodynamics at current speed v_ms
    float drag = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * setup->drag_coef * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (v_ms * v_ms) * (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;

    // mechanical grip with curr tyres
    float normal_force = (setup->mass_kg * Config::GRAVITY * cosf(pitch_angle)) + downforce;
    if (normal_force < 0.0f) { normal_force = 0.0f; }
    float max_mech_brake = normal_force * Config::BASE_MECH_GRIP;

    // total braking = brakes(tyres) + drag force
    float total_brake_force = max_mech_brake + drag;

    float base_decel = total_brake_force / setup->mass_kg;

    // gravity impacts if its going up / should also affect when its going down
    return base_decel + (Config::GRAVITY * sinf(pitch_angle));
}

__host__ __device__ float get_allowed_speed(const F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
    float current_pitch = get_track_pitch_angle(track, car->current_seg, num_segments);
    float cl_a = (setup->drag_coef * 3.f) * Config::FRONTAL_AREA;
    float downforce = 0.5f * Config::AIR_DENSITY * (car->v * car->v) * cl_a;

    float current_normal = (setup->mass_kg * Config::GRAVITY * cosf(current_pitch)) + downforce;
    if (current_normal < 0.0f) current_normal = 0.0f;

    float max_grip = current_normal * Config::BASE_MECH_GRIP;

    float speed = sqrtf((max_grip * track[car->current_seg].radius_m) / setup->mass_kg);

    float dist_to_curve = track[car->current_seg].length_m - car->current_m;

    for (int i = 1; i <= Config::LOOKAHEAD_METERS; ++i) {
        int lookahead = (car->current_seg + i) % num_segments;
        
        if (!is_straight(track[lookahead].radius_m, setup)) {
            // calling back to this, this is a huge bug we had in the code as we were using the same downforce calculated for the current speed (line26 rn)
            // FIX: Isolated v² algebraically on both sides of the friction limit equation:
            // F_centrifugal = F_grip_mechanical + F_grip_downforce -> Solving for v² gives the formula below.
            // This calculates the true, physics-safe speed limit of the corner purely from geometry.
            float future_pitch = get_track_pitch_angle(track, lookahead, num_segments);
            float centrifugal_term = setup->mass_kg / track[lookahead].radius_m;
            float aero_grip_term = 0.5f * Config::AIR_DENSITY * cl_a * Config::BASE_MECH_GRIP;

            float corner_v_sq = 0.0f;

            if (centrifugal_term > aero_grip_term) {
                float gravity_grip = setup->mass_kg * Config::GRAVITY * cosf(future_pitch) * Config::BASE_MECH_GRIP;
                corner_v_sq = gravity_grip / (centrifugal_term - aero_grip_term);
            } else {
                // Aero grip completely overpowers cornering forces (flat out corner)
                // Set to a high upper safety limit or current top speed capabilities
                corner_v_sq = 100.0f * 100.0f; 
            }

            float corner_v = sqrtf(corner_v_sq);
            // Getting average G force in all braking zone
            float avg_speed_during_braking = (speed + corner_v) * 0.5f;
            float effective_decel = get_max_deceleration(avg_speed_during_braking, current_pitch, setup);
            // 20% cutoff for humanlike transition
            effective_decel *= 0.80f;
            if (effective_decel < 1.0f) { effective_decel = 1.0f; }

            float v_critical = sqrtf(corner_v_sq + (2.0f * effective_decel * dist_to_curve));
            
            if (v_critical < speed) {
                speed = v_critical;
            }
        }
        dist_to_curve += track[lookahead].length_m; 
    }
    return speed;
}

__host__ __device__ float calculate_mguk_deployment(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
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

__host__ __device__ void update_driver_pedals(F1Car* car, F1CarDynamics& dynamics, const CarSetup* setup, const TrackSegment* track, int num_segments, float dt) {
    switch (car->action) {
        case DriverAction::BRAKE: {
            car->throttle_pedal = 0.0f;
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
            car->throttle_pedal = 0.0f;
            car->brake_pedal = 0.0f;
            if (car->battery_mj < Config::MAX_BATTERY_MJ) {
                car->battery_mj += (Config::MGUK_REGEN_KW * dt) / 1000.0f;
            }
            break;
        }
        case DriverAction::ACCELERATE: {
            car->brake_pedal = 0.0f;
            // Calculate the current power output based on RPM
            float rpm_diff = (car->rpm - Config::PEAK_POWER_RPM) / 4000.0f;
            float rpm_factor = 1.0f - (rpm_diff * rpm_diff);
            if (rpm_factor < 0.2f) rpm_factor = 0.2f;
            float current_power_kw = setup->ice_power_kw * rpm_factor;

            float mguk_ratio = calculate_mguk_deployment(car, setup, track, num_segments);

            if (mguk_ratio > 0.0f) {
                float power_to_add = setup->mguk_power_kw * mguk_ratio;
                current_power_kw += power_to_add;
                car->battery_mj -= (power_to_add * dt) / 1000.0f;
            }
            

            // ? 
            float safe_rpm = car->rpm;
            if (safe_rpm < 4000.0f) safe_rpm = 4000.0f;
            // TORQUE mechanics Prevents the division-by-zero or division-by-one stability issues at low speeds
            float engine_omega = (safe_rpm * 2.0f * 3.14159265f) / 60.f;
            float engine_torque = (current_power_kw * 1000.f) / engine_omega;

            // Translate engine torque down to the contact patch of the tyre
            float wheel_torque = engine_torque * Config::get_gear_ratio(car->current_gear) * Config::FINAL_DRIVE;
            dynamics.desired_engine_force = wheel_torque / Config::WHEEL_RADIUS;

            // if radius is to big in this case > 5000.f the lateral ratio stays at 0
            float lateral_ratio = 0.0f;
            if (track[car->current_seg].radius_m < 5000.0f) {
                lateral_ratio = dynamics.lateral_force / dynamics.max_grip;
            }

            float throttle_allowed = 1.0f - lateral_ratio;
            if (throttle_allowed > 1.0f) throttle_allowed = 1.0f;
            if (throttle_allowed < 0.0f) throttle_allowed = 0.0f;


            // rwd thing ~55% 
            float rear_grip_ratio = 0.55f;
            float max_traction_force = dynamics.long_grip * rear_grip_ratio;

            float actual_engine_force = (dynamics.desired_engine_force > max_traction_force) ? max_traction_force : dynamics.desired_engine_force;
            float ideal_pedal = actual_engine_force / dynamics.desired_engine_force;

            car->throttle_pedal = (ideal_pedal > throttle_allowed) ? throttle_allowed : ideal_pedal;
            break;
        }
    }
}