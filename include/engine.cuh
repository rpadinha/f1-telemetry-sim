#ifndef ENGINE_CUH
#define ENGINE_CUH

#include "config.cuh"
#include "physics.cuh"
#include "math_utils.cuh"

// Ice power Curve based on RPM
// Models the parabolic drop-off away from peak RPM
__host__ __device__ inline float get_engine_power_kw(float rpm, float max_power_kw) {
    float rpm_diff = (rpm - Config::PEAK_POWER_RPM) / 4000.0f;
    float rpm_factor = 1.0f - (rpm_diff * rpm_diff);
    if (rpm_factor < 0.2f) rpm_factor = 0.2f;
    return max_power_kw * rpm_factor;
}

// MGuK deployment
// for now deploying eletric energy based on battery soc, speed and upcoming straight length 
__host__ __device__ inline float calculate_mguk_deployment(F1Car* car, const CarSetup* setup, const TrackSegment* track, int num_segments) {
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

// Toque & Contanct Patch Drive Force
// P = tau * Omega -> Tau = P / Omega -> F_wheel = (tau * GearRatio * FinalDrive) / WheelRadius
__host__ __device__ inline float compute_drive_force(const F1Car* car, const CarSetup* setup, float throttle_pedal, float extra_power_kw = 0.0f) {
    if (throttle_pedal <= 0.0f || car->gear_shift_timer > 0.0f) { return 0.0f; }

    float ice_power = (car->fuel_kg > 0.0f) ? get_engine_power_kw(car->rpm, setup->ice_power_kw) : 0.0f;
    float total_power_kw = ice_power + extra_power_kw;

    float safe_rpm = (car->rpm < Config::RPM_IDLE) ? Config::RPM_IDLE : car->rpm;
    float engine_omega = (safe_rpm * 2.0f * 3.14159265f) / 60.0f;

    float engine_torque = (total_power_kw * 1000.0f) / engine_omega;

    float gear_ratio = Config::get_gear_ratio(car->current_gear);
    float wheel_torque = engine_torque * gear_ratio * Config::FINAL_DRIVE;

    return (wheel_torque / Config::WHEEL_RADIUS) * throttle_pedal;
}

// Dynamic Fuel Burn
// FIA maximum mass fuel flow limit: 100 kg/h (~0.02778 kg/s) at 100% throttle
__host__ __device__ inline void burn_fuel(F1Car* car, float throttle_pedal, float dt) {
    if (car->fuel_kg > 0.0f && throttle_pedal > 0.0f) {
        constexpr float FUEL_FLOW_KG_S = 100.0f / 3600.0f;  // max fuel flow in kg/s -> 100 kg/h (~0.02778 kg/s) at 100% throttle
        car->fuel_kg -= (throttle_pedal * FUEL_FLOW_KG_S) * dt;
        if (car->fuel_kg < 0.0f) {
            car->fuel_kg = 0.0f;
        }
    }
}

#endif