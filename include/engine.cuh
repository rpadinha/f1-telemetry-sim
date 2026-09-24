#ifndef ENGINE_CUH
#define ENGINE_CUH

#include "config.cuh"
#include "physics.cuh"

// Ice power Curve based on RPM
// Models the parabolic drop-off away from peak RPM
__host__ __device__ inline float get_engine_power_kw(float rpm, float max_power_kw) {
    float rpm_diff = (rpm - Config::PEAK_POWER_RPM) / 4000.0f;
    float rpm_factor = 1.0f - (rpm_diff * rpm_diff);
    if (rpm_factor < 0.2f) rpm_factor = 0.2f;
    return max_power_kw * rpm_factor;
}

// Toque & Contanct Patch Drive Force
// P = tau * Omega -> Tau = P / Omega -> F_wheel = (tau * GearRatio * FinalDrive) / WheelRadius
__host__ __device__ inline float compute_drive_force(const F1Car* car, const CarSetup* setup, float extra_power_kw) {
    if (car->throttle_pedal <= 0.0f || car->gear_shift_timer > 0.0f) return 0.0f;

    float ice_power = (car->fuel_kg > 0.0f) ? get_engine_power_kw(car->rpm, setup->ice_power_kw) : 0.0f;
    float total_power_kw = ice_power + extra_power_kw;

    float safe_rpm = (car->rpm < Config::RPM_IDLE) ? Config::RPM_IDLE : car->rpm;
    float engine_omega = (safe_rpm * 2.0f * 3.14159265f) / 60.0f;
    float engine_torque = (total_power_kw * 1000.0f) / engine_omega;

    float gear_ratio = Config::get_gear_ratio(car->current_gear);
    float wheel_torque = engine_torque * gear_ratio * Config::FINAL_DRIVE;

    return wheel_torque / Config::WHEEL_RADIUS;
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