#ifndef TYRES_CUH
#define TYRES_CUH

#include "config.cuh"
#include "physics.cuh"
#include <math.h>

// calculates dynamic grip based on tire compound, thermal window and wear
__host__ __device__ inline float calculate_tyre_grip(TyreCompound compound, float temp_c, float wear_pct) {
    TyreProperties properties = Config::get_tyre_properties(compound);

    // thermal parabola: eta_temp = 1.0 - (delta_T / window)² * penalty
    float delta_temp = temp_c - properties.opt_temp_c;
    float normalized_diff = delta_temp / properties.temp_window_c;
    float eta_temp = 1.0f - (normalized_diff * normalized_diff * 0.25f);
    if (eta_temp < 0.65f) eta_temp = 0.65f;

    float eta_wear = 1.0f - (wear_pct * 0.35f);
    if (eta_wear < 0.65f) eta_wear = 0.65f;

    return properties.base_grip * eta_temp * eta_wear;
}

__host__ __device__ inline float calculate_effective_grip(const F1Car* car) {
    float avg_temp = (car->tyre_temp_fl + car->tyre_temp_fr + car->tyre_temp_rl + car->tyre_temp_rr) * 0.25f;
    float avg_wear = (car->tyre_wear_fl + car->tyre_wear_fr + car->tyre_wear_rl + car->tyre_wear_rr) * 0.25f;
    return calculate_tyre_grip(car->current_compound, avg_temp, avg_wear);
}

__host__ __device__ inline float calculate_slip_angle(float f_lat, float max_grip) {
    if (max_grip <= 1e-3f) return 0.0f;


    float fy_norm = f_lat / max_grip;

    float peak_sin = sinf(Config::PACEJKA_C * 1.5708f);

    if (fy_norm > peak_sin * 0.99f) { fy_norm = peak_sin * 0.99f; }
    float alpha_rad = (1.0f / Config::PACEJKA_B) * tanf(asinf(fy_norm) / Config::PACEJKA_C);

    return alpha_rad;
}

// thermal balance & wear integration per timestep
__host__ __device__ inline void update_tyres(F1Car* car, float dt) {
    TyreProperties properties = Config::get_tyre_properties(car->current_compound);
    const F1CarDynamics& dynamics = car->dynamics;

    float* temps[4] = { &car->tyre_temp_fl, &car->tyre_temp_fr, &car->tyre_temp_rl, &car->tyre_temp_rr };
    float* wears[4] = { &car->tyre_wear_fl, &car->tyre_wear_fr, &car->tyre_wear_rl, &car->tyre_wear_rr };
    float loads[4]  = { dynamics.load_fl, dynamics.load_fr, dynamics.load_rl, dynamics.load_rr };

    float total_load = loads[0] + loads[1] + loads[2] + loads[3];
    if (total_load < 1.0f) total_load = 1.0f;
    
    for (int i = 0; i < 4; i++) {
        float load_ratio = loads[i] / total_load;

        // 1. Corner-specific lateral force based on individual wheel vertical load
        float f_lat_i = dynamics.lateral_force * load_ratio;

        // 2. Corner-specific longitudinal force:
        // Braking (car->a < 0): 60% front axle, 40% rear axle
        // Throttle (car->a > 0): 100% REAR AXLE ONLY (F1 is RWD)
        float f_long_i = 0.0f;
        if (car->a < 0.0f) {
            f_long_i = (i < 2) ? (fabsf(car->a) * dynamics.total_mass * 0.30f)   // 60% front / 2 wheels
                               : (fabsf(car->a) * dynamics.total_mass * 0.20f);  // 40% rear / 2 wheels
        } else {
            f_long_i = (i < 2) ? 0.0f                                            // Fronts produce 0 driving force
                               : (car->a * dynamics.total_mass * 0.50f);        // Rears carry all drive torque
        }

        // 3. Friction work from sliding slip velocity
        float v_slip_lat = car->v * sinf(dynamics.slip_angle_rad);
        float heat_friction = (f_lat_i * v_slip_lat * 0.00020f) + (f_long_i * 0.000085f);

        // 4. Carcass flexing (viscoelastic hysteresis)
        float heat_carcass = loads[i] * (car->v + 15.0f) * 0.0000045f;

        // 5. Thermal equilibrium: track conduction & convective air cooling
        float q_track = 0.012f * (Config::TRACK_TEMP_C - *temps[i]);
        float cool    = 0.0022f * sqrtf(car->v + 1.0f) * (*temps[i] - Config::AMBIENT_TEMP_C);

        *temps[i] += (heat_friction + heat_carcass + q_track - cool) * dt;
        if (*temps[i] < Config::AMBIENT_TEMP_C) *temps[i] = Config::AMBIENT_TEMP_C;

        // Wear integration
        float f_tangential = sqrtf(f_lat_i * f_lat_i + f_long_i * f_long_i);
        *wears[i] += properties.wear_rate * (1.0f + f_tangential / 3000.0f) * (car->v * dt);
        if (*wears[i] > 1.0f) *wears[i] = 1.0f;
    }
}

// Load Degressivity (instead of grip growing perfectly linear with vertical load)
__host__ __device__ inline float apply_load_sensitivity(float base_mu, float normal_force, float nominal_load_per_tyre) {
    constexpr float LOAD_SENSITIVITY = 0.000012f;
    float delta_load = normal_force - nominal_load_per_tyre;
    if (delta_load < 0.0f) delta_load = 0.0f;
    return base_mu / (1.0f + LOAD_SENSITIVITY * delta_load);
}

#endif