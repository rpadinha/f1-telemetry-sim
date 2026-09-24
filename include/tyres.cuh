#ifndef TYRES_CUH
#define TYRES_CUH

#include "config.cuh"
#include "physics.cuh"
#include <math.h>

// Instantaneous Effective Friction (mu_eff)
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

// thermal balance & wear integration per timestep
__host__ __device__ inline void update_tyres(F1Car* car, const F1CarDynamics dynamics, const CarSetup* setup, float dt) {
    TyreProperties properties = Config::get_tyre_properties(car->current_compound);

    // total load 
    float total_load = dynamics.load_fl + dynamics.load_fr + dynamics.load_rl + dynamics.load_rr;
    if (total_load < 1.0f) total_load = 1.0f; // avoid division by 1.0f / 0.f


    float* temps[4] = { &car->tyre_temp_fl, &car->tyre_temp_fr, &car->tyre_temp_rl, &car->tyre_temp_rr };
    float* wears[4] = { &car->tyre_wear_fl, &car->tyre_wear_fr, &car->tyre_wear_rl, &car->tyre_wear_rr };
    float loads[4]  = { dynamics.load_fl, dynamics.load_fr, dynamics.load_rl, dynamics.load_rr };

    // Longitudinal bias: braking makes fronts more hot (60/40), accelerating gives more heat to the rear (40/60)
    float long_bias[4] = {
        (car->a < 0.0f) ? 0.60f : 0.40f, // FL
        (car->a < 0.0f) ? 0.60f : 0.40f, // FR
        (car->a > 0.0f) ? 0.60f : 0.40f, // RL
        (car->a > 0.0f) ? 0.60f : 0.40f  // RR
    };

    float air_cooling_factor = 0.014f * sqrt(car->v + 1.0f);

    for (int i = 0 ; i < 4 ; i++) {
        // The force this tyre suffers is based on the weight percentage
        float load_ratio = loads[i] / total_load;
        
        float f_lat_i = dynamics.lateral_force * load_ratio;
        float f_long_i = (fabsf(car->a) * dynamics.total_mass) * load_ratio * long_bias[i] * 2.0f; // x2 because bias goes for both

        // combined tangencial force (sqrt(Fx^2 + Fy^2))
        float f_tangential = sqrtf(f_lat_i * f_lat_i + f_long_i * f_long_i);

        // Q_in = F_tangential * v * k_friction
        float heat = f_tangential * car->v * 0.00012f;
        float cool = air_cooling_factor * (*temps[i] - Config::AMBIENT_TEMP_C);

        *temps[i] += (heat - cool) * dt;
        if (*temps[i] < Config::AMBIENT_TEMP_C) *temps[i] = Config::AMBIENT_TEMP_C;

        // progressive wear as tangencial force to the tyre
        float stress_multiplier = 1.0f + (f_tangential / 3000.0f);
        *wears[i] += properties.wear_rate * stress_multiplier * (car->v * dt);
        
        if (*wears[i] > 1.0f) *wears[i] = 1.0f;
    }
}

// Load Degressivity (instead of grip gorwing perfectly linear with vertical load)
__host__ __device__ inline float apply_load_sensitivity(float base_mu, float normal_force, float nominal_load) {
    constexpr float LOAD_SENSITIVITY = 0.000028f;
    float delta_load = normal_force - nominal_load;
    return base_mu / (1.0f + LOAD_SENSITIVITY * delta_load);
}

#endif