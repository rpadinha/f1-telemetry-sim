#ifndef TYRES_CUH
#define TYRES_CUH

#include "config.cuh"
#include "physics.cuh"
#include <math.h>

// Instantaneous Effective Friction (mu_eff)
// calculates dynamic grip based on tire compound, thermal window and wear
__host__ __device__ inline float calculate_effective_grip(const F1Car* car) {
    TyreProperties properties = Config::get_tyre_properties(car->current_compound);

    float avg_temp = (car->tyre_temp_front_c + car->tyre_temp_rear_c) * 0.5f;

    // thermal parabola: eta_temp = 1.0 - (delta_T / window)² * penalty
    float delta_temp = avg_temp - properties.opt_temp_c;
    float normalized_diff = delta_temp / properties.temp_window_c;
    float eta_temp = 1.0f - (normalized_diff* normalized_diff * 0.25f);

    if (eta_temp < 0.65f) { eta_temp = 0.65f; }

    float eta_wear = 1.0f - (car->tyre_wear_pct * 0.35f);
    if (eta_wear < 0.60f) { eta_wear = 0.60f; }

    return properties.base_grip * eta_temp * eta_wear;
}

// thermal balance & wear integration per timestep
__host__ __device__ inline void update_tyres(F1Car* car,const F1CarDynamics* dynamics, const CarSetup* setup, float dt) {
    TyreProperties properties = Config::get_tyre_properties(car->current_compound);

    // tangential forces generating friction heat
    float f_lat = dynamics->lateral_force;
    // Approximated longitudinal load based on current acceleration
    float f_long = fabsf(car->a) * (setup->mass_kg + car->fuel_kg);
    float total_tangential_force = f_lat + f_long;

    // thermal dynamics
    // Heat generation proportional to friction work: Q_in = k * F_tangential * v
    // Front tires heat more under braking/turning; rear tires heat under longitudinal drive
    float heat_front = (f_lat * 0.60f + (car->a < 0.0f ? f_long * 0.65f : 0.0f)) * car->v * 0.000035f;
    float heat_rear  = (f_lat * 0.40f + (car->a > 0.0f ? f_long * 0.80f : 0.0f)) * car->v * 0.000035f;

    // Forced convective cooling with airflow: Q_out = h * sqrt(v) * (T_tyre - T_ambient)
    float air_cooling_factor = 0.08f * sqrtf(car->v + 1.0f);

    float cool_front = air_cooling_factor * (car->tyre_temp_front_c - Config::AMBIENT_TEMP_C) * 0.45f;
    float cool_rear  = air_cooling_factor * (car->tyre_temp_rear_c - Config::AMBIENT_TEMP_C) * 0.45f;

    car->tyre_temp_front_c += (heat_front - cool_front) * dt;
    car->tyre_temp_rear_c  += (heat_rear - cool_rear) * dt;

    // Enforce floor temperature equal to ambient
    if (car->tyre_temp_front_c < Config::AMBIENT_TEMP_C) car->tyre_temp_front_c = Config::AMBIENT_TEMP_C;
    if (car->tyre_temp_rear_c < Config::AMBIENT_TEMP_C) car->tyre_temp_rear_c = Config::AMBIENT_TEMP_C;

    // mechanical abrasion wear
    // delta_wear = wear_rate * stress_multiplier * distance_travelled
    float stress_multiplier = 1.0f + (total_tangential_force / 12000.0f);
    float distance_step = car->v * dt;

    car->tyre_wear_pct += properties.wear_rate * stress_multiplier * distance_step;
    if (car->tyre_wear_pct > 1.0f) {
        car->tyre_wear_pct = 1.0f;
    }
}

// Load Degressivity (instead of grip gorwing perfectly linear with vertical load)
__host__ __device__ inline float apply_load_sensitivity(float base_mu, float normal_force, float nominal_load) {
    constexpr float LOAD_SENSITIVITY = 0.000028f;
    float delta_load = normal_force - nominal_load;
    return base_mu / (1.0f + LOAD_SENSITIVITY * delta_load);
}

#endif