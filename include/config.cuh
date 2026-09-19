#ifndef CONFIG_CUH
#define CONFIG_CUH

#include <physics.cuh>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace Config {
    // Physics and ambient
    constexpr float AIR_DENSITY = 1.225f;               // Air density in kg/m^3
    constexpr float FRONTAL_AREA = 1.5f;                // Frontal area of the car in m^2
    constexpr float GRAVITY = 9.81f;                    // Gravitational acceleration in m/s^2

    // Car Limits
    constexpr float DECEL_RATE = 49.0f;                 // Deceleration rate in m/s^2
    constexpr float BASE_MECH_GRIP = 1.6f;              // Mechanical grip coefficient
    constexpr float ICE_MIN_FORCE = 15000.f;            // ICE minimum force in Newtons
    constexpr float WHEEL_RADIUS = 0.36f;               // Wheel radius in meters
    constexpr float WHEEL_BASE = 3.6f;                  // Wheelbase of the car in m
    constexpr float FINAL_DRIVE = 5.0f;                 // Rear Differential
    constexpr float GRAVITY_CENTER_HEIGHT = 0.30f;      // Gravity center height of the f1 car in m
    constexpr float FUEL_FLOW_KG_S = 100.0f / 3600.0f;  // max fuel flow in
    // Gear Ratios and RPM Limits
    // so here we got error: identifier "Config::GEAR_RATIOS" is undefined in device code
    /*constexpr float GEAR_RATIOS[8] = {3.2f, 2.6f, 2.1f, //
                                      1.7f, 1.4f, 1.2f, //
                                      1.0f, 0.9f};      */
                                      //
    // instead we will use the CUDA_CALLABLE to make accessible in device code
    CUDA_CALLABLE inline float get_gear_ratio(int gear) {
        // added static because static const forces the compiler to bake this directly into registers or uniform memory
        static const float ratios[8] = {3.2f, 2.6f, 2.1f, 1.7f, 1.4f, 1.2f, 1.0f, 0.9f};
        if (gear < 1) gear = 1;
        if (gear > 8) gear = 8;
        return ratios[gear - 1];
    }
    constexpr float RPM_REDLINE = 12500.0f;             // Redline RPM of the car // wowzers
    constexpr float RPM_UPSHIFT = 11800.0f;             // Ideal upshift RPM of the car
    constexpr float RPM_DOWNSHIFT = 7500.0f;            // Ideal downshift RPM of the car
    constexpr float PEAK_POWER_RPM = 10500.0f;          // RPM at which the car produces peak power
    constexpr float RPM_IDLE = 5000.f;                  // RPM idle
    constexpr float MAX_ENGINE_BRAKING = 1500.0f;       // Engine braking max force

    // Tyres Things
    /* Here we will have to do the same we did above as we need to return values according
    to the TyreCompound current_compound */
    CUDA_CALLABLE inline TyreProperties get_tyre_properties(TyreCompound compound) {
        switch (compound) {
            case TyreCompound::C1: // hardest, low grip, low wear
                return { 1.42f, 115.0f, 15.0f, 0.00008f, 0.000008f };
            case TyreCompound::C2: // 
                return { 1.50f, 110.0f, 12.0f, 0.00010f, 0.000012f };
            case TyreCompound::C3: // 
                return { 1.60f, 105.0f, 10.0f, 0.00014f, 0.000018f };
            case TyreCompound::C4: // 
                return { 1.72f, 100.0f, 8.0f,  0.00020f, 0.000028f };
            case TyreCompound::C5: // softest, high grip, high wear
                return { 1.82f, 95.0f,  6.0f,  0.00030f, 0.000045f }; 
            default:
                return { 1.60f, 105.0f, 10.0f, 0.0014f, 0.000018f };
        }
    }

    // track and ambient
    constexpr float AMBIENT_TEMP_C = 25.0f;
    constexpr float TRACK_TEMP_C = 42.0f;

    // ERS System (MGU-K)
    constexpr float MAX_BATTERY_MJ = 4.0f;              // Maximum battery capacity in MegaJoules
    constexpr float MGUK_REGEN_KW = 350.0f;             // Maximum regenerative power of the MGU-K in kW

    // Simulation Parameters
    constexpr int LOOKAHEAD_METERS = 300;               // Lookahead distance in meters
    constexpr float PHYSICS_DT = 0.002f;                // Physics timestep in seconds
    constexpr int NUM_SETUPS = 1000;                    // Number of setups to simulate
}

#endif