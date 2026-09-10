#ifndef CONFIG_CUH
#define CONFIG_CUH

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
    constexpr float DECEL_RATE = 39.0f;                 // Deceleration rate in m/s^2
    constexpr float BASE_MECH_GRIP = 1.6f;              // Mechanical grip coefficient
    constexpr float ICE_MIN_FORCE = 15000.f;            // ICE minimum force in Newtons
    constexpr float WHEEL_RADIUS = 0.36f;               // Wheel radius in meters
    constexpr float FINAL_DRIVE = 5.0f;                 // Rear Differential

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
    constexpr float RPM_REDLINE = 125000.0f;            // Redline RPM of the car
    constexpr float RPM_UPSHIFT = 11800.0f;             // Ideal upshift RPM of the car
    constexpr float RPM_DOWNSHIFT = 7500.0f;            // Ideal downshift RPM of the car
    constexpr float PEAK_POWER_RPM = 10500.0f;          // RPM at which the car produces peak power
    constexpr float RPM_IDLE = 5000.f;                  // RPM idle
    constexpr float MAX_ENGINE_BRAKING = 1500.0f;       // idk about this chief

    // ERS System (MGU-K)
    constexpr float MAX_BATTERY_MJ = 4.0f;              // Maximum battery capacity in MegaJoules
    constexpr float MGUK_REGEN_KW = 350.0f;             // Maximum regenerative power of the MGU-K in kW

    // Simulation Parameters
    constexpr int LOOKAHEAD_METERS = 300;               // Lookahead distance in meters
    constexpr float PHYSICS_DT = 0.002f;                // Physics timestep in seconds
    constexpr int NUM_SETUPS = 1000;                    // Number of setups to simulate
}

#endif