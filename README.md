# F1 High-Performance Physics and Telemetry Simulator

A high-performance Formula 1 simulation engine built from the ground up in C++ and CUDA. This project bridges the gap between theoretical vehicle dynamics and real-world racing by running a custom predictive physics AI against actual track telemetry (extracted via FastF1). 

Designed to process multiple aerodynamic and powertrain configurations concurrently using GPU parallelization, the engine visualizes the physics calculations in real-time via a custom SFML telemetry HUD.

**Note:** A CUDA-enabled NVIDIA GPU is currently strictly required to run this simulation engine.

![Simulation Demo](https://github.com/user-attachments/assets/8e620258-ab30-4dbd-971d-0b837a5ba5d7)



## Key Engineering Features

* **Predictive AI "Lookahead" Braking:** The AI driver dynamically scans upcoming track segments, calculating the true geometric limits of corners. It accounts for "Aero Decay" (loss of downforce at lower speeds) and applies a realistic human-transition margin to execute perfect braking zones.
* **3D Track Physics & Elevation:** Processes Z-axis GPS telemetry to calculate track pitch. The simulation accurately applies longitudinal gravity, making the car struggle to accelerate up steep inclines (e.g., Spa's Raidillon) and requiring longer braking distances on downhills.
* **Dynamic Traction & Aerodynamics:** 
    * **RWD Grip Constraints:** Engine torque is dynamically choked by the rear tires' mechanical grip (~55% of the car's mass), preventing arcade-like AWD acceleration and forcing realistic throttle modulation out of slow chicanes.
    * **Kamm Circle Physics:** Balances lateral cornering forces against longitudinal braking/accelerating limits in real-time.
* **Hybrid Powertrain Management:** Features a fully automated gearbox based on wheel RPM, Engine Braking mechanics, and a dynamic MGU-K (ERS) deployment system that maps battery usage based on upcoming straight lengths and current State of Charge (SOC).
* **Live Telemetry HUD (SFML):** A dynamic UI featuring a State Machine that allows real-time comparison between the simulated AI car and a real-world Ghost Car (e.g., Max Verstappen's Q3 lap), displaying micro-sector speeds, gear shifts, and pedal inputs.

## Technical Stack

* **Core Logic:** C++ (Strictly modular architecture separating AI brain, environment physics, and powertrain mechanics)
* **Parallel Computing:** CUDA (NVIDIA) for batch evaluating vehicle setups simultaneously.
* **Graphics & UI:** SFML (Simple and Fast Multimedia Library)
* **Data Ingestion:** Python (FastF1 library for real circuit and driver telemetry extraction)

## Environment and Requirements

* **Operating System:** Linux (Developed and optimized on Ubuntu 26.04)
* **Hardware:** NVIDIA GPU with CUDA architecture support (Validated on RTX 2070 Super) - **CUDA is required to run the engine.**
* **Dependencies:** `sfml-graphics`, `sfml-window`, `sfml-system`, `nvcc`, `cmake`

## Build and Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/rpadinha/f1-telemetry-sim.git](https://github.com/rpadinha/f1-telemetry-sim.git)

2. Build the project using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   
3. Run the simulation:
   ```bash
   ./f1-telemetry-sim
