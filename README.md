# F1 High-Performance Physics and Telemetry Simulator

A high-performance Formula 1 simulation engine developed in C++ and CUDA. This project features a custom physics solver and real-time telemetry visualization using SFML, designed to simulate vehicle dynamics across thousands of concurrent setups and compare theoretical racing logic against real-world track telemetry.

<img width="1330" height="807" alt="image" src="https://github.com/user-attachments/assets/3f0d0f8d-5904-47a6-ad76-42d635c1dda3" />
*(SFML Visualizer rendering real-time pedal inputs, RPM, speed deltas, and track position)*

## Key Features

*   **CUDA-Accelerated Batch Simulation:** Leverages GPU parallelization to evaluate thousands of vehicle configurations (aerodynamic drag, mass distribution, engine power) simultaneously, utilizing a Monte Carlo approach to identify optimal lap times.
*   **Advanced Vehicle Dynamics:**
    *   **Kamm Circle Physics:** Calculates dynamic lateral and longitudinal grip limits based on aerodynamic downforce and mechanical grip constraints.
    *   **Dynamic Throttle Modulation:** Implements lateral G-force dependent throttle application (roll-on) to prevent traction loss during high-speed corner exits.
    *   **Transmission Mechanics:** Accurate real-time RPM calculations, gear ratio shifting logic, and final drive differential tuning.
*   **Real-Time Telemetry Visualization (SFML):** A custom graphical interface rendering live telemetry comparisons between simulated states and real-world data, including dynamic pedal inputs, RPM gauges, and sector timings.
*   **Data Ingestion Pipeline:** Processes and integrates real-world circuit and telemetry data extracted via the Python FastF1 library.

## Technical Stack

*   **Core Logic:** C++
*   **Parallel Computing:** CUDA (NVIDIA)
*   **Graphics & UI:** SFML (Simple and Fast Multimedia Library)

## Environment and Requirements

*   **Operating System:** Linux (Developed and optimized on Pop!_OS)
*   **Hardware:** NVIDIA GPU with CUDA architecture support (Validated on RTX 2070 Super)
*   **Dependencies:** `sfml-graphics`, `sfml-window`, `sfml-system`, `nvcc`

## Build and Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/RupiX71/f1-telemetry-sim.git](https://github.com/RupiX71/f1-telemetry-sim.git)
