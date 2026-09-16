# F1 High-Performance Physics and Telemetry Simulator

A high-performance Formula 1 simulation engine built from the ground up in C++ and CUDA. This project bridges the gap between theoretical vehicle dynamics and real-world racing by running a custom predictive physics AI against actual track telemetry (extracted via FastF1). 

Designed to process multiple aerodynamic and powertrain configurations concurrently using GPU parallelization, the engine visualizes the physics calculations in real-time via a custom SFML telemetry HUD.

**Note:** A CUDA-enabled NVIDIA GPU is currently strictly required to run this simulation engine.

![Simulation Demo](https://github.com/user-attachments/assets/8e620258-ab30-4dbd-971d-0b837a5ba5d7)


### Mathematical Modeling & Physics Formulation

| Domain | Phenomenon | Formulation |
| :--- | :--- | :--- |
| **Aerodynamics** | Drag Force | $F_{\mathrm{drag}} = \frac{1}{2} \rho \cdot v^2 \cdot C_d \cdot A$ |
| | DRS Active Drag | $C_{d,\mathrm{DRS}} = 0.65 \cdot C_d$ |
| | Downforce Generation | $F_{\mathrm{downforce}} = \frac{1}{2} \rho \cdot v^2 \cdot (3.0 \cdot C_d \cdot A)$ |
| **Track & Loads** | Pitch Angle | $\theta = \arcsin\left(\frac{z_{i+1} - z_i}{\Delta s}\right)$ |
| | Normal Load ($F_N$) | $F_N = (m \cdot g \cdot \cos\theta) + F_{\mathrm{downforce}}$ |
| | Longitudinal Gravity | $F_{g,\mathrm{long}} = -m \cdot g \cdot \sin\theta$ |
| **Tire Dynamics** | Maximum Grip Budget | $F_{\mathrm{grip,max}} = F_N \cdot \mu_{\mathrm{base}}$ |
| | Lateral Cornering Load | $F_{\mathrm{lat}} = \frac{m \cdot v^2}{R}$ |
| | Kamm Circle Long. Limit | $F_{\mathrm{long,grip}} = \sqrt{F_{\mathrm{grip,max}}^2 - F_{\mathrm{lat}}^2}$ |
| | RWD Traction Limit | $F_{\mathrm{traction,max}} = 0.55 \cdot F_{\mathrm{long,grip}}$ |
| **Powertrain** | Wheel Speed & RPM | $\mathrm{RPM} = \left(\frac{v}{r_{\mathrm{wheel}}}\right) \cdot \mathrm{Ratio}\_{\mathrm{gear}} \cdot \mathrm{FinalDrive} \cdot \left(\frac{60}{2\pi}\right)$ |
| | Wheel Driving Force | $F_{\mathrm{engine}} = \frac{\tau_{\mathrm{engine}} \cdot \mathrm{Ratio}\_{\mathrm{gear}} \cdot \mathrm{FinalDrive}}{r\_{\mathrm{wheel}}}$ |
| | Passive Engine Braking | $F_{\mathrm{engine\_brake}} = \left(\frac{\mathrm{RPM}}{\mathrm{RPM}\_{\mathrm{redline}}}\right) \cdot F\_{\mathrm{brake,max}}$ |
| **Predictive AI** | Corner Speed Limit | $v_{\mathrm{corner}}^2 = \frac{m \cdot g \cdot \cos\theta \cdot \mu_{\mathrm{base}}}{\left(\frac{m}{R}\right) - \left(\frac{1}{2}\rho \cdot C_L A \cdot \mu_{\mathrm{base}}\right)}$ |
| | Torricelli Decel Boundary | $v_{\mathrm{critical}} = \sqrt{v_{\mathrm{corner}}^2 + 2 \cdot a_{\mathrm{decel}} \cdot d}$ |
| **Numerical Step** | Euler Integration | $a = \frac{F_{\mathrm{net}}}{m}, \quad v_{t+\Delta t} = v_t + a\Delta t, \quad s_{t+\Delta t} = s_t + v\Delta t$ |

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
