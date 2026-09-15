# F1 High-Performance Physics and Telemetry Simulator

A high-performance Formula 1 simulation engine built from the ground up in C++ and CUDA. This project bridges the gap between theoretical vehicle dynamics and real-world racing by running a custom predictive physics AI against actual track telemetry (extracted via FastF1). 

Designed to process multiple aerodynamic and powertrain configurations concurrently using GPU parallelization, the engine visualizes the physics calculations in real-time via a custom SFML telemetry HUD.

**Note:** A CUDA-enabled NVIDIA GPU is currently strictly required to run this simulation engine.

![Simulation Demo](https://github.com/user-attachments/assets/8e620258-ab30-4dbd-971d-0b837a5ba5d7)


## Equations used
* **Aerodynamic Drag Force:**
  $$F_{\text{drag}} = \frac{1}{2} \rho \cdot v^2 \cdot C_d \cdot A$$
  With DRS active, the aerodynamic drag coefficient is scaled:
  $$C_{d,\text{DRS}} = 0.65 \cdot C_d$$

* **Aerodynamic Downforce:**
  $$F_{\text{downforce}} = \frac{1}{2} \rho \cdot v^2 \cdot C_L \cdot A$$
  Approximated using the overall efficiency coefficient:
  $$C_L \cdot A = 3.0 \cdot C_d \cdot A$$

* **Track Pitch Angle (Elevation Change):**
  $$\theta = \arcsin\left(\frac{\Delta z}{\Delta s}\right) = \arcsin\left(\frac{z_{i+1} - z_i}{\Delta s}\right)$$

* **Total Normal Force:**
  $$F_N = (m \cdot g \cdot \cos\theta) + F_{\text{downforce}}$$

* **Longitudinal Gravity Component:**
  $$F_{g,\text{long}} = -m \cdot g \cdot \sin\theta$$

* **Maximum Available Tire Grip:**
  $$F_{\text{grip,max}} = F_N \cdot \mu_{\text{base}}$$

* **Cornering Centrifugal / Lateral Force:**
  $$F_{\text{lat}} = \frac{m \cdot v^2}{R}$$

* **Kamm's Friction Circle (Longitudinal Budget):**
  $$F_{\text{long,grip}} = \sqrt{F_{\text{grip,max}}^2 - F_{\text{lat}}^2}$$

* **Rear-Wheel-Drive (RWD) Traction Constraint:**
  $$F_{\text{traction,max}} = 0.55 \cdot F_{\text{long,grip}}$$

* **Wheel Angular Velocity & Engine RPM:**
  $$\omega_{\text{wheel}} = \frac{v}{r_{\text{wheel}}}$$
  $$\text{RPM} = \omega_{\text{wheel}} \cdot \text{Ratio}_{\text{gear}} \cdot \text{FinalDrive} \cdot \left(\frac{60}{2\pi}\right)$$

* **Engine Torque & Contact Patch Driving Force:**
  $$\tau_{\text{engine}} = \frac{P_{\text{total}}}{\omega_{\text{engine}}}$$
  $$F_{\text{engine}} = \frac{\tau_{\text{engine}} \cdot \text{Ratio}_{\text{gear}} \cdot \text{FinalDrive}}{r_{\text{wheel}}}$$

* **Engine Braking Passive Force:**
  $$F_{\text{engine\_braking}} = \left(\frac{\text{RPM}}{\text{RPM}_{\text{redline}}}\right) \cdot F_{\text{engine\_braking,max}}$$

* **Corner Velocity by Radial Force Equilibrium:**
  $$\frac{m \cdot v_{\text{corner}}^2}{R} = \mu_{\text{base}} \left(m \cdot g \cdot \cos\theta + \frac{1}{2}\rho \cdot v_{\text{corner}}^2 \cdot C_L A\right)$$
  Solving algebraically for $v_{\text{corner}}^2$:
  $$v_{\text{corner}}^2 = \frac{m \cdot g \cdot \cos\theta \cdot \mu_{\text{base}}}{\left(\frac{m}{R}\right) - \left(\frac{1}{2}\rho \cdot C_L A \cdot \mu_{\text{base}}\right)}$$

* **Torricelli Critical Braking Threshold:**
  $$v_{\text{critical}} = \sqrt{v_{\text{corner}}^2 + 2 \cdot a_{\text{decel}} \cdot d}$$

* **Translational Motion Integration (Explicit Euler):**
  $$a = \frac{F_{\text{net}}}{m}$$
  $$v(t + \Delta t) = v(t) + a \cdot \Delta t$$
  $$s(t + \Delta t) = s(t) + v \cdot \Delta t$$

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
