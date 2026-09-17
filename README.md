# F1 High-Performance Physics and Telemetry Simulator

A high-performance Formula 1 simulation engine built from the ground up in C++ and CUDA. This project bridges the gap between theoretical vehicle dynamics and real-world racing by running a custom predictive physics AI against actual track telemetry (extracted via FastF1). 

Designed to process multiple aerodynamic and powertrain configurations concurrently using GPU parallelization, the engine visualizes the physics calculations in real-time via a custom SFML telemetry HUD.

**Note:** A CUDA-enabled NVIDIA GPU is currently strictly required to run this simulation engine.

# Demo Simulation
<img width="1280" height="720" alt="Kooha-2026-09-15-19-59-34" src="https://github.com/user-attachments/assets/8e9bb48c-b047-46be-bf78-c386b4e02b25" />

# Current state of the project
We are testing only on Max Verstappen's pole position around Monza in 2025, with a time of **01:18.792** our model currently simulates a full lap with starter paramenters equal to the real data in **01:11:348** meaning we have a seven to eight seconds quicker than reality permits. Below this segment it's available all equations currently being used to limit and run the car around the track to simulate a f1 car. This simulator models what a single-seater would do in an **ideal, frictionless vacuum**, and the driver having **godlike reflexes** as of this moment.

For aerodynamics, the equations mentioned below are **Drag Force**, that works as a natural brake imposed by the athmosphere, it defines the top speed of the car in a straight line withoout this the car would be a rocket. **Drag Reduction System or DRS** for short is a 35% cut in aero drag in permited zones (this is for the 2022-2025 regulations), allowing to gain 15km/h to 25km/h in the permitted zones. The last one is **Downforce** which pushes the single-seater against the ground according to its speed. This force generates vertical load without adding inertial mass.

For Geometry and Loads, we have the **Pitch Angle** that measures the pitch of the track in each segment(1m) it allows the model to check if he is going up or down, we also have the **Normal Load** that quantifies the total force that compresses the tires against the track, **Longitudinal Gravity** simulates the weight component that actuates on the movement axis and finally the **Longitudinal Mass Transfer** that simulates the chassis inertia in dynamic transitions making the limit of traction of the wheels dependent of the suspension sinking.

Tyres Dynamics include **Maximum Grip** that estabilish the theoretical ceilling of force that the 4 tyres can exchange with the asphalt before loosing traction completely, **Lateral Cornering Load** calculates the radial inertia that tries to push the car off the curvature R on the track. **The Kahm Circle Long. Limit** manages the grip available in the corner, and then we have the rwd traction limit that uses the rear_ratio supporting a RWD traction car.

For the Powertrain we have the **Angular Velocity and Rotation**, that connects the speed of the car to physical rotation through the gear ratios. This tells us the exact moment we need to switch gears. **Wheel Driving Force** converts the kilowatts from both the ICE and the MGU-K em Newtons. We also have **Passive Engine Braking** that simulates the resistance by cylinder compression and internal friction to decelerate.

For the Predictive AI and Numerical Step, the **Corner Speed Limit** allows our simulation to antecipate what velocity of a future curve of any given radius, by isolating the velocity, we eliminate the necessity of more calculations, the **Torricelli Decel Boundary** gives the sim driver the vision of the track computing the distance d until the next curve and evaluates the braking point making the driver transition from accelerating to braking. Finally we have the Explicit Euler Integration, for our sim driver and real driver to appear on screen that runs 500 times per second (dt = 0.002s). This computes all of our variables, converts it to acceleration and projects the velocity and position of our car to the next instant.

Things we need to add to make it more humanlike: 
* Reaction Time (Humans are not godlike)
* Pedals are instantaneous aswell (it takes sometime so make the brake 100%)
* Load Sensitivity/Degressivity, rubber suffers from load degressivity meaning 70 to 80% of max grip is available
* The car is a 2D mass point currently.
* Gear changes are instant meaning we are not cutting the power when upshifting for a little bit of time
* MGU-K in real life has a clipping built in (did not know that)
* Suspension? curbs etc?

### Mathematical Modeling & Physics Formulation

| Domain | Phenomenon | Formulation |
| :--- | :--- | :--- |
| **Aerodynamics** | Drag Force | $F_{\mathrm{drag}} = \frac{1}{2} \rho \cdot v^2 \cdot C_d \cdot A$ |
| | DRS Active Drag | $C_{d,\mathrm{DRS}} = 0.65 \cdot C_d$ |
| | Downforce Generation | $F_{\mathrm{downforce}} = \frac{1}{2} \rho \cdot v^2 \cdot (3.0 \cdot C_d \cdot A)$ |
| **Track & Loads** | Pitch Angle | $\theta = \arcsin\left(\frac{z_{i+1} - z_i}{\Delta s}\right)$ |
| | Normal Load ($F_N$) | $F_N = (m \cdot g \cdot \cos\theta) + F_{\mathrm{downforce}}$ |
| | Longitudinal Gravity | $F_{g,\mathrm{long}} = -m \cdot g \cdot \sin\theta$ |
| | Longitudinal Mass Transfer | $\Delta F_z = \frac{m \cdot a \cdot h_{cg}}{L}$
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
