import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constants
dt = 0.1  # time step
horizon = 10  # prediction horizon
lambda_u = 0.1  # control penalty in cost function

# Target position
x_target = np.array([10, 10])  # Target position for the robot

# Initial position and orientation
x_initial = np.array([0, 0, 0])  # [x, y, theta]

# System Dynamics for the mobile robot
def system_dynamics(x, u, dt):
    """
    x: current state [x, y, theta]
    u: control input [v, omega]
    dt: time step
    """
    x_next = np.zeros(3)
    x_next[0] = x[0] + u[0] * np.cos(x[2]) * dt  # x position
    x_next[1] = x[1] + u[0] * np.sin(x[2]) * dt  # y position
    x_next[2] = x[2] + u[1] * dt  # orientation theta
    return x_next

# Generate a Reference Trajectory
def generate_reference_trajectory(x_current, x_target, horizon):
    """
    Generates a straight-line trajectory from the current position to the target
    over the given horizon.
    """
    trajectory = np.linspace(x_current[:2], x_target[:2], horizon)
    return trajectory

# Cost Function
def cost_function(u_flat, x_current, x_target, horizon, dt):
    """
    u_flat: flattened control inputs [v, omega] over the prediction horizon
    x_current: current state [x, y, theta]
    reference_trajectory: desired positions over the prediction horizon
    """
    u = u_flat.reshape(horizon, 2)  # reshape to [v, omega]
    x = np.copy(x_current)
    cost = 0.0
    
    for i in range(horizon):
        x = system_dynamics(x, u[i], dt)  # predict next state
        cost += np.sum((x[:2] - reference_trajectory)**2)  # position error squared
        cost += lambda_u * (u[i][0]**2 + u[i][1]**2)  # control penalty
    
    return cost

# Constraints (control inputs)
def control_constraints(u_flat):
    u = u_flat.reshape(horizon, 2)
    constraints = []
    for i in range(horizon):
        constraints.append(u[i][0] - 1.0)  # max speed constraint
        constraints.append(-u[i][0] - 1.0)  # min speed constraint
        constraints.append(u[i][1] - 0.5)  # max angular velocity
        constraints.append(-u[i][1] - 0.5)  # min angular velocity
    return constraints

# MPC Optimization function
def mpc_optimize(x_current, x_target, horizon, dt):
    u_initial = np.zeros(2 * horizon)  # initial guess for [v, omega] over the horizon
    result = minimize(cost_function, u_initial, args=(x_current, x_target, horizon, dt),
                      constraints={'type': 'ineq', 'fun': control_constraints})
    u_optimal = result.x.reshape(horizon, 2)
    return u_optimal

# Simulation Loop
x_current = np.copy(x_initial)
x_history = [x_current[:2]]
u_history = []

for t in range(100):  # simulate for 100 time steps
    u_optimal = mpc_optimize(x_current, x_target, horizon, dt)
    u = u_optimal[0]  # apply the first control input
    x_current = system_dynamics(x_current, u, dt)
    
    # Record history for plotting
    x_history.append(x_current[:2])
    u_history.append(u)
    
    # Check if close enough to target
    if np.linalg.norm(x_current[:2] - x_target) < 0.1:
        print("Target reached")
        break

# Convert history to numpy arrays for plotting
x_history = np.array(x_history)
u_history = np.array(u_history)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_history[:, 0], x_history[:, 1], '-o', label="Path")
plt.plot(x_target[0], x_target[1], 'rx', label="Target")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("2D Mobile Robot MPC Path")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(u_history[:, 0], label="Linear Velocity (v)")
plt.plot(u_history[:, 1], label="Angular Velocity (omega)")
plt.xlabel("Time Step")
plt.ylabel("Control Inputs")
plt.title("Control Inputs over Time")
plt.legend()
plt.grid()
plt.show()
