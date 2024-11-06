import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Robot parameters
L = 0.5  # Length of the robot (wheelbase)
dt = 0.1  # Time step (sampling period)

# MPC parameters
N = 10  # Prediction horizon
v_max = 1.0  # Max velocity
delta_max = np.pi / 4  # Max steering angle (45 degrees)
Q = np.diag([10, 10, 1])  # State error cost matrix (for position and orientation)
R = np.diag([1, 1])  # Control input cost matrix (for velocity and steering)

# Dynamics of the kinematic bicycle model
def dynamics(x, u):
    """
    x: state [x, y, theta]
    u: control input [v, delta] (velocity, steering angle)
    """
    theta = x[2]
    v = u[0]
    delta = u[1]
    
    x_next = np.zeros(3)
    x_next[0] = x[0] + v * np.cos(theta) * dt
    x_next[1] = x[1] + v * np.sin(theta) * dt
    x_next[2] = x[2] + (v / L) * np.tan(delta) * dt
    return x_next

# Reference trajectory
def reference_trajectory(timesteps):
    """
    Defines the reference trajectory (straight line in this case)
    """
    x_ref = np.linspace(0, 10, timesteps)
    y_ref = np.zeros(timesteps)
    theta_ref = np.zeros(timesteps)
    return np.vstack([x_ref, y_ref, theta_ref]).T

# MPC optimization setup
def mpc_controller(x_init, x_ref, N):
    x = cp.Variable((3, N+1))  # State variables (x, y, theta)
    u = cp.Variable((2, N))    # Control variables (v, delta)
    
    cost = 0
    constraints = []
    
    # Initial condition
    constraints += [x[:, 0] == x_init]
    
    for t in range(N):
        # Cost function (tracking error + control effort)
        cost += cp.quad_form(x[:, t] - x_ref[t], Q) + cp.quad_form(u[:, t], R)
        
        # Dynamics constraint
        constraints += [x[:, t+1] == dynamics(x[:, t], u[:, t])]
        
        # Control input constraints
        constraints += [u[0, t] <= v_max, u[0, t] >= -v_max]
        constraints += [u[1, t] <= delta_max, u[1, t] >= -delta_max]
    
    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Return first control input and predicted trajectory
    return u[:, 0].value, x.value

# Simulate the system using MPC
def simulate_mpc(x_init, timesteps):
    x = x_init
    trajectory = [x_init]
    x_ref = reference_trajectory(timesteps)

    for t in range(timesteps - N):
        u_opt, _ = mpc_controller(x, x_ref[t:t+N], N)
        x = dynamics(x, u_opt)
        trajectory.append(x)
    
    return np.array(trajectory)

# Initial state [x, y, theta]
x_init = np.array([0, 0, 0])

# Simulation parameters
timesteps = 50

# Run the MPC simulation
trajectory = simulate_mpc(x_init, timesteps)

# Plot the trajectory
x_traj = trajectory[:, 0]
y_traj = trajectory[:, 1]
ref_traj = reference_trajectory(timesteps)

plt.plot(x_traj, y_traj, label='MPC Trajectory', linewidth=2)
plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', label='Reference Trajectory', linewidth=2)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Mobile Robot MPC Trajectory')
plt.legend()
plt.grid(True)
plt.show()
