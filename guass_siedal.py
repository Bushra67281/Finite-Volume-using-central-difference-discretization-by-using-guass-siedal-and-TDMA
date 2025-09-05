import numpy as np
import matplotlib.pyplot as plt

# Plate parameters
Lx = 0.3        # Length of the plate (x-direction) in meters
Ly = 0.4        # Height of the plate (y-direction) in meters
Lz = 0.01       # Plate thickness in meters
k = 1000.0      # Thermal conductivity, W/m.K
h = 200.0       # Heat transfer coefficient at the right edge, W/m^2.K
T_amb = 20.0    # Ambient temperature, Celsius

# Boundary conditions
T_north = 100.0     # Fixed temperature at the northern boundary (top edge), °C
q_west = 500e3      # Heat flux at the western boundary (left edge), W/m^2

# Grid parameters
dx = 0.06
dy = dx
Nx = int(Lx / dx)   # Number of grid points along the x-axis
Ny = int(Ly / dy)   # Number of grid points along the y-axis
Area = Lz * dy      # Cross-sectional area perpendicular to the heat flow, m^2

# Equivalent resistance for mixed boundary condition
R_eq = h / (1 + (h * dx) / (2 * k))

# Initialize the coefficient matrix and right-hand side vector
A = np.zeros((Nx * Ny, Nx * Ny))  # Matrix for finite difference coefficients
b = np.zeros(Nx * Ny)             # Right-hand side vector
T_boundary = np.zeros(Ny)         # Temperature on the boundary

# Coefficients for neighboring points
aE = k * Area / dx
aW = k * Area / dx
aN = k * Area / dy
aS = k * Area / dy

# Source terms
bW = q_west * Area
bE = R_eq * dy * Lz * T_amb
bN = 2 * k * Area / dy * T_north
bS = 0  # Insulated boundary at the south
SpN = -2 * k * Area / dy
SpE = -R_eq * dy * Lz

# Bottom-left (south-west) corner
A[0, 0] = aE + aN
A[0, 1] = -aN
A[0, Ny] = -aE
b[0] = bW + bS

# Left boundary (west side)
for j in range(1, Ny - 1):
    A[j, j - 1] = -aS
    A[j, j + 1] = -aN
    A[j, j + Ny] = -aE
    A[j, j] = aE + aN + aS
    b[j] = bW

# Top-left (north-west) corner
idx = Ny - 1
A[idx, idx - 1] = -aS
A[idx, idx + Ny] = -aE
A[idx, idx] = aE + aS - SpN
b[idx] = bW + bN

# Internal nodes
for i in range(1, Nx - 1):
    for j in range(1, Ny - 1):
        idx = i * Ny + j
        A[idx, idx - Ny] = -aW
        A[idx, idx - 1] = -aS
        A[idx, idx] = aW + aE + aN + aS
        A[idx, idx + 1] = -aN
        A[idx, idx + Ny] = -aE

# Bottom-right (south-east) corner
idx = Ny * (Nx - 1)
A[idx, idx - Ny] = -aW
A[idx, idx + 1] = -aN
A[idx, idx] = aW + aN - SpE
b[idx] = bE + bS

# Right boundary (east side)
for j in range(1, Ny - 1):
    idx = Ny * (Nx - 1) + j
    A[idx, idx - Ny] = -aW
    A[idx, idx - 1] = -aS
    A[idx, idx + 1] = -aN
    A[idx, idx] = aW + aN + aS - SpE
    b[idx] = bE

# Top-right (north-east) corner
idx = Nx * Ny - 1
A[idx, idx - Ny] = -aW
A[idx, idx - 1] = -aS
A[idx, idx] = aS + aW - SpN - SpE
b[idx] = bN + bE

# Bottom boundary
for i in range(1, Nx - 1):
    idx = i * Ny
    A[idx, idx - Ny] = -aW
    A[idx, idx] = aW + aE + aN
    A[idx, idx + 1] = -aN
    A[idx, idx + Ny] = -aE
    b[idx] = bS

# Top boundary
for i in range(1, Nx - 1):
    idx = (i + 1) * Ny - 1
    A[idx, idx - Ny] = -aW
    A[idx, idx - 1] = -aS
    A[idx, idx] = aW + aE + aS - SpN
    b[idx] = bN

# Gauss-Seidel method to solve the system
def gauss_seidel(A, b, tol=1e-3, max_iter=10000):
    n = len(b)
    x = np.zeros(n)  # Initial guess
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sigma) / A[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Converged in {k + 1} iterations")
            return x
    print("Gauss-Seidel did not converge within the maximum number of iterations")
    return x

# Solve the linear system
T = gauss_seidel(A, b)

# Reshape the solution into a 2D grid for plotting
T2 = T.reshape((Ny, Nx))

# Generate x and y values for plotting
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Plot the results using contour plot
plt.figure()
plt.contourf(X, Y, T2, 10, cmap='inferno')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution across the Plate')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.axis('equal')
plt.show()

# Plot the results using imagesc-like plot
plt.figure()
plt.imshow(T2, extent=[0, Lx, 0, Ly], origin='lower', cmap='inferno')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution (Heatmap)')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.show()

# Compute boundary temperatures at the east boundary (right side)
for j in range(Ny):
    idx = Ny * (Nx - 1) + j
    T_boundary[j] = (h * T_amb + 2 * k * T[idx] / dx) / (h + 2 * k / dx)
    print(f"T_b at node {idx}: {T_boundary[j]:.2f} °C")
