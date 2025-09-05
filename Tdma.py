import numpy as np
import matplotlib.pyplot as plt

#  given Parameters
Lx = 0.3  # plate width given in  m
Ly = 0.4  # plate height given in m
Lz = 0.01  # plate thickness given in m
k = 1000.0  # Thermal conductivity given in W/m.K
h = 200.0  # Heat transfer coefficient given in  W/m²·K
T_inf = 20.0  # Ambient temperature, °C
# Boundary Conditions
Tn = 100.0  # Fixed temperature at North boundary, °C
qw = 500e+3  # Given heat flux at West boundary, W/m²
# Discretization
Nx = 3  #  nodes in x-direction
Ny = 4  #  nodes in y-direction
dx = Lx / Nx
dy = Ly / Ny
# Area for heat transfer
Area = Lz * dy
# Equivalent thermal resistance at the East boundary (mixed boundary condition)
R_eq = h / (1 + (h * dx) / (2 * k))

# Initializing  temp field
T = np.zeros((Ny, Nx))
# Initializing boundary coefficients
aN = k * Area / dy
aS = k * Area / dy
aE = k * Area / dx
aW = k * Area / dx
# top and bottom boundary source term
bN = 2 * k * Area / dy  # Constant Temp at the top
SpN = -2 * k * Area / dy  #  North boundary penalty term
# Initializing  matrices for tdma
A = np.zeros((Ny, Ny))
b = np.zeros(Ny)
# solving tridiagonal systems by using tdma solver function
def TDMA(a, b):
    n = len(b)
    P = np.zeros(n)
    Q = np.zeros(n)
    x = np.zeros(n)
    # Forward elimination
    P[0] = -a[0, 1] / a[0, 0]
    Q[0] = b[0] / a[0, 0]
    for i in range(1, n):
        denominator = a[i, i] + a[i, i - 1] * P[i - 1]
        if i < n - 1:
            P[i] = -a[i, i + 1] / denominator
        Q[i] = (b[i] - a[i, i - 1] * Q[i - 1]) / denominator
    # Back substitution
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x
# Iteration loop for TDMA method
tol = 1e-3
error = 1.0
iterations = 0
while error > tol:
    T_old = np.copy(T)
    #  applying Loop over each column in the x-direction
    for i in range(Nx):
        if i == 0:  # West boundary that is  heat flux condition
            TE = T[:, i + 1]  # East neighbor
            aE = k * Area / dx
            aW = 0.0
            bw = qw * Area
            TW = np.zeros(Ny)  # No West neighbor (boundary condition)
        elif i == Nx - 1:  # East boundary (convection)
            TE = np.zeros(Ny)  # No East neighbor (boundary condition)
            TW = T[:, i - 1]  # West neighbor
            aW = k * Area / dx
            aE = R_eq * dy * Lz  # Use equivalent resistance at East boundary
            bw = R_eq * dy * Lz * T_inf
        else:  # Interior nodes
            TE = T[:, i + 1]
            TW = T[:, i - 1]
            aW = k * Area / dx
            aE = k * Area / dx
            bw = 0.0
        # Bottom node (j=0)
        A[0, 0] = aW + aE + aN
        A[0, 1] = -aN
        b[0] = aW * TW[0] + aE * TE[0] + bw
        # Interior nodes (j=1 to Ny-2)
        for j in range(1, Ny - 1):
            A[j, j - 1] = -aS
            A[j, j] = aW + aE + aS + aN
            A[j, j + 1] = -aN
            b[j] = aW * TW[j] + aE * TE[j] + bw
        # Top node (j=Ny-1)
        A[Ny - 1, Ny - 2] = -aS
        A[Ny - 1, Ny - 1] = aW + aE + aS - SpN
        b[Ny - 1] = aW * TW[Ny - 1] + aE * TE[Ny - 1] + bw + bN * Tn
        # Solve using TDMA
        T[:, i] = TDMA(A, b)
    iterations += 1
    error = np.max(np.abs(T - T_old))
print(f'Solution converged in {iterations} iterations using TDMA.')
# Plotting the result
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
# Contour plot
plt.figure(1)
plt.contourf(X, Y, T, levels=10, cmap='magma')
plt.colorbar()
plt.title('Temperature Distribution (°C)')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
# Displaying  tempe in an image form
plt.figure(2)
plt.imshow(T, origin='lower', extent=[0, Lx, 0, Ly], cmap='magma')
plt.colorbar()
plt.title('Temperature Distribution (°C)')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.show()
# Postprocessing: East boundary temperatures
print('East Boundary Temperatures:')
for j in range(Ny):
    T_east = T[j, Nx - 1]  # Temperature at East boundary
    T_b_value = (h * T_inf + 2 * k * T_east / dx) / (h + 2 * k / dx)
    print(f'T_b at node {j + 1}: {T_b_value:.2f} °C')
