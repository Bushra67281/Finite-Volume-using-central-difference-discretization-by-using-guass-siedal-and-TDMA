import numpy as np
import matplotlib.pyplot as plt

# Plate properties and boundary conditions
plate_width = 0.3  # Width of the plate, in meters
plate_height = 0.4  # Height of the plate, in meters
plate_thickness = 0.01  # Thickness of the plate, in meters
thermal_conductivity = 1000.0  # Thermal conductivity of material, in W/m.K
north_temp = 100  # Temperature on the North boundary, in degrees Celsius
heat_flux_west = 500e+3  # Heat flux on the West boundary, in W/m^2

# Coordinates in x and y directions
x_vals = np.array([0, 0.033, 0.084, 0.15, 0.216, 0.267, 0.3])
y_vals = np.array([0, 0.031, 0.075, 0.133, 0.2, 0.267, 0.325, 0.369, 0.4])

# Node spacings in x and y directions
dx_vals = np.array([0.033, 0.051, 0.066, 0.066, 0.051, 0.033])
dy_vals = np.array([0.031, 0.044, 0.058, 0.067, 0.067, 0.058, 0.044, 0.031])

nodes_x = len(dx_vals)  # Number of nodes in x-direction
nodes_y = len(dy_vals)  # Number of nodes in y-direction

# Initialize temperature and coefficient matrices
temperature = np.zeros((nodes_x, nodes_y))
coeff_matrix = np.zeros((nodes_y, nodes_y))
rhs_vector = np.zeros((nodes_y, 1))
temp_prev = np.zeros((nodes_x, nodes_y))
residual = np.ones((nodes_x, nodes_y))

# Set tolerance and iteration counter
tolerance = 1e-2
iteration_count = 0

# Iterative loop for temperature calculation
while np.max(np.abs(residual)) > tolerance:
    for i in range(nodes_x):
        for j in range(nodes_y):
            # Boundary conditions and coefficient calculations
            if j == 0:  # South boundary
                north_coeff = thermal_conductivity * dx_vals[i] * plate_thickness / ((dy_vals[j] / 2) + (dy_vals[j + 1] / 2))
                south_coeff = 0
                if i == 0:  # West boundary
                    west_coeff = 0
                    west_rhs = dy_vals[j] * plate_thickness * heat_flux_west
                else:
                    west_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i - 1] / 2))
                    west_rhs = 0
                if i != nodes_x - 1:  # East interior node
                    east_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i + 1] / 2))
                    east_rhs = 0
                else:  # East boundary
                    east_coeff = 0
                    east_rhs = 0

                # Coefficient for current point
                central_coeff = north_coeff + east_coeff + west_coeff + south_coeff
                coeff_matrix[j, j] = central_coeff
                if j + 1 < nodes_y:
                    coeff_matrix[j, j + 1] = -north_coeff

                # Right-hand side terms
                if i == 0:
                    rhs_vector[j, 0] = west_rhs + east_coeff * temperature[i + 1, j]
                elif i != 0 and i != nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_coeff * temperature[i + 1, j]
                elif i == nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_rhs
            elif j != 0 and j != nodes_y - 1:  # Interior nodes
                north_coeff = thermal_conductivity * dx_vals[i] * plate_thickness / ((dy_vals[j] / 2) + (dy_vals[j + 1] / 2))
                south_coeff = thermal_conductivity * dx_vals[i] * plate_thickness / ((dy_vals[j] / 2) + (dy_vals[j - 1] / 2))
                if i != 0:
                    west_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i - 1] / 2))
                    west_rhs = 0
                else:  # West boundary
                    west_coeff = 0
                    west_rhs = dy_vals[j] * plate_thickness * heat_flux_west
                if i != nodes_x - 1:
                    east_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i + 1] / 2))
                else:  # East boundary
                    east_coeff = 0
                    east_rhs = 0

                # Coefficient for current point
                central_coeff = north_coeff + east_coeff + west_coeff + south_coeff
                coeff_matrix[j, j] = central_coeff
                coeff_matrix[j, j - 1] = -south_coeff
                if j + 1 < nodes_y:
                    coeff_matrix[j, j + 1] = -north_coeff

                # Right-hand side terms
                if i == 0:
                    rhs_vector[j, 0] = west_rhs + east_coeff * temperature[i + 1, j]
                elif i != 0 and i != nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_coeff * temperature[i + 1, j]
                elif i == nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_rhs
            elif j == nodes_y - 1:  # North boundary
                north_coeff = thermal_conductivity * dx_vals[i] * plate_thickness / (dy_vals[j] / 2)
                south_coeff = thermal_conductivity * dx_vals[i] * plate_thickness / ((dy_vals[j] / 2) + (dy_vals[j - 1] / 2))
                if i == 0:
                    west_coeff = 0
                    west_rhs = dy_vals[j] * plate_thickness * heat_flux_west
                else:
                    west_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i - 1] / 2))
                if i == nodes_x - 1:
                    east_coeff = 0
                    east_rhs = 0
                else:
                    east_coeff = thermal_conductivity * dy_vals[j] * plate_thickness / ((dx_vals[i] / 2) + (dx_vals[i + 1] / 2))

                # Coefficient for current point
                central_coeff = north_coeff + east_coeff + west_coeff + south_coeff
                coeff_matrix[j, j] = central_coeff
                coeff_matrix[j, j - 1] = -south_coeff

                # Right-hand side terms for north boundary
                if i == 0:
                    rhs_vector[j, 0] = west_rhs + east_coeff * temperature[i + 1, j] + north_coeff * north_temp
                elif i != 0 and i != nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_coeff * temperature[i + 1, j] + north_coeff * north_temp
                elif i == nodes_x - 1:
                    rhs_vector[j, 0] = west_coeff * temperature[i - 1, j] + east_rhs + north_coeff * north_temp

        # Solve for temperature at current row
        temperature[i, :] = np.linalg.solve(coeff_matrix, rhs_vector).flatten()

    # Update residual and previous temperature values
    residual = temperature - temp_prev
    temp_prev = np.copy(temperature)
    iteration_count += 1

# Generate contour plot for temperature distribution
X_mesh, Y_mesh = np.meshgrid(x_vals[0:nodes_x], y_vals[0:nodes_y])

plt.figure()
plt.contourf(X_mesh.T, Y_mesh.T, temperature, 20, cmap="jet")  # Adjust orientation for correct plotting
plt.colorbar()
plt.title('Temperature Distribution (Contour)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

# Generate image plot for temperature distribution
plt.figure()
plt.imshow(temperature.T, extent=[0, plate_width, 0, plate_height], origin='lower', aspect='auto', cmap="jet")
plt.colorbar()
plt.title('Temperature Distribution (Image)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.show()
