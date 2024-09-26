import matplotlib.pyplot as plt
import numpy as np

#Define array geometry as a class object for code reuse
#Eg type of array (ULA, UCA), define highest detectable frequency, define the array reference point, number of array elements 
# UNIFORM LINEAR ARRAY CAN RESOLVE AZIMUTH ANGLES of 180 degrees whereas UNIFORM CIRCULAR ARRAYS CAN RESOLVE ANGLES OF 360 degrees
# BOTH types of arrays are 2D and therefore have limited capability of resolving both azimuth and elevation angles. 

class ArrayGeometry:
    def __init__(self, array_type, frequency, num_elements, nyquist_factor=2, custom_coordinates=None):
        """
        Initialize the ArrayGeometry class.

        Parameters:
        - array_type: str, type of the array ('ULA' for Uniform Linear Array, 'UCA' for Uniform Circular Array)
        - frequency: float, highest signal frequency in Hz
        - num_elements: int, number of antennas in the array
        - nyquist_factor: int, how many times the half-wavelength should be divided (e.g., 2 means spacing = wavelength / 4)
        - custom_coordinates: list of tuples or None, custom coordinates for the array elements
        """
        self.array_type = array_type
        self.frequency = frequency
        self.num_elements = num_elements
        self.nyquist_factor = nyquist_factor
        self.custom_coordinates = custom_coordinates if custom_coordinates is not None else []
        
        self.wavelength = 3e8 / frequency
        self.spacing = self.calculate_spacing()
        self.coordinates = self.calculate_coordinates()

    def calculate_spacing(self):
        """
        Calculate the spacing based on the nyquist_factor.
        """
        if self.nyquist_factor < 1:
            raise ValueError("Nyquist factor must be a positive integer greater than or equal to 1.")
        
        # Spacing is calculated as wavelength divided by twice the nyquist factor
        return self.wavelength / (2 * self.nyquist_factor)

    def calculate_coordinates(self):
        """
        Calculate the coordinates of the array elements based on the array type.
        """
        if self.array_type == 'ULA':
            return self.calculate_ula_coordinates()
        elif self.array_type == 'UCA':
            return self.calculate_uca_coordinates()
        else:
            raise ValueError("Invalid array type. Choose 'ULA' or 'UCA'.")

    def calculate_ula_coordinates(self):
        """
        Calculate the coordinates for a Uniform Linear Array (ULA).
        """
        if self.custom_coordinates:
            return self.custom_coordinates
        
        return [(i * self.spacing, 0, 0) for i in range(self.num_elements)]

    def calculate_uca_coordinates(self):
        """
        Calculate the coordinates for a Uniform Circular Array (UCA).
        """
        if self.custom_coordinates:
            return self.custom_coordinates
        
        radius = self.spacing / (2 * np.sin(np.pi / self.num_elements))
        return [
            (radius * np.cos(2 * np.pi * i / self.num_elements), radius * np.sin(2 * np.pi * i / self.num_elements), 0)
            for i in range(self.num_elements)
        ]

    def print_coordinates(self):
        """
        Print the coordinates of each array element.
        """
        print(f"Coordinates for {self.array_type} with {self.num_elements} elements:")
        for index, coord in enumerate(self.coordinates):
            print(f"Element {index + 1}: X={coord[0]:.3f}, Y={coord[1]:.3f}, Z={coord[2]:.3f}")

    def plot_array(self):
        """
        Plot the array configuration based on its type in 3D.
        """
        if self.array_type == 'ULA':
            self.plot_ula()
        elif self.array_type == 'UCA':
            self.plot_uca()
        else:
            raise ValueError("Invalid array type. Choose 'ULA' or 'UCA'.")

    def plot_ula(self):
        """
        Plot the Uniform Linear Array (ULA) configuration in 3D and mark distances from the reference point.
        """
        x_coords, y_coords, z_coords = zip(*self.coordinates)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, color='b', marker='o')
        ax.scatter([0], [0], [0], color='r', marker='x', label='Reference Point (0,0)')

        # Annotate distances from the reference point
        for i in range(len(self.coordinates)):
            x, y, z = self.coordinates[i]
            distance = np.sqrt(x**2 + y**2 + z**2)
            ax.text(x, y, z, f'{distance:.2f}', color='black', fontsize=8)

        ax.set_title('Uniform Linear Array (ULA) - 3D')
        ax.set_xlabel('X Coordinate (meters)')
        ax.set_ylabel('Y Coordinate (meters)')
        ax.set_zlabel('Z Coordinate (meters)')
        ax.legend()
        plt.show()

    def plot_uca(self):
        """
        Plot the Uniform Circular Array (UCA) configuration in 3D and mark distances from the reference point.
        """
        x_coords, y_coords, z_coords = zip(*self.coordinates)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, color='b', marker='o')
        ax.scatter([0], [0], [0], color='r', marker='x', label='Reference Point (0,0)')

        # Annotate distances from the reference point
        for i in range(len(self.coordinates)):
            x, y, z = self.coordinates[i]
            distance = np.sqrt(x**2 + y**2 + z**2)
            ax.text(x, y, z, f'{distance:.2f}', color='black', fontsize=8)

        ax.set_title('Uniform Circular Array (UCA) - 3D')
        ax.set_xlabel('X Coordinate (meters)')
        ax.set_ylabel('Y Coordinate (meters)')
        ax.set_zlabel('Z Coordinate (meters)')
        ax.legend()
        plt.show()

        