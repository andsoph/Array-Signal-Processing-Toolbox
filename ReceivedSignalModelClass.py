import matplotlib.pyplot as plt
import numpy as np

class ReceivedSignalModel:
    def __init__(self, array_geometry, angles, noise_variance, snapshots):
        """
        Initialize the ReceivedSignalModel class.

        Parameters:
        - array_geometry: ArrayGeometry object, contains the array configuration
        - angles: list of float, angles of arrival for each source in degrees
        - noise_variance: float, variance of the noise (assumed to be Gaussian)
        - snapshots: int, number of snapshots (samples) for the received signal
        """
        self.array_geometry = array_geometry
        self.angles = angles
        self.noise_variance = noise_variance
        self.snapshots = snapshots

        # Infer number of sources from the length of angles list
        self.num_sources = len(angles)

        # Extract information from array_geometry
        self.num_antennas = self.array_geometry.num_elements
        self.wavelength = self.array_geometry.wavelength
        self.coordinates = self.array_geometry.coordinates
        
        # Create the steering matrix
        self.steering_matrix = self.compute_steering_matrix()

    def compute_steering_matrix(self):
        """
        Compute the steering matrix A(theta) using the array's geometry.
        Handles both ULA and UCA.
        """
        A = np.zeros((self.num_antennas, self.num_sources), dtype=complex)

        # Loop over each source angle
        for i, theta in enumerate(self.angles):
            # Loop over each antenna element
            for m, (x, y, z) in enumerate(self.coordinates):
                if self.array_geometry.array_type == 'ULA':
                    # For ULA, consider only the x-coordinate
                    A[m, i] = np.exp(-1j * 2 * np.pi / self.wavelength * x * np.sin(np.deg2rad(theta)))
                elif self.array_geometry.array_type == 'UCA':
                    # For UCA, consider both x and y coordinates
                    A[m, i] = np.exp(-1j * 2 * np.pi / self.wavelength * (x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))))
                else:
                    raise ValueError("Invalid array type. Only 'ULA' and 'UCA' are supported.")
        return A
        
    def generate_source_signals(self):
        """
        Generate the source signals s(t) within a specific frequency range.
        
        Returns:
        - s(t): numpy array, source signal matrix of size (num_sources, snapshots)
        """
        # Frequency range in Hz
        f_min = 2.1e9  # Minimum frequency
        f_max = 2.6e9  # Maximum frequency

        # Generate the time vector
        t = np.arange(self.snapshots) / self.snapshots

        # Initialize the signal matrix
        s = np.zeros((self.num_sources, self.snapshots), dtype=complex)

        # Frequency increment to ensure different frequencies for each source
        frequency_increment = (f_max - f_min) / self.num_sources

        # Generate signals
        for i in range(self.num_sources):
            # Calculate the frequency for this source
            frequency = f_min + i * frequency_increment

            # Generate the signal with this frequency
            s[i, :] = np.exp(1j * 2 * np.pi * frequency * t)
        
        return s

    def generate_noise(self):
        """
        Generate the noise matrix n(t).
        
        Returns:
        - n(t): numpy array, noise matrix of size (num_antennas, snapshots)
        """
        noise = np.sqrt(self.noise_variance / 2) * (np.random.randn(self.num_antennas, self.snapshots) + 1j * np.random.randn(self.num_antennas, self.snapshots))
        return noise

    def generate_received_signal(self):
        """
        Generate the received signal y(t).
        
        Returns:
        - y(t): numpy array, received signal matrix of size (num_antennas, snapshots)
        """
        s_t = self.generate_source_signals()
        n_t = self.generate_noise()
        y_t = np.dot(self.steering_matrix, s_t) + n_t
        return y_t

    def compute_covariance_matrix(self, y_t):
        """
        Compute the covariance matrix of the received signal R_y.
        
        Parameters:
        - y_t: numpy array, received signal matrix of size (num_antennas, snapshots)
        
        Returns:
        - R_y: numpy array, covariance matrix of size (num_antennas, num_antennas)
        """
        R_y = (y_t @ y_t.conj().T) / self.snapshots
        return R_y