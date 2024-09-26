import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class MUSICAlgorithm:
    def __init__(self, R, array_geometry, snapshots):
        self.R = R
        self.array_geometry = array_geometry
        self.snapshots = snapshots
        self.eigenvalues, self.eigenvectors = self._eigen_decomposition()

    def _eigen_decomposition(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.R)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return eigenvalues, eigenvectors

    def estimate_number_of_sources_mdl(self):
        M = len(self.eigenvalues)
        mdl_values = []

        for p in range(1, M):
            noise_eigenvalues = self.eigenvalues[p:]
            geometric_mean = np.prod(noise_eigenvalues) ** (1 / (M - p))
            arithmetic_mean = np.mean(noise_eigenvalues)
            log_likelihood = -self.snapshots * (M - p) * np.log(geometric_mean / arithmetic_mean)
            mdl = log_likelihood + 0.5 * p * (2 * M - p) * np.log(self.snapshots)
            mdl_values.append(mdl)

        estimated_sources = np.argmin(mdl_values) + 1
        return estimated_sources, mdl_values

    def estimate_noise_subspace(self, num_sources):
        return self.eigenvectors[:, num_sources:]

    def music_spectrum(self, noise_subspace, angles):
        coordinates = np.array(self.array_geometry.coordinates)
        spectrum = []

        for theta in angles:
            a_theta = np.exp(-1j * 2 * np.pi / self.array_geometry.wavelength *
                             (coordinates[:, 0] * np.sin(np.deg2rad(theta)) +
                              coordinates[:, 1] * np.cos(np.deg2rad(theta))))
            spectrum_value = 1 / np.abs(a_theta.conj().T @ noise_subspace @ noise_subspace.conj().T @ a_theta)
            spectrum.append(spectrum_value)

        return np.array(spectrum)

    def estimate_doas(self, spectrum, angles):
        peaks, _ = signal.find_peaks(spectrum)
        return angles[peaks]

    def plot_spectrum(self, spectrum, angles):
        plt.figure(figsize=(10, 6))
        plt.plot(angles, 10 * np.log10(spectrum), label='MUSIC Spectrum')
        plt.title('MUSIC Spectrum')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def root_music(self, noise_subspace):
        M = noise_subspace.shape[0]  # Number of sensors/antennas
        roots = []
        
        # Calculate the noise subspace polynomial
        noise_covariance = noise_subspace @ noise_subspace.conj().T
        polynomial_coeffs = np.zeros(M, dtype=np.complex128)
        for k in range(M):
            polynomial_coeffs[k] = np.sum(noise_covariance.diagonal(k))

        # Find the roots of the polynomial
        polynomial_roots = np.roots(polynomial_coeffs)
        
        # Find the roots closest to the unit circle
        unit_circle_roots = polynomial_roots[np.abs(np.abs(polynomial_roots) - 1).argsort()[:M-1]]
        
        # Convert the roots to angles (DOAs)
        estimated_doas = np.angle(unit_circle_roots)
        estimated_doas = np.sort(np.rad2deg(np.arcsin(estimated_doas / np.pi)))
        
        return estimated_doas