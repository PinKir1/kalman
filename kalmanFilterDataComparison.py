import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from datetime import datetime


class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = np.dot(np.eye(self.P.shape[0]) - np.dot(K, self.H), self.P)
        return self.x


class ExperimentRunner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"kalman_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.frequency = 1
        self.amplitude = 5
        self.sampling_interval = 0.001
        self.total_time = 1

        self.parameters = {
            "Q": [0.1, 10.0],
            "R": [1.0, 20.0],
            "P": [0.1, 10.0],
            "x0": [-2.0, 2.0],
            "offset": [0.0, 5.0],
        }

    def run_experiment(self, Q, R, P, x0, offset):
        time_steps = np.arange(0, self.total_time, self.sampling_interval)
        true_signal = offset + self.amplitude * np.sin(
            2 * np.pi * self.frequency * time_steps
        )
        noisy_signal = true_signal + np.random.normal(0, 2.0, len(true_signal))

        F = np.array([[1]])
        H = np.array([[1]])
        x = np.array([[x0]])
        kf = KalmanFilter(F, H, np.array([[Q]]), np.array([[R]]), np.array([[P]]), x)

        kalman_estimates = [
            kf.update(measurement)[0][0] for measurement in noisy_signal
        ]
        error_before = noisy_signal - true_signal
        error_after = np.array(kalman_estimates) - true_signal

        return {
            "time_steps": time_steps,
            "true_signal": true_signal,
            "noisy_signal": noisy_signal,
            "kalman_estimates": kalman_estimates,
            "variance_before": np.var(error_before),
            "variance_after": np.var(error_after),
            "mse_before": np.mean(error_before**2),
            "mse_after": np.mean(error_after**2),
            "convergence_time": self.get_convergence_time(error_after),
        }

    def get_convergence_time(self, error, threshold=0.1):
        indices = np.where(np.abs(error) < threshold)[0]
        return (
            indices[0] * self.sampling_interval if indices.size > 0 else self.total_time
        )

    def save_plot(self, results, params):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(
            results["time_steps"],
            results["noisy_signal"],
            label="Noisy Signal",
            alpha=0.6,
            color="orange",
        )
        plt.plot(
            results["time_steps"],
            results["true_signal"],
            label="True Signal",
            linestyle="--",
            color="blue",
        )
        plt.plot(
            results["time_steps"],
            results["kalman_estimates"],
            label="Kalman Filter Estimates",
            color="green",
        )
        plt.title(f"Kalman Filter Results: {params}")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(
            results["time_steps"],
            np.array(results["kalman_estimates"]) - results["true_signal"],
            label="Estimation Error",
            color="red",
        )
        plt.axhline(0, linestyle="--", color="black")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.legend()
        plt.grid()

        filename = f"kalman_{'_'.join(f'{k}={v}' for k, v in params.items())}.png"
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
        return filename

    def run_all(self):
        results_file = os.path.join(self.results_dir, "results.txt")
        param_combinations = product(*self.parameters.values())

        with open(results_file, "w") as f:
            f.write("Kalman Experiment Results\n")
            f.write("=" * 50 + "\n\n")

            for params in param_combinations:
                param_dict = dict(zip(self.parameters.keys(), params))
                results = self.run_experiment(**param_dict)
                plot_file = self.save_plot(results, param_dict)

                f.write(f"Parameters: {param_dict}\n")
                f.write(
                    f"Variance Before: {results['variance_before']:.2f}, After: {results['variance_after']:.2f}\n"
                )
                f.write(
                    f"MSE Before: {results['mse_before']:.2f}, After: {results['mse_after']:.2f}\n"
                )
                f.write(f"Convergence Time: {results['convergence_time']:.3f} s\n")
                f.write(f"Saved Plot: {plot_file}\n")
                f.write("=" * 50 + "\n\n")


runner = ExperimentRunner()
runner.run_all()
print(f"Results saved in: {runner.results_dir}")
