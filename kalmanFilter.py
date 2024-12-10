import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        return self.x


def generate_noisy_signal(base_signal, noise_std):
    return base_signal + np.random.normal(0, noise_std, len(base_signal))


t = np.arange(0, 1, 0.001)
true_signal = 10 + 5 * np.sin(2 * np.pi * t)
initial_noise_std = 2.0
noisy_signal = generate_noisy_signal(true_signal, initial_noise_std)

F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[1]])
R = np.array([[10]])
P = np.array([[1]])
x = np.array([[0]])

kf = KalmanFilter(F, H, Q, R, P, x)


def apply_kalman_filter(kf, signal):
    estimates = []
    for measurement in signal:
        kf.predict()
        estimates.append(kf.update(np.array([[measurement]]))[0, 0])
    return estimates


kalman_estimates = apply_kalman_filter(kf, noisy_signal)

fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.4)

(line_noisy,) = ax.plot(t, noisy_signal, "orange", alpha=0.6, label="Noisy Signal")
(line_true,) = ax.plot(t, true_signal, "b--", label="True Signal")
(line_kalman,) = ax.plot(t, kalman_estimates, "g", label="Kalman Estimate")

ax.grid(True)
ax.legend()
ax.set_title("Kalman Filter Visualization")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")

axQ = plt.axes([0.1, 0.25, 0.65, 0.03])
axR = plt.axes([0.1, 0.20, 0.65, 0.03])
axP = plt.axes([0.1, 0.15, 0.65, 0.03])
axNoise = plt.axes([0.1, 0.10, 0.65, 0.03])

sQ = Slider(axQ, "Q", 0.1, 50.0, valinit=1.0, valstep=0.1)
sR = Slider(axR, "R", 0.1, 50.0, valinit=10.0, valstep=0.1)
sP = Slider(axP, "P", 0.1, 50.0, valinit=1.0, valstep=0.1)
sNoise = Slider(axNoise, "Noise", 0.1, 10.0, valinit=initial_noise_std, valstep=0.1)


def update(val):
    kf.Q = np.array([[sQ.val]])
    kf.R = np.array([[sR.val]])
    kf.P = np.array([[sP.val]])

    new_noisy_signal = generate_noisy_signal(true_signal, sNoise.val)
    kf.x = np.array([[0]])

    new_estimates = apply_kalman_filter(kf, new_noisy_signal)

    line_noisy.set_ydata(new_noisy_signal)
    line_kalman.set_ydata(new_estimates)

    noise_var_before = np.var(new_noisy_signal - true_signal)
    noise_var_after = np.var(new_estimates - true_signal)
    ax.set_title(f"Noise Variance: {noise_var_before:.2f} → {noise_var_after:.2f}")
    fig.canvas.draw_idle()


reset_ax = plt.axes([0.8, 0.15, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset")


def reset(event):
    sQ.reset()
    sR.reset()
    sP.reset()
    sNoise.reset()


reset_button.on_clicked(reset)
sQ.on_changed(update)
sR.on_changed(update)
sP.on_changed(update)
sNoise.on_changed(update)

plt.show()
