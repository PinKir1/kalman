# Kalman  
Completed

## Project Overview  
This repository presents an implementation and study of the Kalman filter algorithm for signal processing and noise reduction. The project aims to demonstrate how variations in filter parameters influence its performance in smoothing noisy signals.

The Kalman filter is a widely-used mathematical tool for:
- Filtering out measurement noise from signals
- Estimating states in dynamic systems
- Enhancing measurement accuracy

This implementation focuses on applying the Kalman filter to a sinusoidal signal contaminated with Gaussian noise to showcase its noise reduction capabilities.  
![image](https://github.com/user-attachments/assets/60180db8-71b2-41c7-b88b-be0cb9609755)

## Features
- Comparison of noise variance before and after applying the filter
- Visualization of filter performance using `matplotlib`
- Exploration of the impact of different filter parameters:
  - Process noise covariance (Q)
  - Measurement noise covariance (R)
  - Initial error covariance (P)
  - State transition matrix (F)
  - Measurement matrix (H)
- Python-based implementation of a basic Kalman filter class
- Interactive plots displaying:
  - The original signal
  - Noisy measurements
  - The filtered signal

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Usage

1. Install the required packages using `pip`:
```bash
pip install numpy matplotlib
```

2. Execute the main script:
```bash
python kalman_filter.py
```
