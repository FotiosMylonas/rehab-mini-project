# Rehab Mini Project — IMU Step Counting & Movement Frequency

Small Python mini-project demonstrating an end-to-end pipeline for wearable sensor processing relevant to home rehabilitation:

1) Extract IMU files from local Zenodo zip downloads  
2) Load raw foot IMU time-series data  
3) Butterworth low-pass filter (clinical movement band)  
4) Peak-based step detection  
5) Compute cadence (steps/min) and movement frequency (Hz)  
6) Save a plot + JSON summary metrics

## How to run
### 1) Create & activate a virtual environment

python -m venv .VirtualEnvironment
.\.VirtualEnvironment\Scripts\Activate.ps1