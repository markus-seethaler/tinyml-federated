# Federated Client

This folder contains the Arduino implementation for the federated learning client in the TinyML-Federated project.

## Overview

The federated client runs on Arduino Nano 33 BLE Sense devices and implements:

1. **BLE Communication** - Exchanges model weights and commands with the server
2. **Signal Processing** - Extracts features from raw accelerometer data
3. **Neural Network** - Performs on-device training and inference
4. **Theft Detection** - Classifies sensor readings into three classes:
   - No theft attempt
   - Carrying away attempt
   - Lock breaking attempt

## Hardware Requirements

- Arduino Nano 33 BLE Sense
- USB cable for programming
- Power source (for standalone operation)

## Software Dependencies

- Arduino IDE (1.8.13+) or PlatformIO
- Required Arduino libraries:
  - ArduinoBLE
  - Arduino_LSM9DS1
  - ArduinoFFT
  - NeuralNetwork

## Installation

### Using Arduino IDE

1. Install the [Arduino IDE](https://www.arduino.cc/en/software)
2. Install the Arduino Nano 33 BLE board support package:
   - Go to Tools > Board > Boards Manager
   - Search for "Arduino Nano 33 BLE" and install
3. Install required libraries:
   - Go to Sketch > Include Library > Manage Libraries
   - Search for and install:
     - "ArduinoBLE"
     - "Arduino_LSM9DS1"
     - "ArduinoFFT"
     - "NeuralNetwork"
4. Connect your Arduino Nano 33 BLE Sense to your computer
5. Open `SmartBikeLock.ino` in Arduino IDE
6. Select the correct board and port under Tools menu
7. Click the upload button (right arrow)

### Using PlatformIO

1. Install [PlatformIO](https://platformio.org/install)
2. Open PlatformIO in your preferred IDE (e.g., VS Code)
3. Create a new project with the following settings:
   - Board: Arduino Nano 33 BLE
   - Framework: Arduino
4. Copy all the .cpp and .h files to the src directory
5. Add dependencies in `platformio.ini`:
   ```ini
   lib_deps = 
       arduino-libraries/ArduinoBLE
       arduino-libraries/Arduino_LSM9DS1
       kosme/arduinoFFT
       https://github.com/GiorgosXou/NeuralNetworks
   ```
6. Connect your Arduino and upload the code

## Project Structure

The client implementation consists of several modular components:

- `SmartBikeLock.ino` - Main Arduino sketch with setup and loop functions
- `Communication.h/cpp` - BLE communication interface
- `Config.h` - Configuration parameters for NN, signal processing, and BLE
- `NeuralNetworkBikeLock.h/cpp` - Neural network wrapper for bike lock application
- `SignalProcessing.h/cpp` - Feature extraction from accelerometer data
- `TimingBenchmark.h` - Optional benchmarking tools for performance evaluation

## Key Components

### Communication Module

Handles BLE communication with the federated server:
- Establishes BLE connection
- Sends and receives model weights
- Sends prediction results
- Receives commands and training labels

### Signal Processing Module

Processes raw accelerometer data:
- Collects samples at specified frequency
- Applies Fast Fourier Transform (FFT)
- Extracts frequency domain features
- Calculates statistical features (mean, max, variance)

### Neural Network Module

Manages the on-device neural network:
- Handles model initialization
- Performs inference for theft detection
- Supports on-device training
- Manages model weights

## Configuration

Key parameters can be adjusted in `Config.h`:

### Neural Network Configuration
- `NUM_LAYERS` - Number of layers in the neural network
- `LAYERS` - Array specifying the size of each layer
- `MAX_EPOCHS` - Maximum number of training epochs
- `ERROR_THRESHOLD` - Convergence threshold for training

### Signal Processing Configuration
- `SAMPLES` - Number of accelerometer samples to collect (256)
- `SAMPLING_FREQ` - Frequency for data collection (100Hz)
- `FEATURE_BINS` - Number of frequency bins for feature extraction
- `FREQ_BANDS` - Frequency band boundaries for binning

### BLE Configuration
- `DEVICE_NAME` - Name of the BLE device
- `SERVICE_UUID` - UUID for the main service
- `CHUNK_SIZE_RECEIVE/SEND` - Sizes for chunked data transfer

## Usage

### Commands

The federated client responds to the following commands from the server:

1. `GET_WEIGHTS` - Send current model weights to the server
2. `SET_WEIGHTS` - Receive new model weights from the server
3. `START_TRAINING` - Collect data and perform on-device training
4. `START_CLASSIFICATION` - Perform inference and send prediction results

### Operation Modes

1. **Classification Mode**
   - Triggered by `START_CLASSIFICATION` command
   - Collects and processes accelerometer data
   - Performs inference to detect theft attempts
   - Sends prediction probabilities to the server

2. **Training Mode**
   - Triggered by `START_TRAINING` command
   - Collects and processes accelerometer data
   - Uses the label provided by the server
   - Performs on-device backpropagation
   - Can be part of a federated learning round

## Customization

- To modify the neural network architecture, adjust the `LAYERS` array in `Config.h`
- To change feature extraction parameters, update the `SignalConfig` namespace in `Config.h`
- To optimize BLE communication, adjust the parameters in `BLEConfig` namespace in `Config.h`

## Troubleshooting

- **BLE connection issues**: Check that the server is using the correct MAC address
- **Training errors**: Verify that labels are being correctly sent from the server
- **Sensor errors**: Ensure the IMU is properly initialized
- **Memory issues**: If you experience crashes, try reducing the network size or feature counts

## Next Steps

After deploying the federated client, use the [federated-server](../federated-server) to orchestrate the federated learning process