# Data Collection Client

This folder contains the Arduino implementation for collecting accelerometer data for the TinyML-Federated project.

## Overview

The data collection client is designed to run on an Arduino Nano 33 BLE Sense and captures accelerometer data when triggered by the data collection server. The client:

1. Waits for a command from the server via Serial connection
2. Records 256 samples from the accelerometer at 100Hz (approximately 2.56 seconds of data)
3. Transmits the data back to the server in real-time

## Hardware Requirements

- Arduino Nano 33 BLE Sense
- USB cable for connecting to computer

## Installation

### Using Arduino IDE

1. Install the [Arduino IDE](https://www.arduino.cc/en/software)
2. Install the Arduino Nano 33 BLE board support package:
   - Open the IDE
   - Go to Tools > Board > Boards Manager
   - Search for "Arduino Nano 33 BLE" and install
3. Install required libraries:
   - Go to Sketch > Include Library > Manage Libraries
   - Search for and install "Arduino_LSM9DS1"
4. Connect your Arduino Nano 33 BLE Sense to your computer
5. Open `data-collection.ino` in Arduino IDE
6. Select the correct board and port under Tools menu
7. Click the upload button (right arrow)

### Using PlatformIO

1. Install [PlatformIO](https://platformio.org/install)
2. Open PlatformIO in your preferred IDE (e.g., VS Code)
3. Create a new project with the following settings:
   - Board: Arduino Nano 33 BLE
   - Framework: Arduino
4. Replace the generated code with the contents of `data-collection.ino`
5. Add dependencies in `platformio.ini`:
   ```ini
   lib_deps = 
       arduino-libraries/Arduino_LSM9DS1
   ```
6. Connect your Arduino and upload the code

## Understanding the Code

The main functionality is contained in `data-collection.ino`:

- `setup()`: Initializes Serial communication and the IMU sensor
- `loop()`: Waits for the 'r' command from the server to start recording
- `recordData()`: Records 256 samples from the accelerometer at 100Hz and sends them to the server

Key parameters:
- `SAMPLES`: Number of data points to collect (256)
- `SAMPLING_FREQ`: Frequency in Hz for data collection (100Hz)
- `SAMPLING_PERIOD_MS`: Time between samples (10ms)

## Usage

1. Upload the code to your Arduino Nano 33 BLE Sense
2. Keep the Arduino connected to your computer via USB
3. Run the data collection server (see [data-collection-server](../data-collection-server/README.md))
4. The Arduino will wait for commands from the server
5. When the server sends the 'r' command, the Arduino will collect accelerometer data and transmit it

## Customization

You can modify the following parameters in the code:
- `SAMPLES`: Change the number of samples collected
- `SAMPLING_FREQ`: Adjust the sampling frequency
- `SAMPLING_PERIOD_MS`: Automatically calculated from sampling frequency

## Troubleshooting

- **IMU initialization fails**: Ensure the Arduino is properly connected and the board is correctly selected in your IDE
- **No data received**: Check Serial port configuration in both client and server (baud rate should be 115200)
- **Inconsistent sampling rate**: The code uses a simple timing approach; for more precise timing consider using hardware timers

## Example Data Format

The Arduino sends data in the following format:

```
START
x,y,z
x,y,z
...
x,y,z
END
```

Where:
- `START` and `END` are markers for the server to identify the data
- `x`, `y`, and `z` are the accelerometer readings in m/s² for each axis

## Next Steps

After collecting data, move on to the [data-collection-server](../data-collection-server/README.md) to label and manage your datasets.