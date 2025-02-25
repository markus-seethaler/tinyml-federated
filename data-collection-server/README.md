# Data Collection Server

This folder contains the Python server implementation for collecting, labeling, and managing accelerometer data from Arduino devices for the TinyML-Federated project.

## Overview

The data collection server:

1. Establishes a Serial connection with the Arduino client
2. Sends commands to trigger data collection
3. Receives and processes accelerometer data
4. Allows labeling of data for theft detection (no theft, carrying away, lock breaking)
5. Manages a dataset with proper metadata and file organization

## Requirements

- Python 3.7 or higher
- Arduino Nano 33 BLE Sense with the data collection client installed
- USB connection to the Arduino

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:

   ```bash
   pip install pandas pyserial
   ```

## Usage

1. Connect the Arduino running the data collection client to your computer
2. Identify the correct serial port:
   - On Windows, it will be a COM port (e.g., COM3)
   - On Linux, it will be something like `/dev/ttyACM0`
   - On macOS, it will be something like `/dev/cu.usbmodem14101`

3. Update the `PORT` variable in `main.py` with your serial port

4. Run the server:
   ```bash
   python main.py
   ```

5. Follow the interactive menu to:
   - Record new samples
   - Delete samples
   - View class distribution
   - Exit the program

## Interactive Commands

The server provides an interactive menu with the following options:

- **1: Record new sample**
  - Sends a command to the Arduino to start recording
  - After data collection, prompts you to label the data (0: No theft, 1: Carrying away, 2: Lock breaking)
  - Saves the data and updates the metadata

- **2: Delete sample**
  - Shows a list of current samples
  - Allows you to delete specific samples or ranges of samples
  - Maintains proper indexing and file organization

- **3: Display class distribution**
  - Shows the current distribution of classes in the dataset
  - Helpful for ensuring balanced or unbalanced datasets

- **4: Exit**
  - Exits the program

## Data Structure

The server creates two main data structures:

1. **Individual CSV Files**
   - Located in the `motion_data` folder
   - Each file contains one recording session (256 samples)
   - Format: `timestamp_ms,acc_x,acc_y,acc_z`

2. **Metadata CSV File**
   - Named `motion_metadata.csv` in the root folder
   - Contains information about each recording:
     - `sample_id`: Unique identifier for each sample
     - `timestamp`: When the recording was made
     - `label`: Class label (0: No theft, 1: Carrying away, 2: Lock breaking)
     - `filename`: Reference to the CSV file containing the raw data

## Example Workflow

1. Mount the Arduino on a bike lock in a position that allows detecting movement
2. Run the server and select option 1 to start recording
3. Simulate a "no theft" situation (normal state) and label as 0
4. Record several samples in this state
5. Simulate a "carrying away" attempt and label as 1
6. Record several samples in this state
7. Simulate a "lock breaking" attempt and label as 2
8. Record several samples in this state
9.  Use option 2 to delete any anomalous or incorrect recordings

## Data Management Tips

- Record data in various positions and orientations for better generalization
- Include variations in movement intensity and patterns
- Consider collecting data in different environmental conditions

## Troubleshooting

- **Serial port not found**: Check the connection and update the `PORT` variable in `main.py`
- **Connection errors**: Ensure the Arduino is properly programmed with the data collection client
- **Data recording issues**: Verify the baud rate matches between client and server (should be 115200)

## Next Steps

After collecting a sufficient dataset, you can move on to:

1. Analyzing the data for feature extraction
2. Import the data into the simulation environment