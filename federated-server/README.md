# Federated Server

This folder contains the Python server implementation for orchestrating federated learning with Arduino Nano 33 BLE Sense devices in the TinyML-Federated project.

## Overview

The federated server is responsible for:

1. **Client Management** - Connecting to and communicating with the Arduino
2. **Model Distribution** - Sending model weights to the client
3. **Model Aggregation** - Collecting and combining updated models from the client
4. **Training Coordination** - Orchestrating federated learning rounds
5. **Performance Evaluation** - Benchmarking communication and computation

## Requirements

- Python 3.7 or higher
- Arduino Nano 33 BLE Sense device running the federated client
- BLE support on the host computer

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install bleak numpy asyncio
   ```

## Usage

1. Ensure your Arduino device is programmed with the federated client code
2. Power on your Arduino devices
3. Update the `device_address` in `main.py` with your Arduino's BLE address:
   - Find the address in the Serial output when the Arduino starts
   - Format: "xx:xx:xx:xx:xx:xx" (e.g., "ac:3f:75:88:8f:12")
4. Run the server:
   ```bash
   python main.py
   ```
5. Use the interactive menu to control the federated learning process

## Interactive Commands

The server provides an interactive menu with the following options:

- **c**: Start classification
  - Triggers the client to collect data and perform inference
  - Displays the prediction probabilities for all three classes

- **t**: Start training
  - Prompts for a label (0: no theft, 1: carrying away, 2: lock breach)
  - Triggers the client to collect data and train the model with the given label

- **g**: Get weights
  - Retrieves the current model weights from the client
  - Displays the weights organized by network layers

- **s**: Set weights
  - Sends new model weights to the client
  - In this example, it sends random weights (replace with your trained weights)

- **bi**: Run inference benchmark
  - Executes an inference timing benchmark on the client
  - Results are displayed on the Arduino's Serial monitor

- **bt**: Run training benchmark
  - Executes a training timing benchmark on the client
  - Results are displayed on the Arduino's Serial monitor

- **mg**: Measure GET_WEIGHTS performance
  - Tests the speed and reliability of weight retrieval
  - Provides statistics on timing and data rate

- **ms**: Measure SET_WEIGHTS performance
  - Tests the speed and reliability of weight transmission
  - Provides statistics on timing and data rate

- **mb**: Measure both GET and SET performance
  - Combines both measurement tests

- **q**: Quit the program

## Communication Protocol

The server communicates with clients using BLE with the following characteristics:

- **WEIGHTS_READ_CHAR**: For receiving weights FROM Arduino
- **WEIGHTS_WRITE_CHAR**: For sending weights TO Arduino
- **CONTROL_CHAR**: For sending commands
- **LABEL_CHAR**: For sending training labels
- **PREDICTION_CHAR**: For receiving prediction probabilities

Commands are sent as single bytes:
- `1`: GET_WEIGHTS
- `2`: SET_WEIGHTS
- `3`: START_TRAINING
- `4`: START_CLASSIFICATION
- `5`: START_INFERENCE_BENCHMARK
- `6`: START_TRAINING_BENCHMARK

## Implementing Federated Learning

The current implementation only provides the framework for implementing federated learning, as my BLE server module only supported one concurrent connection.
To implement a complete federated learning system with several clients the following code snippets provide a starting point:

1. **Model Initialization**:
   ```python
   # Initialize model with random or pre-trained weights
   initial_weights = np.random.normal(0, 0.01, handler.total_weights).astype(np.float32)
   ```

2. **Federated Learning Round**:
   ```python
   # For each client
   for client_address in client_addresses:
       # Connect to client
       handler = BLECommandHandler(client_address)
       await handler.connect()
       
       # Send current global model
       await handler.send_weights(global_weights)
       
       # Perform local training (several iterations)
       for i in range(local_epochs):
           label = get_appropriate_label()
           await handler.start_training(label)
       
       # Retrieve updated model
       client_weights = await handler.get_weights()
       
       # Store client weights for aggregation
       all_client_weights.append(client_weights)
       
       await handler.disconnect()
   
   # Aggregate models (e.g., using FedAvg)
   global_weights = aggregate_weights(all_client_weights)
   ```

3. **Model Aggregation**:
   ```python
   def aggregate_weights(client_weights_list):
       """Aggregate weights using FedAvg algorithm"""
       # Simple averaging
       return np.mean(client_weights_list, axis=0)
   ```

## Performance Considerations

The server includes timing measurement tools to evaluate:

- **Communication Speed**: Measures throughput in kbit/s for weight transfer
- **Transfer Reliability**: Tests success rate of weight transfers
- **Statistical Analysis**: Provides mean, standard deviation, min, and max times

These measurements are important for optimizing the federated learning process, especially on bandwidth-constrained devices.

## Customization

To adapt the server for your specific needs:

- **Multiple Clients**: Extend the code to connect to and manage multiple devices
- **Model Aggregation**: Implement different aggregation strategies (weighted average, etc.)
- **Client Selection**: Add logic to select which clients participate in each round
- **Hyperparameter Tuning**: Adjust learning rates, local epochs, etc.
- **Model Persistence**: Add code to save and load model weights from disk

## Troubleshooting

- **Connection Issues**: Verify the device address and ensure the Arduino is powered and running the client code
- **Transfer Failures**: Try reducing chunk sizes in both client and server code
- **Timeouts**: Adjust timeout parameters for slower devices
- **BLE Errors**: Some platforms have limitations on BLE packet sizes; adjust chunk sizes accordingly

## Next Steps

After implementing federated learning, you can:

1. Evaluate model performance on different theft detection scenarios
2. Compare federated learning results with centralized approaches
3. Deploy the system on real bike locks

For experimentation without physical devices, see the [federated-simulation](../federated-simulation) component.