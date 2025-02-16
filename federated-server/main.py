import asyncio
import time
from statistics import mean, stdev
import numpy as np
import struct
from bleak import BleakClient


class BLETimingHandler:
    def __init__(self, command_handler):
        self.handler = command_handler
        self.get_weights_times = []
        self.send_weights_times = []

    async def measure_get_weights(self, num_trials=5):
        """Measure time taken and data rate to receive weights from Arduino"""
        print(f"\nMeasuring GET_WEIGHTS timing ({num_trials} trials)...")

        # Calculate total data size in bits
        bits_transferred = self.handler.total_weights * 32  # 32 bits per float32

        for i in range(num_trials):
            start_time = time.perf_counter()
            weights = await self.handler.get_weights()
            end_time = time.perf_counter()

            if weights is not None:
                duration = end_time - start_time
                data_rate = bits_transferred / (duration * 1000)  # Convert to kbit/s

                self.get_weights_times.append(duration)
                print(f"Trial {i + 1}: {duration:.3f} seconds, {data_rate:.2f} kbit/s")
            else:
                print(f"Trial {i + 1}: Failed")

            await asyncio.sleep(1)

        if self.get_weights_times:
            avg_time = mean(self.get_weights_times)
            avg_data_rate = bits_transferred / (avg_time * 1000)

            if len(self.get_weights_times) > 1:
                std_dev = stdev(self.get_weights_times)
                print(f"\nGET_WEIGHTS Statistics:")
                print(f"Average time: {avg_time:.3f} seconds")
                print(f"Average data rate: {avg_data_rate:.2f} kbit/s")
                print(f"Standard deviation: {std_dev:.3f} seconds")
                print(f"Min time: {min(self.get_weights_times):.3f} seconds")
                print(f"Max time: {max(self.get_weights_times):.3f} seconds")
            else:
                print(f"\nGET_WEIGHTS Statistics:")
                print(f"Time: {avg_time:.3f} seconds")
                print(f"Data rate: {avg_data_rate:.2f} kbit/s")

    async def measure_send_weights(self, num_trials=5):
        """Measure time taken and data rate to send weights to Arduino"""
        print(f"\nMeasuring SET_WEIGHTS timing ({num_trials} trials)...")

        # Create test weights and calculate total data size in bits
        test_weights = np.random.normal(0, 0.5, self.handler.total_weights).astype(np.float32)
        bits_transferred = self.handler.total_weights * 32  # 32 bits per float32

        for i in range(num_trials):
            start_time = time.perf_counter()
            success = await self.handler.send_weights(test_weights)
            end_time = time.perf_counter()

            if success:
                duration = end_time - start_time
                data_rate = bits_transferred / (duration * 1000)  # Convert to kbit/s

                self.send_weights_times.append(duration)
                print(f"Trial {i + 1}: {duration:.3f} seconds, {data_rate:.2f} kbit/s")
            else:
                print(f"Trial {i + 1}: Failed")

            await asyncio.sleep(1)

        if self.send_weights_times:
            avg_time = mean(self.send_weights_times)
            avg_data_rate = bits_transferred / (avg_time * 1000)

            if len(self.send_weights_times) > 1:
                std_dev = stdev(self.send_weights_times)
                print(f"\nSET_WEIGHTS Statistics:")
                print(f"Average time: {avg_time:.3f} seconds")
                print(f"Average data rate: {avg_data_rate:.2f} kbit/s")
                print(f"Standard deviation: {std_dev:.3f} seconds")
                print(f"Min time: {min(self.send_weights_times):.3f} seconds")
                print(f"Max time: {max(self.send_weights_times):.3f} seconds")
            else:
                print(f"\nSET_WEIGHTS Statistics:")
                print(f"Time: {avg_time:.3f} seconds")
                print(f"Data rate: {avg_data_rate:.2f} kbit/s")

class BLECommandHandler:
    # UUIDs
    WEIGHTS_READ_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # For receiving FROM Arduino
    WEIGHTS_WRITE_CHAR_UUID = "19B10005-E8F2-537E-4F6C-D104768A1214"  # For sending TO Arduino
    CONTROL_CHAR_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"
    LABEL_CHAR_UUID = "19B10003-E8F2-537E-4F6C-D104768A1214"
    PREDICTION_CHAR_UUID = "19B10004-E8F2-537E-4F6C-D104768A1214"

    # Commands
    GET_WEIGHTS = 1
    SET_WEIGHTS = 2
    START_TRAINING = 3
    START_CLASSIFICATION = 4
    START_INFERENCE_BENCHMARK = 5
    START_TRAINING_BENCHMARK = 6

    # Network Architecture
    LAYERS = [11, 60, 3]

    def __init__(self, device_address):
        self.device_address = device_address
        self.predictions = []
        self.received_weights = []
        self.client = None
        self.total_weights = sum(self.LAYERS[i] * self.LAYERS[i + 1]
                               for i in range(len(self.LAYERS) - 1))


    async def run_inference_benchmark(self):
        """Run inference timing benchmark"""
        try:
            print("Starting inference timing benchmark...")
            await self.client.write_gatt_char(
                self.CONTROL_CHAR_UUID,
                bytes([self.START_INFERENCE_BENCHMARK])
            )

            # Wait for benchmark to complete
            await asyncio.sleep(15)  # Adjust based on your NUM_ITERATIONS
            print("Inference benchmark complete")
            return True
        except Exception as e:
            print(f"Benchmark error: {str(e)}")
            return False

    async def run_training_benchmark(self, label):
        """Run training timing benchmark"""
        if not (0 <= label <= 2):
            print("Invalid label. Must be 0, 1, or 2")
            return False

        try:
            print(f"Starting training benchmark with label {label}...")

            # Write label first
            await self.client.write_gatt_char(
                self.LABEL_CHAR_UUID,
                bytes([label])
            )

            # Then start benchmark
            await self.client.write_gatt_char(
                self.CONTROL_CHAR_UUID,
                bytes([self.START_TRAINING_BENCHMARK])
            )

            # Wait for benchmark to complete
            await asyncio.sleep(15)  # Adjust based on your NUM_ITERATIONS
            print("Training benchmark complete")
            return True
        except Exception as e:
            print(f"Benchmark error: {str(e)}")
            return False

    def weights_callback(self, sender, data):
        """Handle incoming weights data"""
        chunk_weights = []
        num_floats = len(data) // 4
        for i in range(num_floats):
            value = struct.unpack('f', data[i * 4:(i + 1) * 4])[0]
            chunk_weights.append(value)

        self.received_weights.extend(chunk_weights)
        print(f"Received chunk of {num_floats} weights. "
              f"Total received: {len(self.received_weights)}/{self.total_weights}")

    def prediction_callback(self, sender, data):
        """Handle incoming prediction probabilities"""
        predictions = []
        num_floats = len(data) // 4
        for i in range(num_floats):
            value = struct.unpack('f', data[i * 4:(i + 1) * 4])[0]
            predictions.append(value)
        self.predictions = predictions
        print("\nPrediction probabilities:")
        print(f"No theft: {predictions[0] * 100:.1f}%")
        print(f"Carrying away: {predictions[1] * 100:.1f}%")
        print(f"Lock breach: {predictions[2] * 100:.1f}%")

    async def connect(self):
        """Connect to the device and set up notifications"""
        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            print(f"Connected: {self.client.is_connected}")

            # Set up notifications only for reading weights and predictions
            await self.client.start_notify(self.WEIGHTS_READ_CHAR_UUID, self.weights_callback)
            await self.client.start_notify(self.PREDICTION_CHAR_UUID, self.prediction_callback)

            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from the device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected")

    async def get_weights(self):
        """Request and receive current network weights"""
        try:
            self.received_weights = []
            print("Requesting weights...")
            await self.client.write_gatt_char(self.CONTROL_CHAR_UUID,
                                              bytes([self.GET_WEIGHTS]))

            # Wait for all weights
            timeout = 240  # seconds
            start_time = asyncio.get_event_loop().time()

            while len(self.received_weights) < self.total_weights:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print("Timeout waiting for weights")
                    return None
                await asyncio.sleep(0.1)

            return np.array(self.received_weights)

        except Exception as e:
            print(f"Error getting weights: {str(e)}")
            return None

    async def send_weights(self, weights):
        """Send weights to the device with optimized transfer"""
        try:
            if len(weights) != self.total_weights:
                print(f"Error: Expected {self.total_weights} weights, got {len(weights)}")
                return False

            print("Initiating weight update...")

            # Send SET_WEIGHTS command
            await self.client.write_gatt_char(
                self.CONTROL_CHAR_UUID,
                bytearray([self.SET_WEIGHTS]),
                response=True
            )
            await asyncio.sleep(0.05)  # 5ms delay before starting transfer

            # Prepare all chunks first
            chunk_size = 52
            chunks = []
            for i in range(0, len(weights), chunk_size):
                chunk = weights[i:i + chunk_size]
                chunk_bytes = bytearray()
                for weight in chunk:
                    chunk_bytes.extend(struct.pack('f', float(weight)))
                chunks.append(chunk_bytes)

            total_sent = 0
            start_time = time.time()

            # Send chunks with minimal delay
            for i, chunk_bytes in enumerate(chunks):
                await self.client.write_gatt_char(
                    self.WEIGHTS_WRITE_CHAR_UUID,
                    chunk_bytes,
                    response=True
                )
                total_sent += len(chunk_bytes) // 4  # divide by 4 because each float is 4 bytes
                print(f"Sent chunk {i + 1}/{len(chunks)} ({total_sent}/{len(weights)} weights)")
                #await asyncio.sleep(0.005)  # Delay is unnecessary because write operations to client require confirmation by client before next package is sent

            end_time = time.time()
            duration = end_time - start_time
            print(f"Weight update complete in {duration:.3f} seconds")
            print(f"Effective throughput: {(len(weights) * 4 * 8) / (duration * 1000):.2f} kbps")

            return True

        except Exception as e:
            print(f"Error sending weights: {str(e)}")
            return False

    async def start_classification(self):
        """Start classification mode"""
        try:
            self.predictions = []

            print("Starting classification...")
            await self.client.write_gatt_char(self.CONTROL_CHAR_UUID,
                                              bytes([self.START_CLASSIFICATION]))

            # Wait for results
            timeout = 10  # seconds
            start_time = asyncio.get_event_loop().time()

            while not self.predictions:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print("Timeout waiting for classification results")
                    return None
                await asyncio.sleep(0.1)

            return self.predictions

        except Exception as e:
            print(f"Classification error: {str(e)}")
            return None

    async def start_training(self, label):
        """Start training mode with given label"""
        if not (0 <= label <= 2):
            print("Invalid label. Must be 0, 1, or 2")
            return False

        try:
            print(f"Starting training with label {label}...")

            # Write label first
            await self.client.write_gatt_char(self.LABEL_CHAR_UUID,
                                              bytes([label]))

            # Then start training
            await self.client.write_gatt_char(self.CONTROL_CHAR_UUID,
                                              bytes([self.START_TRAINING]))

            # Wait a reasonable time for training to complete
            await asyncio.sleep(2)

            print("Training complete")
            return True

        except Exception as e:
            print(f"Training error: {str(e)}")
            return False

    def print_weights_matrix(self, weights):
        """Print weights organized by network layers"""
        if len(weights) != self.total_weights:
            print("Incomplete weight data")
            return

        idx = 0
        for layer_num in range(len(self.LAYERS) - 1):
            inputs = self.LAYERS[layer_num]
            outputs = self.LAYERS[layer_num + 1]
            print(f"\nLayer {layer_num + 1} Weights ({inputs}x{outputs}):")
            layer_weights = weights[idx:idx + inputs * outputs].reshape((outputs, inputs))
            print(np.array2string(layer_weights, precision=6, suppress_small=True))
            idx += inputs * outputs


async def main():
    handler = BLECommandHandler("ac:3f:75:88:8f:12")  # Your device address

    if not await handler.connect():
        return

    try:
        # Create timing handler
        timing_handler = BLETimingHandler(handler)

        while True:
            command = input("\nEnter command (c=classify, t=train, g=get weights, "
                            "s=set weights, bi=inference benchmark, bt=training benchmark, "
                            "mg=measure get_weights, ms=measure send_weights, mb=measure both, "
                            "q=quit): ")

            if command.lower() == 'q':
                break
            elif command.lower() == 'bi':
                await handler.run_inference_benchmark()
            elif command.lower() == 'bt':
                label = int(input("Enter label for training benchmark (0-2): "))
                await handler.run_training_benchmark(label)
            elif command.lower() == 'c':
                await handler.start_classification()
            elif command.lower() == 't':
                label = int(input("Enter label (0=no theft, 1=carrying away, 2=lock breach): "))
                await handler.start_training(label)
            elif command.lower() == 'g':
                weights = await handler.get_weights()
                if weights is not None:
                    handler.print_weights_matrix(weights)
            elif command.lower() == 's':
                # Example: create random weights for testing
                new_weights = np.random.normal(0, 0.5, handler.total_weights).astype(np.float32)
                await handler.send_weights(new_weights)
            elif command.lower() == 'mg':
                await timing_handler.measure_get_weights()
            elif command.lower() == 'ms':
                await timing_handler.measure_send_weights()
            elif command.lower() == 'mb':
                await timing_handler.measure_get_weights()
                await timing_handler.measure_send_weights()

            await asyncio.sleep(0.1)

    finally:
        await handler.disconnect()


if __name__ == "__main__":
    asyncio.run(main())