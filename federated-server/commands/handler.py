"""
Command handlers for interacting with Arduino devices.
"""

import asyncio
import struct
import time
import numpy as np
from ble.client import BLEClient
from ble.protocol import BLEProtocol
from models.nn_config import NNConfig

class CommandHandler:
    def __init__(self, ble_client):
        self.ble_client = ble_client
        self.predictions = []
        self.received_weights = []
        self.total_weights = NNConfig.calculate_total_weights()
        
    async def setup(self):
        """Set up notifications after connection."""
        if not self.ble_client.connected:
            return False
            
        # Set up notifications for weights and predictions
        success1 = await self.ble_client.start_notify(
            BLEProtocol.WEIGHTS_READ_CHAR_UUID, 
            self._weights_callback
        )
        
        success2 = await self.ble_client.start_notify(
            BLEProtocol.PREDICTION_CHAR_UUID, 
            self._prediction_callback
        )
        
        return success1 and success2
        
    def _weights_callback(self, sender, data):
        """Handle incoming weights data."""
        chunk_weights = []
        num_floats = len(data) // 4
        for i in range(num_floats):
            value = struct.unpack('f', data[i * 4:(i + 1) * 4])[0]
            chunk_weights.append(value)

        self.received_weights.extend(chunk_weights)
        print(f"Received chunk of {num_floats} weights. "
              f"Total received: {len(self.received_weights)}/{self.total_weights}")

    def _prediction_callback(self, sender, data):
        """Handle incoming prediction probabilities."""
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
    
    async def get_weights(self):
        """Request and receive current network weights from the device."""
        try:
            self.received_weights = []
            print("Requesting weights...")
            
            success = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytes([BLEProtocol.Command.GET_WEIGHTS])
            )
            
            if not success:
                print("Failed to send GET_WEIGHTS command")
                return None

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
        """Send weights to the device with optimized transfer."""
        try:
            if len(weights) != self.total_weights:
                print(f"Error: Expected {self.total_weights} weights, got {len(weights)}")
                return False

            print("Initiating weight update...")

            # Send SET_WEIGHTS command
            success = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytearray([BLEProtocol.Command.SET_WEIGHTS]),
                response=True
            )
            
            if not success:
                print("Failed to send SET_WEIGHTS command")
                return False
                
            await asyncio.sleep(0.05)  # 5ms delay before starting transfer

            # Prepare all chunks first
            chunk_size = BLEProtocol.CHUNK_SIZE_RECEIVE
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
                success = await self.ble_client.write_char(
                    BLEProtocol.WEIGHTS_WRITE_CHAR_UUID,
                    chunk_bytes,
                    response=True
                )
                
                if not success:
                    print(f"Failed to send chunk {i+1}")
                    return False
                    
                total_sent += len(chunk_bytes) // 4  # divide by 4 because each float is 4 bytes
                print(f"Sent chunk {i + 1}/{len(chunks)} ({total_sent}/{len(weights)} weights)")
                # No explicit delay needed as write operations require confirmation

            end_time = time.time()
            duration = end_time - start_time
            print(f"Weight update complete in {duration:.3f} seconds")
            print(f"Effective throughput: {(len(weights) * 4 * 8) / (duration * 1000):.2f} kbps")

            return True

        except Exception as e:
            print(f"Error sending weights: {str(e)}")
            return False

    async def start_classification(self):
        """Start classification mode on the device."""
        try:
            self.predictions = []

            print("Starting classification...")
            success = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytes([BLEProtocol.Command.START_CLASSIFICATION])
            )
            
            if not success:
                print("Failed to send START_CLASSIFICATION command")
                return None

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
        """Start training mode with given label on the device."""
        if not (0 <= label <= 2):
            print("Invalid label. Must be 0, 1, or 2")
            return False

        try:
            print(f"Starting training with label {label}...")

            # Write label first
            success1 = await self.ble_client.write_char(
                BLEProtocol.LABEL_CHAR_UUID,
                bytes([label])
            )
            
            if not success1:
                print("Failed to send training label")
                return False

            # Then start training
            success2 = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytes([BLEProtocol.Command.START_TRAINING])
            )
            
            if not success2:
                print("Failed to send START_TRAINING command")
                return False

            # Wait a reasonable time for training to complete
            await asyncio.sleep(2)

            print("Training complete")
            return True

        except Exception as e:
            print(f"Training error: {str(e)}")
            return False

    async def run_inference_benchmark(self):
        """Run inference timing benchmark on the device."""
        try:
            print("Starting inference timing benchmark...")
            success = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytes([BLEProtocol.Command.START_INFERENCE_BENCHMARK])
            )
            
            if not success:
                print("Failed to send START_INFERENCE_BENCHMARK command")
                return False

            # Wait for benchmark to complete
            await asyncio.sleep(15)  # Adjust based on your NUM_ITERATIONS
            print("Inference benchmark complete")
            return True
        except Exception as e:
            print(f"Benchmark error: {str(e)}")
            return False

    async def run_training_benchmark(self, label):
        """Run training timing benchmark on the device."""
        if not (0 <= label <= 2):
            print("Invalid label. Must be 0, 1, or 2")
            return False

        try:
            print(f"Starting training benchmark with label {label}...")

            # Write label first
            success1 = await self.ble_client.write_char(
                BLEProtocol.LABEL_CHAR_UUID,
                bytes([label])
            )
            
            if not success1:
                print("Failed to send training label")
                return False

            # Then start benchmark
            success2 = await self.ble_client.write_char(
                BLEProtocol.CONTROL_CHAR_UUID,
                bytes([BLEProtocol.Command.START_TRAINING_BENCHMARK])
            )
            
            if not success2:
                print("Failed to send START_TRAINING_BENCHMARK command")
                return False

            # Wait for benchmark to complete
            await asyncio.sleep(15)  # Adjust based on your NUM_ITERATIONS
            print("Training benchmark complete")
            return True
        except Exception as e:
            print(f"Benchmark error: {str(e)}")
            return False
            
    def print_weights_matrix(self, weights):
        """Print weights organized by network layers."""
        if len(weights) != self.total_weights:
            print("Incomplete weight data")
            return

        idx = 0
        layer_sizes = NNConfig.get_layer_sizes()
        
        for layer_num, (inputs, outputs) in enumerate(layer_sizes):
            print(f"\nLayer {layer_num + 1} Weights ({inputs}x{outputs}):")
            layer_weights = weights[idx:idx + inputs * outputs].reshape((outputs, inputs))
            print(np.array2string(layer_weights, precision=6, suppress_small=True))
            idx += inputs * outputs