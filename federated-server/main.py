"""
Main program for the federated learning server.
"""

import asyncio
import argparse
import numpy as np
import sys
import signal
from ble.client import BLEClient
from ble.protocol import BLEProtocol
from commands.handler import CommandHandler
from benchmark.timing import BLETimingHandler
from models.nn_config import NNConfig

class GracefulExit:
    """Handle graceful shutdown on keyboard interrupt or termination signal."""
    def __init__(self):
        self.exit_now = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        
    def _exit_gracefully(self, signum, frame):
        print("\nReceived exit signal. Cleaning up...")
        self.exit_now = True

async def display_help():
    """Display available commands."""
    print("\nAvailable commands:")
    print("  c  - Start classification")
    print("  t  - Start training (with label)")
    print("  g  - Get weights from device")
    print("  s  - Set random weights on device")
    print("  bi - Run inference benchmark")
    print("  bt - Run training benchmark")
    print("  mg - Measure GET_WEIGHTS performance")
    print("  ms - Measure SET_WEIGHTS performance")
    print("  mb - Measure both GET and SET performance")
    print("  h  - Display this help message")
    print("  q  - Quit the program")

async def main_menu(command_handler, timing_handler, exit_handler):
    """Interactive menu for controlling the federated learning process."""
    await display_help()
    
    while not exit_handler.exit_now:
        try:
            command = input("\nEnter command ('h' for help): ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'h':
                await display_help()
            elif command == 'c':
                await command_handler.start_classification()
            elif command == 't':
                try:
                    label = int(input("Enter label (0=no theft, 1=carrying away, 2=lock breach): "))
                    if not (0 <= label <= 2):
                        print("Invalid label. Must be 0, 1, or 2")
                        continue
                    await command_handler.start_training(label)
                except ValueError:
                    print("Please enter a valid number (0-2)")
            elif command == 'g':
                weights = await command_handler.get_weights()
                if weights is not None:
                    command_handler.print_weights_matrix(weights)
            elif command == 's':
                print("Generating random weights...")
                new_weights = np.random.normal(0, 0.5, command_handler.total_weights).astype(np.float32)
                await command_handler.send_weights(new_weights)
            elif command == 'bi':
                await command_handler.run_inference_benchmark()
            elif command == 'bt':
                try:
                    label = int(input("Enter label for training benchmark (0-2): "))
                    if not (0 <= label <= 2):
                        print("Invalid label. Must be 0, 1, or 2")
                        continue
                    await command_handler.run_training_benchmark(label)
                except ValueError:
                    print("Please enter a valid number (0-2)")
            elif command == 'mg':
                await timing_handler.measure_get_weights()
            elif command == 'ms':
                await timing_handler.measure_send_weights()
            elif command == 'mb':
                await timing_handler.measure_both()
            else:
                print(f"Unknown command: '{command}'")
                await display_help()
                
        except Exception as e:
            print(f"Error executing command: {str(e)}")

async def main():
    """Main entry point for the federated learning server."""
    parser = argparse.ArgumentParser(description='Federated Learning Server for Arduino-based Bike Lock')
    parser.add_argument('--device', '-d', type=str, 
                        help='BLE device address (e.g., "xx:xx:xx:xx:xx:xx")')
    args = parser.parse_args()
    
    device_address = args.device
    
    # If no device address provided, prompt the user
    if not device_address:
        device_address = input("Enter BLE device address (e.g., xx:xx:xx:xx:xx:xx): ")
    
    print(f"Connecting to device: {device_address}")
    
    ble_client = BLEClient(device_address)
    exit_handler = GracefulExit()
    
    try:
        print("Establishing connection...")
        if not await ble_client.connect():
            print("Failed to connect to device. Exiting.")
            return 1
            
        command_handler = CommandHandler(ble_client)
        print("Setting up notification handlers...")
        if not await command_handler.setup():
            print("Failed to set up notification handlers. Exiting.")
            return 1
            
        timing_handler = BLETimingHandler(command_handler)
        
        print(f"\nConnected to bike lock device at {device_address}")
        print("Neural network configuration:")
        print(f"  Layers: {NNConfig.LAYERS}")
        print(f"  Total weights: {command_handler.total_weights}")
        
        await main_menu(command_handler, timing_handler, exit_handler)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        print("Disconnecting from device...")
        await ble_client.disconnect()
    
    print("Exiting.")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)