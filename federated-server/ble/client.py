"""
BLE client for communicating with Arduino devices.
"""

import asyncio
from bleak import BleakClient
from ble.protocol import BLEProtocol

class BLEClient:
    def __init__(self, device_address):
        self.device_address = device_address
        self.client = None
        self.connected = False
        self.callbacks = {}
    
    async def connect(self):
        """Connect to the BLE device."""
        try:
            self.client = BleakClient(self.device_address)
            await self.client.connect()
            self.connected = self.client.is_connected
            print(f"Connected: {self.connected}")
            return self.connected
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from the BLE device."""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            self.connected = False
            print("Disconnected")
            
    async def start_notify(self, char_uuid, callback):
        """Start notification for a characteristic."""
        if not self.connected:
            print("Not connected to device")
            return False
            
        try:
            await self.client.start_notify(char_uuid, callback)
            return True
        except Exception as e:
            print(f"Notification error: {str(e)}")
            return False
            
    async def write_char(self, char_uuid, data, response=True):
        """Write data to a characteristic."""
        if not self.connected:
            print("Not connected to device")
            return False
            
        try:
            await self.client.write_gatt_char(char_uuid, data, response=response)
            return True
        except Exception as e:
            print(f"Write error: {str(e)}")
            return False