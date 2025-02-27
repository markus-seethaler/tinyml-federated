"""
BLE protocol constants and utilities for the federated learning system.
"""

class BLEProtocol:
    # UUIDs
    SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
    WEIGHTS_READ_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"  # From Arduino
    WEIGHTS_WRITE_CHAR_UUID = "19B10005-E8F2-537E-4F6C-D104768A1214"  # To Arduino
    CONTROL_CHAR_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"
    LABEL_CHAR_UUID = "19B10003-E8F2-537E-4F6C-D104768A1214"
    PREDICTION_CHAR_UUID = "19B10004-E8F2-537E-4F6C-D104768A1214"

    # Commands
    class Command:
        NONE = 0
        GET_WEIGHTS = 1
        SET_WEIGHTS = 2
        START_TRAINING = 3
        START_CLASSIFICATION = 4
        START_INFERENCE_BENCHMARK = 5
        START_TRAINING_BENCHMARK = 6

    # Transfer parameters
    CHUNK_SIZE_RECEIVE = 52  # Max floats per chunk when receiving
    CHUNK_SIZE_SEND = 32     # Max floats per chunk when sending