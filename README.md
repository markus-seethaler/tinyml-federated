# TinyML-Federated: Federated Learning for Microcontrollers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of Federated Learning (FL) on Tiny Machine Learning (TinyML) devices, developed as part of a master's thesis research. The project focuses on developing scalable and robust architectural Design Principles (DP) for FL on microcontrollers, specifically using Arduino Nano 33 BLE Sense devices.

## Project Description

This project implements a smart bike lock system capable of detecting and classifying different theft attempts using accelerometer data and on-device machine learning. The system can detect three different states:

1. **No theft attempt** - Normal state with no suspicious activity
2. **Carrying away attempt** - Detection of someone trying to move the bike
3. **Lock breaking attempt** - Detection of someone trying to break or tamper with the lock

The implementation features:

- **Data Collection**: Custom implementation to collect accelerometer data from Arduino devices
- **Feature Extraction**: Signal processing to extract meaningful features from raw sensor data
- **Federated Learning**: Training a neural network across distributed devices without sharing raw data
- **On-device Inference**: Real-time theft detection directly on the microcontroller

## Repository Structure

| Component | Description |
|-----------|-------------|
| [data-collection-client](./data-collection-client) | Arduino implementation for collecting accelerometer data from sensors |
| [data-collection-server](./data-collection-server) | Python interface for receiving, labeling, and managing data collected from Arduino devices |
| [federated-client](./federated-client) | Client-side implementation of federated learning algorithms for Arduino Nano 33 BLE |
| [federated-server](./federated-server) | Server-side implementation for orchestrating federated learning, aggregating models, and communicating with clients |
| [federated-simulation](./federated-simulation) | Simulation environment for testing federated learning approaches with virtual devices |

## Getting Started

Start by exploring the specific component you're interested in:

1. To collect data for training, start with the [data-collection-client](./data-collection-client/README.md) and [data-collection-server](./data-collection-server/README.md)
2. To train a federated learning model, use the [federated-client](./federated-client/README.md) and [federated-server](./federated-server/README.md)
3. To experiment with different FL configurations without hardware, check out the [federated-simulation](./federated-simulation/README.md)

## Hardware Requirements

- Arduino Nano 33 BLE Sense
- USB cable for programming and data collection

## Software Dependencies

The project has different dependencies for each component:

- **Arduino**: Arduino IDE or PlatformIO
- **Python**: Python 3.7+ with various libraries (detailed in component-specific READMEs)
- **Bluetooth**: BLE communication libraries

See individual component READMEs for detailed dependency information.

## Research Context

This project is part of a master's thesis on Federated Learning for embedded systems. The goal is to develop and evaluate design principles for implementing FL on resource-constrained devices, with a practical application in smart bike lock security.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

