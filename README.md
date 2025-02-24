# TinyML-Federated: Federated Learning for Microcontrollers

## Overview
This repository contains the implementation of Federated Learning (FL) on Tiny Machine Learning (TinyML) devices, developed as part of a master's thesis research. The project focuses on developing scalable and robust architectural Design Principles (DP) for FL on microcontrollers.

## Project Description
In the subfolders you can find different implementations used in the master thesis to evaluate the feasability of these design principles. The overarching aim was to develop a smart bike lock which can detect distinguish between 3 different predictions:
1. No theft attempt
2. Carrying away attempt
3. Lock breaking attempt
The implementation features a custom implementation to collect data from the accelerometer of the Arduino and receive the data via Serial into a Python interface where the data can be labeled and saved to practical csv files which are managed by a csv metadata file.

Furthermore the main FL implementation features a modular client implementation for the Arduino 33 BLE Sense. These modules can be easily adapted or even decoupled and used in other implementations. The FL server runs on a Python interface and is connected via BLE.

To simulate a large number of devices without actually owning the hardware the simulation environment provides the possiblity to test out different FL configurations all in one environment.

To learn more about how to build and deploy the different solutions consult the READMEs in the specific folders.

## Repository Structure
- **data-collection-client**: Implementation for data collection on client devices
- **data-collection-server**: Server-side code for managing data collection
- **federated-client**: Client-side implementation of federated learning algorithms
- **federated-server**: Server-side implementation for federated learning orchestration
- **federated-simulation**: Simulation environment for testing federated learning approaches


