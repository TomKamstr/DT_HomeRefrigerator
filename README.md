# Home Refrigerator Digital Twin
This repository contains the source code for creating a digital twin of a home refrigerator. The primary objective of this project is to enhance energy efficiency and promote sustainability through smart use of household appliances. Our codebase includes Python for the primary application logic, along with some Arduino code for interfacing with physical devices.

## Overview
The digital twin of a home refrigerator can provide valuable insights on the usage pattern, potential inefficiencies, and possible enhancements to the existing model. With the power of IoT and machine learning, we aim to model the operation of a home refrigerator and provide insights that can help consumers save on energy bills and reduce their carbon footprint.

## Repository Structure
```
.
├── Arduino
│   ├── SensorDataCollection
│   │   └── M5Stack_Inside.ino
|   |   └── M5Stack_Outside.ino
│   └── README.md
├── Python
│   ├── src
│   │   ├── data_processing.py
│   │   ├── digital_twin_model.py
│   │   └── energy_savings.py
│   ├── tests
│   │   ├── test_data_processing.py
│   │   ├── test_digital_twin_model.py
│   │   └── test_energy_savings.py
│   └── README.md
├── LICENSE
└── README.md
```

## Installation

### Requirements
- Python 3.7+
- Arduino IDE
- Required Python Libraries
    - numpy
    - pandas
    - sklearn
    - matplotlib
    - Darts

You can install the Python dependencies using pip:
```
pip install -r requirements.txt
```
This command needs to be executed in the Python directory of the repository.

### Hardware
- Arduino UNO (or similar)
- Relevant Sensors (Temperature, Humidity, Door Status)

## Usage

### Python
Navigate to the Python directory:
```
cd Python
```
Run the digital twin model:
```
python src/digital_twin_model.py
```
Note: Make sure your sensor data is correctly located and referenced in the `digital_twin_model.py` script.

### Arduino
Open `SensorDataCollection.ino` in the Arduino IDE, compile, and upload to your Arduino device.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or need further clarification, feel free to reach out by opening an issue or directly contacting us.
