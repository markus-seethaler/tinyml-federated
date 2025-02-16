#include <Arduino_LSM9DS1.h>

#define SAMPLES 256
#define SAMPLING_FREQ 100
#define SAMPLING_PERIOD_MS (1000/SAMPLING_FREQ)

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        while (1);
    }
}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();
        if (command == 'r') {  // Start recording when 'r' is received
            recordData();
        }
    }
}

void recordData() {
    float x, y, z;
    unsigned long millisOld;
    
    Serial.println("START"); // Marker for Python script
    
    // Data Collection
    for(int i = 0; i < SAMPLES; i++) {
        millisOld = millis();
        
        if (IMU.accelerationAvailable()) {
            IMU.readAcceleration(x, y, z);
            // Send all three axes separated by commas
            Serial.print(x * 9.81f, 6);  // Convert to m/s²
            Serial.print(",");
            Serial.print(y * 9.81f, 6);
            Serial.print(",");
            Serial.println(z * 9.81f, 6);
        } else {
            Serial.println("0,0,0");
        }
        
        // Maintain sampling rate
        while((millis() - millisOld) < SAMPLING_PERIOD_MS);
    }
    
    Serial.println("END"); // Marker for Python script
}