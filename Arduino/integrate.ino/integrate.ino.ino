HardwareSerial EMGSerial(2);

void setup() {
  Serial.begin(115200);

  // RX = GPIO16, TX unused
  EMGSerial.begin(1000000, SERIAL_8N1, 16, -1);

  Serial.println("ESP32 EMG UART TEST");
}

void loop() {
  static uint32_t count = 0;

  while (EMGSerial.available()) {
    char c = EMGSerial.read();

    if (c == '\n') {
      count++;

      if (count % 100 == 0) {
        Serial.print("Received lines: ");
        Serial.println(count);
      }
    }
  }
}