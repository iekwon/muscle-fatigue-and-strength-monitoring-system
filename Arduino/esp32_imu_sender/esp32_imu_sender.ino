/*
 * ============================================================================
 *  ESP32 + MPU6500 -> UDP Streamer  (ROM tracking system)
 * ============================================================================
 *  Flash this SAME sketch to BOTH ESP32 boards. The only thing that changes
 *  between the two boards is CLIENT_ID below:
 *
 *      CLIENT_ID = 1  ->  PROXIMAL segment  (e.g. upper arm, thigh)
 *      CLIENT_ID = 2  ->  DISTAL segment    (e.g. forearm, shank)
 *
 *  Hardware: MPU6500 (accel + gyro only -- no magnetometer, no onboard
 *  fusion). This sketch talks to it with raw I2C register reads (no
 *  external MPU library needed) and runs a Madgwick AHRS filter
 *  (accel+gyro variant) on the ESP32 itself to produce a unit quaternion.
 *
 *  IMPORTANT — keep the board perfectly still for ~2 seconds after power-on:
 *  that window is used to measure gyro bias. Movement during that window
 *  will bake a bad bias into every reading afterwards.
 *
 *  Wiring:
 *      MPU6500 VCC -> 3.3V
 *      MPU6500 GND -> GND
 *      MPU6500 SDA -> GPIO 21
 *      MPU6500 SCL -> GPIO 22
 *      MPU6500 AD0 -> GND  (I2C address 0x68; tie to 3.3V for 0x69)
 *
 *  Packet format sent to the Pi (unchanged from the BNO055 version, so
 *  pi_rom_server.py needs NO changes):
 *      "<id>,<seq>,<w>,<x>,<y>,<z>\n"
 * ============================================================================
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>

// ============================== USER CONFIG ================================
#define CLIENT_ID              1                 // <-- SET TO 1 or 2 PER BOARD
const char*    WIFI_SSID     = "SSN-LG_2.4";
const char*    WIFI_PASSWORD = "Ssn1!Som2@Sase3#";
const char* PI_IP_ADDR = "192.168.137.50";    // Raspberry Pi's IP (hostname -I)
const uint16_t PI_UDP_PORT   = 5005;
const uint32_t SEND_INTERVAL_MS = 20;             // 20 ms -> 50 Hz UDP send rate
const uint8_t  MPU_ADDR         = 0x68;           // 0x69 if AD0 tied high
const float    MADGWICK_BETA    = 0.10f;          // higher = trusts accel more (less drift, more jitter)
// =============================================================================

// ---- MPU6500 registers ----
#define REG_PWR_MGMT_1   0x6B
#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_WHO_AM_I     0x75

// Full-scale selections used below: gyro +/-1000 dps, accel +/-4 g
const float GYRO_SCALE  = 32.8f;    // LSB per (deg/s) at +/-1000 dps
const float ACCEL_SCALE = 8192.0f;  // LSB per g at +/-4 g

WiFiUDP udp;
uint32_t lastSendMs = 0;
uint32_t seqCounter = 0;

// Gyro bias (raw LSB), measured at startup while stationary
float gyroBiasX = 0, gyroBiasY = 0, gyroBiasZ = 0;

// Orientation quaternion (w, x, y, z), updated by the Madgwick filter
float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;

// ----------------------------------------------------------------------------
// Low-level I2C helpers
// ----------------------------------------------------------------------------
void mpuWriteReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t mpuReadReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 1, true);
  return Wire.available() ? Wire.read() : 0;
}

// Reads accel (3) + temp (1, discarded) + gyro (3) = 14 bytes in one burst
void mpuReadRaw(int16_t &ax, int16_t &ay, int16_t &az,
                int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom((int)MPU_ADDR, 14, true);

  ax = (Wire.read() << 8) | Wire.read();
  ay = (Wire.read() << 8) | Wire.read();
  az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();          // temperature, unused
  gx = (Wire.read() << 8) | Wire.read();
  gy = (Wire.read() << 8) | Wire.read();
  gz = (Wire.read() << 8) | Wire.read();
}

void mpuInit() {
  mpuWriteReg(REG_PWR_MGMT_1, 0x01);    // wake up, use gyro-X PLL as clock source
  delay(50);
  mpuWriteReg(REG_SMPLRT_DIV, 0x00);    // sample rate = internal rate (no extra division)
  mpuWriteReg(REG_CONFIG, 0x03);        // DLPF ~44 Hz bandwidth (smooths noise a bit)
  mpuWriteReg(REG_GYRO_CONFIG, 0x10);   // +/-1000 dps
  mpuWriteReg(REG_ACCEL_CONFIG, 0x08);  // +/-4 g
  delay(50);
} 

// ----------------------------------------------------------------------------
// Startup gyro bias calibration -- BOARD MUST BE STILL during this
// ----------------------------------------------------------------------------
void calibrateGyroBias() {
  const int N = 1000;
  double sx = 0, sy = 0, sz = 0;
  int16_t ax, ay, az, gx, gy, gz;

  Serial.printf("[ID %d] Calibrating gyro bias -- KEEP STILL...\n", CLIENT_ID);
  for (int i = 0; i < N; i++) {
    mpuReadRaw(ax, ay, az, gx, gy, gz);
    sx += gx; sy += gy; sz += gz;
    delay(2);
  }
  gyroBiasX = sx / N;
  gyroBiasY = sy / N;
  gyroBiasZ = sz / N;
  Serial.printf("[ID %d] Gyro bias: %.2f, %.2f, %.2f (raw LSB)\n",
                CLIENT_ID, gyroBiasX, gyroBiasY, gyroBiasZ);
}

// Initialize the quaternion from a single accelerometer reading so the
// filter starts near the correct tilt instead of drifting up from "flat".
// Yaw can't be observed from accel alone (no magnetometer), so it starts at 0.
void initQuaternionFromAccel() {
  int16_t ax, ay, az, gx, gy, gz;
  mpuReadRaw(ax, ay, az, gx, gy, gz);
  float axg = ax / ACCEL_SCALE, ayg = ay / ACCEL_SCALE, azg = az / ACCEL_SCALE;

  float roll  = atan2f(ayg, azg);
  float pitch = atan2f(-axg, sqrtf(ayg * ayg + azg * azg));

  float cr = cosf(roll * 0.5f),  sr = sinf(roll * 0.5f);
  float cp = cosf(pitch * 0.5f), sp = sinf(pitch * 0.5f);
  // yaw = 0 -> cy = 1, sy = 0
  q0 = cr * cp;
  q1 = sr * cp;
  q2 = cr * sp;
  q3 = -sr * sp;
}

// ----------------------------------------------------------------------------
// Madgwick AHRS update, accel+gyro (IMU) variant -- no magnetometer input.
// gx,gy,gz in rad/s ; ax,ay,az in any consistent units (normalized inside).
// Updates the global q0,q1,q2,q3.
// ----------------------------------------------------------------------------
void madgwickUpdateIMU(float gx, float gy, float gz,
                        float ax, float ay, float az, float dt) {
  float qDot0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
  float qDot1 = 0.5f * ( q0 * gx + q2 * gz - q3 * gy);
  float qDot2 = 0.5f * ( q0 * gy - q1 * gz + q3 * gx);
  float qDot3 = 0.5f * ( q0 * gz + q1 * gy - q2 * gx);

  // Skip the accel correction step if the reading is degenerate (all zero)
  if (!(ax == 0.0f && ay == 0.0f && az == 0.0f)) {
    float norm = sqrtf(ax * ax + ay * ay + az * az);
    ax /= norm; ay /= norm; az /= norm;

    float _2q0 = 2.0f * q0, _2q1 = 2.0f * q1, _2q2 = 2.0f * q2, _2q3 = 2.0f * q3;
    float _4q0 = 4.0f * q0, _4q1 = 4.0f * q1, _4q2 = 4.0f * q2;
    float _8q1 = 8.0f * q1, _8q2 = 8.0f * q2;
    float q0q0 = q0 * q0, q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;

    float s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
    float s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay
               - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
    float s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay
               - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
    float s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;

    float sNorm = sqrtf(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
    if (sNorm > 1e-9f) {
      s0 /= sNorm; s1 /= sNorm; s2 /= sNorm; s3 /= sNorm;
      qDot0 -= MADGWICK_BETA * s0;
      qDot1 -= MADGWICK_BETA * s1;
      qDot2 -= MADGWICK_BETA * s2;
      qDot3 -= MADGWICK_BETA * s3;
    }
  }

  q0 += qDot0 * dt;
  q1 += qDot1 * dt;
  q2 += qDot2 * dt;
  q3 += qDot3 * dt;

  float qNorm = sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 /= qNorm; q1 /= qNorm; q2 /= qNorm; q3 /= qNorm;
}

// ----------------------------------------------------------------------------
// WiFi
// ----------------------------------------------------------------------------
void connectWiFi() {
  Serial.printf("[ID %d] Connecting to WiFi \"%s\"", CLIENT_ID, WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (millis() - start > 20000) {
      Serial.println("\n  -> still not connected, retrying...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      start = millis();
    }
  }
  Serial.printf("\n[ID %d] WiFi connected. IP: %s\n",
                CLIENT_ID, WiFi.localIP().toString().c_str());
}

// ----------------------------------------------------------------------------
// Setup / loop
// ----------------------------------------------------------------------------
uint32_t lastFilterMicros = 0;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.printf("\n=== ROM Sensor Node — CLIENT_ID %d (MPU6500) ===\n", CLIENT_ID);

  Wire.begin();          // default SDA=21, SCL=22 on most ESP32 dev boards
  Wire.setClock(400000); // 400kHz fast mode

  uint8_t whoAmI = mpuReadReg(REG_WHO_AM_I);
  Serial.printf("[ID %d] WHO_AM_I = 0x%02X (expect ~0x70 for MPU6500)\n",
                CLIENT_ID, whoAmI);

  mpuInit();
  calibrateGyroBias();       // BOARD MUST BE STILL for this
  initQuaternionFromAccel();

  connectWiFi();
  udp.begin(PI_UDP_PORT);

  lastFilterMicros = micros();
  Serial.printf("[ID %d] Streaming quaternions to %s:%d @ %lu Hz\n",
                CLIENT_ID, PI_IP_ADDR, PI_UDP_PORT,
                (unsigned long)(1000 / SEND_INTERVAL_MS));
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // --- Read sensor & update filter as fast as the loop can run ---
  int16_t rax, ray, raz, rgx, rgy, rgz;
  mpuReadRaw(rax, ray, raz, rgx, rgy, rgz);

  uint32_t nowMicros = micros();
  float dt = (nowMicros - lastFilterMicros) / 1000000.0f;
  lastFilterMicros = nowMicros;
  if (dt <= 0 || dt > 0.2f) dt = 0.005f;  // guard against startup/overflow glitches

  float ax = rax / ACCEL_SCALE;
  float ay = ray / ACCEL_SCALE;
  float az = raz / ACCEL_SCALE;
  float gx = (rgx - gyroBiasX) / GYRO_SCALE * (PI / 180.0f);  // -> rad/s
  float gy = (rgy - gyroBiasY) / GYRO_SCALE * (PI / 180.0f);
  float gz = (rgz - gyroBiasZ) / GYRO_SCALE * (PI / 180.0f);

  madgwickUpdateIMU(gx, gy, gz, ax, ay, az, dt);

  // --- Send at the configured UDP rate ---
  uint32_t nowMs = millis();
  if (nowMs - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = nowMs;
    char buf[96];
    int len = snprintf(buf, sizeof(buf), "%d,%lu,%.6f,%.6f,%.6f,%.6f\n",
                        CLIENT_ID, (unsigned long)seqCounter++,
                        q0, q1, q2, q3);
    udp.beginPacket(PI_IP_ADDR, PI_UDP_PORT);
    udp.write((const uint8_t*)buf, len);
    udp.endPacket();
  }
}
