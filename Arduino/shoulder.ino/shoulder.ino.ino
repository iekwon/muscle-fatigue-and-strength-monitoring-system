/*
 * ============================================================================
 * ESP32 #1 + MPU6500 + Nano Binary EMG -> Raspberry Pi 4
 * ============================================================================
 *
 * Nano -> ESP32 UART:
 *   Binary protocol:
 *   A5 5A | TYPE | LENGTH | PAYLOAD | CRC16
 *
 * TYPE 0x01 = EMG
 * TYPE 0x02 = statistics
 *
 * ESP32 -> Pi UDP:
 *   IMU: ASCII
 *     <id>,<seq>,<w>,<x>,<y>,<z>\n
 *
 *   EMG: binary
 *     Header:
 *       uint8  'E'
 *       uint8  client_id
 *       uint32 batch_seq
 *       uint32 ESP32 micros
 *       uint16 sample_count
 *
 *     Samples:
 *       uint32 sample_number
 *       uint32 timestamp_us
 *       uint16 ADC
 *
 * IMPORTANT:
 * Keep the MPU6500 still for ~2 seconds after power-on.
 * ============================================================================
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>

// ============================== USER CONFIG ================================

#define CLIENT_ID              1

const char* WIFI_SSID         = "ESP_Hotspot";
const char* WIFI_PASSWORD     = "ESP32_Test123";
const char* PI_IP_ADDR        = "10.42.0.1";
const uint16_t PI_UDP_PORT    = 5005;

const uint32_t SEND_INTERVAL_MS = 20;   // 50 Hz

const uint8_t MPU_ADDR = 0x68;
const float MADGWICK_BETA = 0.10f;

// ============================== NANO UART ================================

const uint32_t EMG_UART_BAUD = 1000000;

const int EMG_UART_RX_PIN = 16;
const int EMG_UART_TX_PIN = 17;

const size_t EMG_UART_RX_BUFFER_BYTES = 4096;

// ============================== EMG RING ================================

const uint16_t EMG_BUFFER_CAPACITY = 2048;
const uint16_t EMG_BUFFER_MASK = EMG_BUFFER_CAPACITY - 1;

const uint16_t MAX_EMG_SAMPLES_PER_PACKET = 100;
const uint8_t MAX_EMG_PACKETS_PER_TICK = 4;

// ============================== NANO PROTOCOL =============================

// Frame:
// [0]  A5
// [1]  5A
// [2]  type
// [3]  payload length
// [4...] payload
// [last-1,last] CRC16

#define NANO_SYNC0       0xA5
#define NANO_SYNC1       0x5A

#define NANO_TYPE_EMG    0x01
#define NANO_TYPE_STATS  0x02

#define NANO_EMG_HEADER_LEN 10
#define NANO_EMG_SAMPLE_LEN 6

#define NANO_MAX_SAMPLES 32

#define NANO_STATS_LEN 26

#define NANO_MAX_FRAME 261

// ============================================================================
// MPU6500 REGISTERS
// ============================================================================

#define REG_PWR_MGMT_1   0x6B
#define REG_SMPLRT_DIV   0x19
#define REG_CONFIG       0x1A
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_XOUT_H 0x3B
#define REG_WHO_AM_I     0x75

const float GYRO_SCALE  = 32.8f;
const float ACCEL_SCALE = 8192.0f;

// ============================================================================
// GLOBALS
// ============================================================================

WiFiUDP udp;

uint32_t lastSendMs = 0;
uint32_t seqCounter = 0;

float gyroBiasX = 0;
float gyroBiasY = 0;
float gyroBiasZ = 0;

// Quaternion
float q0 = 1.0f;
float q1 = 0.0f;
float q2 = 0.0f;
float q3 = 0.0f;

// ============================================================================
// EMG SAMPLE
// ============================================================================

struct EmgSample {
  uint32_t sampleNumber;
  uint32_t timestampUs;
  uint16_t adcValue;
};

EmgSample emgBuffer[EMG_BUFFER_CAPACITY];

volatile uint16_t emgHead = 0;
volatile uint16_t emgTail = 0;

// Statistics
uint32_t emgOverflowCount = 0;
uint32_t emgParsedCount = 0;
uint32_t emgMalformedCount = 0;

uint32_t emgCrcErrorCount = 0;
uint32_t emgHeaderErrorCount = 0;
uint32_t emgPayloadErrorCount = 0;
uint32_t emgJunkByteCount = 0;

uint32_t emgFrameCount = 0;
uint32_t emgPacketLostCount = 0;
uint32_t emgSampleGapCount = 0;
uint32_t emgBackwardsCount = 0;

uint32_t emgBadAdcCount = 0;
uint32_t emgNanoDropFlagCount = 0;

uint32_t emgSentToPi = 0;

uint32_t emgBatchSeq = 0;

// Expected Nano sequence/sample number
bool haveNanoSeq = false;
bool haveNanoSample = false;

uint32_t expectedNanoSeq = 0;
uint32_t expectedNanoSample = 0;

// ============================================================================
// NANO STATS
// ============================================================================

struct NanoStats {

  bool fresh;

  uint32_t uptimeMs;
  uint32_t adcTotal;
  uint32_t txSamples;
  uint32_t txPackets;
  uint32_t dropped;

  uint16_t ringHwm;
  uint16_t dtMin;
  uint16_t dtMax;
};

NanoStats nanoStats = {};
NanoStats previousNanoStats = {};

bool havePreviousNanoStats = false;

// ============================================================================
// LITTLE-ENDIAN HELPERS
// ============================================================================

static inline uint16_t readU16LE(const uint8_t* p) {

  return
    (uint16_t)p[0] |
    ((uint16_t)p[1] << 8);
}

static inline uint32_t readU32LE(const uint8_t* p) {

  return
    (uint32_t)p[0] |
    ((uint32_t)p[1] << 8) |
    ((uint32_t)p[2] << 16) |
    ((uint32_t)p[3] << 24);
}

static inline void putU16LE(uint8_t* p, uint16_t v) {

  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
}

static inline void putU32LE(uint8_t* p, uint32_t v) {

  p[0] = (uint8_t)(v);
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

// ============================================================================
// CRC-16/MCRF4XX
// Same algorithm used by Nano's _crc_ccitt_update()
// ============================================================================

uint16_t crc16Mcrf4xx(const uint8_t* data, uint16_t length) {

  uint16_t crc = 0xFFFF;

  while (length--) {

    crc ^= *data++;

    for (uint8_t i = 0; i < 8; i++) {

      if (crc & 1) {
        crc = (crc >> 1) ^ 0x8408;
      } else {
        crc >>= 1;
      }
    }
  }

  return crc;
}

// ============================================================================
// PUSH EMG SAMPLE INTO ESP32 RING
// ============================================================================

bool pushEmgSample(
  uint32_t sampleNumber,
  uint32_t timestampUs,
  uint16_t adcValue
) {

  uint16_t nextHead =
    (emgHead + 1) & EMG_BUFFER_MASK;

  // Ring full
  if (nextHead == emgTail) {

    emgOverflowCount++;

    return false;
  }

  emgBuffer[emgHead].sampleNumber = sampleNumber;
  emgBuffer[emgHead].timestampUs = timestampUs;
  emgBuffer[emgHead].adcValue = adcValue;

  emgHead = nextHead;

  return true;
}

// ============================================================================
// HANDLE NANO EMG FRAME
// ============================================================================

void handleNanoEmgFrame(
  const uint8_t* payload,
  uint8_t payloadLength
) {

  if (payloadLength < NANO_EMG_HEADER_LEN) {

    emgPayloadErrorCount++;
    emgMalformedCount++;

    return;
  }

  uint32_t packetSeq =
    readU32LE(payload);

  uint32_t firstSample =
    readU32LE(payload + 4);

  uint8_t sampleCount =
    payload[8];

  uint8_t flags =
    payload[9];

  // Validate sample count
  if (
    sampleCount == 0 ||
    sampleCount > NANO_MAX_SAMPLES ||
    payloadLength !=
      NANO_EMG_HEADER_LEN +
      sampleCount * NANO_EMG_SAMPLE_LEN
  ) {

    emgPayloadErrorCount++;
    emgMalformedCount++;

    return;
  }

  // --------------------------------------------------------------------------
  // Check Nano packet sequence
  // --------------------------------------------------------------------------

  if (haveNanoSeq) {

    int32_t difference =
      (int32_t)(packetSeq - expectedNanoSeq);

    if (difference > 0) {

      emgPacketLostCount +=
        (uint32_t)difference;

    } else if (difference < 0) {

      emgBackwardsCount++;
    }
  }

  haveNanoSeq = true;

  expectedNanoSeq =
    packetSeq + 1;

  // --------------------------------------------------------------------------
  // Check sample number continuity
  // --------------------------------------------------------------------------

  if (haveNanoSample) {

    int32_t difference =
      (int32_t)(firstSample - expectedNanoSample);

    if (difference > 0) {

      emgSampleGapCount +=
        (uint32_t)difference;

    } else if (difference < 0) {

      emgBackwardsCount++;
    }
  }

  haveNanoSample = true;

  expectedNanoSample =
    firstSample + sampleCount;

  // --------------------------------------------------------------------------
  // Nano reports that it dropped samples
  // --------------------------------------------------------------------------

  if (flags & 0x01) {

    emgNanoDropFlagCount++;
  }

  emgFrameCount++;

  // --------------------------------------------------------------------------
  // Decode samples
  // --------------------------------------------------------------------------

  const uint8_t* p =
    payload + NANO_EMG_HEADER_LEN;

  for (uint8_t i = 0; i < sampleCount; i++) {

    uint32_t timestampUs =
      readU32LE(p);

    uint16_t adcValue =
      readU16LE(p + 4);

    uint32_t sampleNumber =
      firstSample + i;

    if (adcValue > 1023) {

      emgBadAdcCount++;
    }

    pushEmgSample(
      sampleNumber,
      timestampUs,
      adcValue
    );

    if (adcValue <= 1023) {

      emgParsedCount++;
    }

    p += NANO_EMG_SAMPLE_LEN;
  }
}

// ============================================================================
// HANDLE NANO STATS FRAME
// ============================================================================

void handleNanoStatsFrame(
  const uint8_t* payload,
  uint8_t payloadLength
) {

  if (payloadLength != NANO_STATS_LEN) {

    emgPayloadErrorCount++;
    emgMalformedCount++;

    return;
  }

  previousNanoStats = nanoStats;

  havePreviousNanoStats =
    (nanoStats.uptimeMs != 0);

  nanoStats.uptimeMs =
    readU32LE(payload);

  nanoStats.adcTotal =
    readU32LE(payload + 4);

  nanoStats.txSamples =
    readU32LE(payload + 8);

  nanoStats.txPackets =
    readU32LE(payload + 12);

  nanoStats.dropped =
    readU32LE(payload + 16);

  nanoStats.ringHwm =
    readU16LE(payload + 20);

  nanoStats.dtMin =
    readU16LE(payload + 22);

  nanoStats.dtMax =
    readU16LE(payload + 24);

  nanoStats.fresh = true;
}

// ============================================================================
// NANO BINARY FRAME BUFFER
// ============================================================================

uint8_t nanoFrame[NANO_MAX_FRAME];

uint16_t nanoFrameLength = 0;

// ============================================================================
// RESET / RESCAN FRAME
// ============================================================================

void nanoRescan() {

  if (nanoFrameLength == 0) {
    return;
  }

  // Remove first byte.
  emgJunkByteCount++;

  uint16_t oldLength =
    nanoFrameLength;

  nanoFrameLength = 0;

  // Feed remaining bytes again.
  for (uint16_t i = 1; i < oldLength; i++) {

    uint8_t b =
      nanoFrame[i];

    // Basic resynchronization.
    if (nanoFrameLength < NANO_MAX_FRAME) {

      nanoFrame[nanoFrameLength++] = b;
    }
  }
}

// ============================================================================
// FEED ONE BYTE INTO BINARY PARSER
// ============================================================================

void feedNanoByte(uint8_t b) {

  // --------------------------------------------------------------------------
  // Prevent buffer overflow
  // --------------------------------------------------------------------------

  if (nanoFrameLength >= NANO_MAX_FRAME) {

    nanoFrameLength = 0;

    emgMalformedCount++;

    return;
  }

  nanoFrame[nanoFrameLength++] = b;

  // --------------------------------------------------------------------------
  // Byte 1 must be A5
  // --------------------------------------------------------------------------

  if (nanoFrameLength == 1) {

    if (b != NANO_SYNC0) {

      nanoFrameLength = 0;
      emgJunkByteCount++;
    }

    return;
  }

  // --------------------------------------------------------------------------
  // Byte 2 must be 5A
  // --------------------------------------------------------------------------

  if (nanoFrameLength == 2) {

    if (b != NANO_SYNC1) {

      nanoFrameLength = 0;
      emgJunkByteCount++;
    }

    return;
  }

  // --------------------------------------------------------------------------
  // Byte 3 = frame type
  // --------------------------------------------------------------------------

  if (nanoFrameLength == 3) {

    if (
      b != NANO_TYPE_EMG &&
      b != NANO_TYPE_STATS
    ) {

      emgHeaderErrorCount++;
      emgMalformedCount++;

      nanoFrameLength = 0;
    }

    return;
  }

  // --------------------------------------------------------------------------
  // Byte 4 = payload length
  // --------------------------------------------------------------------------

  if (nanoFrameLength == 4) {

    uint8_t type =
      nanoFrame[2];

    uint8_t length =
      nanoFrame[3];

    bool validLength = false;

    if (type == NANO_TYPE_EMG) {

      validLength =
        length >=
          NANO_EMG_HEADER_LEN +
          NANO_EMG_SAMPLE_LEN;

      validLength =
        validLength &&
        length <=
          NANO_EMG_HEADER_LEN +
          NANO_MAX_SAMPLES *
          NANO_EMG_SAMPLE_LEN;

      validLength =
        validLength &&
        ((length -
          NANO_EMG_HEADER_LEN) %
          NANO_EMG_SAMPLE_LEN == 0);

    } else {

      validLength =
        (length == NANO_STATS_LEN);
    }

    if (!validLength) {

      emgHeaderErrorCount++;
      emgMalformedCount++;

      nanoFrameLength = 0;
    }

    return;
  }

  // --------------------------------------------------------------------------
  // We now know complete frame length
  // --------------------------------------------------------------------------

  uint8_t payloadLength =
    nanoFrame[3];

  uint16_t totalLength =
    4 +
    payloadLength +
    2;

  if (nanoFrameLength < totalLength) {

    return;
  }

  // --------------------------------------------------------------------------
  // Verify CRC
  // --------------------------------------------------------------------------

  uint16_t receivedCrc =
    readU16LE(
      nanoFrame +
      4 +
      payloadLength
    );

  uint16_t calculatedCrc =
    crc16Mcrf4xx(
      nanoFrame + 2,
      payloadLength + 2
    );

  if (receivedCrc != calculatedCrc) {

    emgCrcErrorCount++;
    emgMalformedCount++;

    nanoFrameLength = 0;

    return;
  }

  // --------------------------------------------------------------------------
  // Valid frame
  // --------------------------------------------------------------------------

  uint8_t type =
    nanoFrame[2];

  const uint8_t* payload =
    nanoFrame + 4;

  if (type == NANO_TYPE_EMG) {

    handleNanoEmgFrame(
      payload,
      payloadLength
    );

  } else if (type == NANO_TYPE_STATS) {

    handleNanoStatsFrame(
      payload,
      payloadLength
    );
  }

  // Ready for next frame
  nanoFrameLength = 0;
}

// ============================================================================
// POLL NANO UART
// ============================================================================

void pollEmgUart() {

  while (Serial2.available() > 0) {

    int c =
      Serial2.read();

    if (c < 0) {
      break;
    }

    feedNanoByte(
      (uint8_t)c
    );
  }
}

// ============================================================================
// ESP32 -> PI EMG PACKET
// ============================================================================

uint8_t emgPacketBuf[
  12 +
  MAX_EMG_SAMPLES_PER_PACKET * 10
];

// ============================================================================
// SEND ONE EMG UDP BATCH
// ============================================================================

void sendOneEmgBatch(
  uint16_t sampleCount
) {

  uint8_t* p =
    emgPacketBuf;

  // Marker
  *p++ = 'E';

  // Client ID
  *p++ = (uint8_t)CLIENT_ID;

  // Batch sequence
  putU32LE(
    p,
    emgBatchSeq++
  );

  p += 4;

  // ESP32 timestamp
  putU32LE(
    p,
    micros()
  );

  p += 4;

  // Number of samples
  putU16LE(
    p,
    sampleCount
  );

  p += 2;

  // Samples
  for (
    uint16_t i = 0;
    i < sampleCount;
    i++
  ) {

    EmgSample& s =
      emgBuffer[emgTail];

    putU32LE(
      p,
      s.sampleNumber
    );

    p += 4;

    putU32LE(
      p,
      s.timestampUs
    );

    p += 4;

    putU16LE(
      p,
      s.adcValue
    );

    p += 2;

    emgTail =
      (emgTail + 1) &
      EMG_BUFFER_MASK;
  }

  uint16_t packetLength =
    p - emgPacketBuf;

  udp.beginPacket(
    PI_IP_ADDR,
    PI_UDP_PORT
  );

  udp.write(
    emgPacketBuf,
    packetLength
  );

  if (udp.endPacket()) {

    emgSentToPi += sampleCount;
  }
}

// ============================================================================
// FLUSH EMG BUFFER
// ============================================================================

void flushEmgBuffer() {

  uint8_t packetsSent = 0;

  while (
    packetsSent <
    MAX_EMG_PACKETS_PER_TICK
  ) {

    uint16_t available =
      (emgHead - emgTail) &
      EMG_BUFFER_MASK;

    if (available == 0) {
      break;
    }

    uint16_t n =
      available >
      MAX_EMG_SAMPLES_PER_PACKET
      ? MAX_EMG_SAMPLES_PER_PACKET
      : available;

    sendOneEmgBatch(n);

    packetsSent++;
  }
}

// ============================================================================
// EMG DEBUG REPORT
// ============================================================================

uint32_t lastEmgStatsMs = 0;

uint32_t previousParsed = 0;
uint32_t previousSent = 0;

void printEmgStatsIfDue(
  uint32_t nowMs
) {

  if (
    nowMs -
    lastEmgStatsMs <
    1000
  ) {

    return;
  }

  lastEmgStatsMs =
    nowMs;

  uint16_t backlog =
    (emgHead - emgTail) &
    EMG_BUFFER_MASK;

  uint32_t parsedThisSecond =
    emgParsedCount -
    previousParsed;

  uint32_t sentThisSecond =
    emgSentToPi -
    previousSent;

  previousParsed =
    emgParsedCount;

  previousSent =
    emgSentToPi;

  Serial.printf(
    "\n[ESP32 EMG]\n"
    "  parsed/s       = %lu\n"
    "  sent_to_pi/s   = %lu\n"
    "  frames         = %lu\n"
    "  malformed      = %lu\n"
    "  CRC errors     = %lu\n"
    "  header errors  = %lu\n"
    "  payload errors = %lu\n"
    "  overflow       = %lu\n"
    "  backlog        = %u\n"
    "  packet gaps    = %lu\n"
    "  sample gaps    = %lu\n"
    "  Nano drop flag = %lu\n"
    "  junk bytes     = %lu\n",

    (unsigned long)parsedThisSecond,
    (unsigned long)sentThisSecond,
    (unsigned long)emgFrameCount,
    (unsigned long)emgMalformedCount,
    (unsigned long)emgCrcErrorCount,
    (unsigned long)emgHeaderErrorCount,
    (unsigned long)emgPayloadErrorCount,
    (unsigned long)emgOverflowCount,
    backlog,
    (unsigned long)emgPacketLostCount,
    (unsigned long)emgSampleGapCount,
    (unsigned long)emgNanoDropFlagCount,
    (unsigned long)emgJunkByteCount
  );

  // --------------------------------------------------------------------------
  // Nano statistics
  // --------------------------------------------------------------------------

  if (nanoStats.fresh) {

    nanoStats.fresh = false;

    float adcRate = 0;
    float txRate = 0;

    if (
      havePreviousNanoStats &&
      nanoStats.uptimeMs >
      previousNanoStats.uptimeMs
    ) {

      float seconds =
        (
          nanoStats.uptimeMs -
          previousNanoStats.uptimeMs
        ) / 1000.0f;

      if (seconds > 0) {

        adcRate =
          (
            nanoStats.adcTotal -
            previousNanoStats.adcTotal
          ) / seconds;

        txRate =
          (
            nanoStats.txSamples -
            previousNanoStats.txSamples
          ) / seconds;
      }
    }

    Serial.printf(
      "[NANO STATS]\n"
      "  uptime       = %lu s\n"
      "  ADC total    = %lu\n"
      "  TX samples   = %lu\n"
      "  TX packets   = %lu\n"
      "  dropped      = %lu\n"
      "  ring HWM     = %u\n"
      "  sample dt    = %u..%u us\n"
      "  ADC rate     = %.1f /s\n"
      "  TX rate      = %.1f /s\n",

      (unsigned long)(
        nanoStats.uptimeMs / 1000
      ),

      (unsigned long)
        nanoStats.adcTotal,

      (unsigned long)
        nanoStats.txSamples,

      (unsigned long)
        nanoStats.txPackets,

      (unsigned long)
        nanoStats.dropped,

      nanoStats.ringHwm,

      nanoStats.dtMin,
      nanoStats.dtMax,

      adcRate,
      txRate
    );

    previousNanoStats =
      nanoStats;

    havePreviousNanoStats =
      true;
  }
}

// ============================================================================
// MPU LOW LEVEL
// ============================================================================

void mpuWriteReg(
  uint8_t reg,
  uint8_t val
) {

  Wire.beginTransmission(
    MPU_ADDR
  );

  Wire.write(reg);
  Wire.write(val);

  Wire.endTransmission();
}

uint8_t mpuReadReg(
  uint8_t reg
) {

  Wire.beginTransmission(
    MPU_ADDR
  );

  Wire.write(reg);

  Wire.endTransmission(false);

  Wire.requestFrom(
    (int)MPU_ADDR,
    1,
    true
  );

  return
    Wire.available()
    ? Wire.read()
    : 0;
}

// ============================================================================
// MPU RAW READ
// ============================================================================

void mpuReadRaw(
  int16_t &ax,
  int16_t &ay,
  int16_t &az,
  int16_t &gx,
  int16_t &gy,
  int16_t &gz
) {

  Wire.beginTransmission(
    MPU_ADDR
  );

  Wire.write(
    REG_ACCEL_XOUT_H
  );

  Wire.endTransmission(false);

  Wire.requestFrom(
    (int)MPU_ADDR,
    14,
    true
  );

  ax =
    (Wire.read() << 8) |
    Wire.read();

  ay =
    (Wire.read() << 8) |
    Wire.read();

  az =
    (Wire.read() << 8) |
    Wire.read();

  // Temperature
  Wire.read();
  Wire.read();

  gx =
    (Wire.read() << 8) |
    Wire.read();

  gy =
    (Wire.read() << 8) |
    Wire.read();

  gz =
    (Wire.read() << 8) |
    Wire.read();
}

// ============================================================================
// MPU INIT
// ============================================================================

void mpuInit() {

  mpuWriteReg(
    REG_PWR_MGMT_1,
    0x01
  );

  delay(50);

  mpuWriteReg(
    REG_SMPLRT_DIV,
    0x00
  );

  mpuWriteReg(
    REG_CONFIG,
    0x03
  );

  mpuWriteReg(
    REG_GYRO_CONFIG,
    0x10
  );

  mpuWriteReg(
    REG_ACCEL_CONFIG,
    0x08
  );

  delay(50);
}

// ============================================================================
// GYRO BIAS
// ============================================================================

void calibrateGyroBias() {

  const int N = 1000;

  double sx = 0;
  double sy = 0;
  double sz = 0;

  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  Serial.printf(
    "[ID %d] Calibrating gyro bias -- KEEP STILL...\n",
    CLIENT_ID
  );

  for (int i = 0; i < N; i++) {

    mpuReadRaw(
      ax,
      ay,
      az,
      gx,
      gy,
      gz
    );

    sx += gx;
    sy += gy;
    sz += gz;

    delay(2);
  }

  gyroBiasX =
    sx / N;

  gyroBiasY =
    sy / N;

  gyroBiasZ =
    sz / N;

  Serial.printf(
    "[ID %d] Gyro bias: %.2f, %.2f, %.2f (raw LSB)\n",
    CLIENT_ID,
    gyroBiasX,
    gyroBiasY,
    gyroBiasZ
  );
}

// ============================================================================
// INITIAL QUATERNION
// ============================================================================

void initQuaternionFromAccel() {

  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpuReadRaw(
    ax,
    ay,
    az,
    gx,
    gy,
    gz
  );

  float axg =
    ax / ACCEL_SCALE;

  float ayg =
    ay / ACCEL_SCALE;

  float azg =
    az / ACCEL_SCALE;

  float roll =
    atan2f(
      ayg,
      azg
    );

  float pitch =
    atan2f(
      -axg,
      sqrtf(
        ayg * ayg +
        azg * azg
      )
    );

  float cr =
    cosf(
      roll * 0.5f
    );

  float sr =
    sinf(
      roll * 0.5f
    );

  float cp =
    cosf(
      pitch * 0.5f
    );

  float sp =
    sinf(
      pitch * 0.5f
    );

  q0 =
    cr * cp;

  q1 =
    sr * cp;

  q2 =
    cr * sp;

  q3 =
    -sr * sp;
}

// ============================================================================
// MADGWICK IMU
// ============================================================================

void madgwickUpdateIMU(
  float gx,
  float gy,
  float gz,
  float ax,
  float ay,
  float az,
  float dt
) {

  float qDot0 =
    0.5f *
    (
      -q1 * gx
      -q2 * gy
      -q3 * gz
    );

  float qDot1 =
    0.5f *
    (
      q0 * gx
      + q2 * gz
      - q3 * gy
    );

  float qDot2 =
    0.5f *
    (
      q0 * gy
      - q1 * gz
      + q3 * gx
    );

  float qDot3 =
    0.5f *
    (
      q0 * gz
      + q1 * gy
      - q2 * gx
    );

  if (
    !(ax == 0.0f &&
      ay == 0.0f &&
      az == 0.0f)
  ) {

    float norm =
      sqrtf(
        ax * ax +
        ay * ay +
        az * az
      );

    ax /= norm;
    ay /= norm;
    az /= norm;

    float _2q0 = 2.0f * q0;
    float _2q1 = 2.0f * q1;
    float _2q2 = 2.0f * q2;
    float _2q3 = 2.0f * q3;

    float _4q0 = 4.0f * q0;
    float _4q1 = 4.0f * q1;
    float _4q2 = 4.0f * q2;

    float _8q1 = 8.0f * q1;
    float _8q2 = 8.0f * q2;

    float q0q0 = q0 * q0;
    float q1q1 = q1 * q1;
    float q2q2 = q2 * q2;
    float q3q3 = q3 * q3;

    float s0 =
      _4q0 * q2q2 +
      _2q2 * ax +
      _4q0 * q1q1 -
      _2q1 * ay;

    float s1 =
      _4q1 * q3q3 -
      _2q3 * ax +
      4.0f * q0q0 * q1 -
      _2q0 * ay -
      _4q1 +
      _8q1 * q1q1 +
      _8q1 * q2q2 +
      _4q1 * az;

    float s2 =
      4.0f * q0q0 * q2 +
      _2q0 * ax +
      _4q2 * q3q3 -
      _2q3 * ay -
      _4q2 +
      _8q2 * q1q1 +
      _8q2 * q2q2 +
      _4q2 * az;

    float s3 =
      4.0f * q1q1 * q3 -
      _2q1 * ax +
      4.0f * q2q2 * q3 -
      _2q2 * ay;

    float sNorm =
      sqrtf(
        s0 * s0 +
        s1 * s1 +
        s2 * s2 +
        s3 * s3
      );

    if (sNorm > 1e-9f) {

      s0 /= sNorm;
      s1 /= sNorm;
      s2 /= sNorm;
      s3 /= sNorm;

      qDot0 -=
        MADGWICK_BETA * s0;

      qDot1 -=
        MADGWICK_BETA * s1;

      qDot2 -=
        MADGWICK_BETA * s2;

      qDot3 -=
        MADGWICK_BETA * s3;
    }
  }

  q0 +=
    qDot0 * dt;

  q1 +=
    qDot1 * dt;

  q2 +=
    qDot2 * dt;

  q3 +=
    qDot3 * dt;

  float qNorm =
    sqrtf(
      q0 * q0 +
      q1 * q1 +
      q2 * q2 +
      q3 * q3
    );

  q0 /= qNorm;
  q1 /= qNorm;
  q2 /= qNorm;
  q3 /= qNorm;
}

// ============================================================================
// WIFI
// ============================================================================

void connectWiFi() {

  Serial.printf(
    "[ID %d] Connecting to WiFi \"%s\"",
    CLIENT_ID,
    WIFI_SSID
  );

  WiFi.mode(WIFI_STA);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  uint32_t start =
    millis();

  while (
    WiFi.status() !=
    WL_CONNECTED
  ) {

    delay(300);

    Serial.print(".");

    if (
      millis() - start >
      20000
    ) {

      Serial.println(
        "\n  -> still not connected, retrying..."
      );

      WiFi.disconnect();

      WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
      );

      start =
        millis();
    }
  }

  Serial.printf(
    "\n[ID %d] WiFi connected. IP: %s\n",
    CLIENT_ID,
    WiFi.localIP()
      .toString()
      .c_str()
  );
}

// ============================================================================
// SETUP
// ============================================================================

uint32_t lastFilterMicros = 0;

void setup() {

  Serial.begin(115200);

  delay(200);

  Serial.printf(
    "\n=== ROM Sensor Node — CLIENT_ID %d (MPU6500 + BINARY EMG) ===\n",
    CLIENT_ID
  );

  // --------------------------------------------------------------------------
  // Nano UART
  // --------------------------------------------------------------------------

  Serial2.setRxBufferSize(
    EMG_UART_RX_BUFFER_BYTES
  );

  Serial2.begin(
    EMG_UART_BAUD,
    SERIAL_8N1,
    EMG_UART_RX_PIN,
    EMG_UART_TX_PIN
  );

  Serial.printf(
    "[ID %d] Nano UART2: %lu baud, RX=GPIO%d\n",
    CLIENT_ID,
    (unsigned long)EMG_UART_BAUD,
    EMG_UART_RX_PIN
  );

  // --------------------------------------------------------------------------
  // MPU
  // --------------------------------------------------------------------------

  Wire.begin();

  Wire.setClock(400000);

  uint8_t whoAmI =
    mpuReadReg(
      REG_WHO_AM_I
    );

  Serial.printf(
    "[ID %d] WHO_AM_I = 0x%02X (expect ~0x70)\n",
    CLIENT_ID,
    whoAmI
  );

  mpuInit();

  calibrateGyroBias();

  initQuaternionFromAccel();

  // --------------------------------------------------------------------------
  // WiFi
  // --------------------------------------------------------------------------

  connectWiFi();

  udp.begin(
    PI_UDP_PORT
  );

  lastFilterMicros =
    micros();

  Serial.printf(
    "[ID %d] Streaming IMU + binary EMG to %s:%d @ 50 Hz\n",
    CLIENT_ID,
    PI_IP_ADDR,
    PI_UDP_PORT
  );
}

// ============================================================================
// LOOP
// ============================================================================

void loop() {

  // --------------------------------------------------------------------------
  // Make sure WiFi is connected
  // --------------------------------------------------------------------------

  if (
    WiFi.status() !=
    WL_CONNECTED
  ) {

    connectWiFi();
  }

  // --------------------------------------------------------------------------
  // Read Nano binary data
  // --------------------------------------------------------------------------

  pollEmgUart();

  // --------------------------------------------------------------------------
  // Read MPU6500
  // --------------------------------------------------------------------------

  int16_t rax;
  int16_t ray;
  int16_t raz;

  int16_t rgx;
  int16_t rgy;
  int16_t rgz;

  mpuReadRaw(
    rax,
    ray,
    raz,
    rgx,
    rgy,
    rgz
  );

  // --------------------------------------------------------------------------
  // Madgwick
  // --------------------------------------------------------------------------

  uint32_t nowMicros =
    micros();

  float dt =
    (
      nowMicros -
      lastFilterMicros
    ) / 1000000.0f;

  lastFilterMicros =
    nowMicros;

  if (
    dt <= 0 ||
    dt > 0.2f
  ) {

    dt = 0.005f;
  }

  float ax =
    rax / ACCEL_SCALE;

  float ay =
    ray / ACCEL_SCALE;

  float az =
    raz / ACCEL_SCALE;

  float gx =
    (
      rgx -
      gyroBiasX
    ) /
    GYRO_SCALE *
    (PI / 180.0f);

  float gy =
    (
      rgy -
      gyroBiasY
    ) /
    GYRO_SCALE *
    (PI / 180.0f);

  float gz =
    (
      rgz -
      gyroBiasZ
    ) /
    GYRO_SCALE *
    (PI / 180.0f);

  madgwickUpdateIMU(
    gx,
    gy,
    gz,
    ax,
    ay,
    az,
    dt
  );

  // --------------------------------------------------------------------------
  // 50 Hz UDP tick
  // --------------------------------------------------------------------------

  uint32_t nowMs =
    millis();

  if (
    nowMs -
    lastSendMs >=
    SEND_INTERVAL_MS
  ) {

    lastSendMs =
      nowMs;

    // ------------------------------------------------------------------------
    // IMU UDP packet
    // ------------------------------------------------------------------------

    char buf[96];

    int len =
      snprintf(
        buf,
        sizeof(buf),
        "%d,%lu,%.6f,%.6f,%.6f,%.6f\n",
        CLIENT_ID,
        (unsigned long)seqCounter++,
        q0,
        q1,
        q2,
        q3
      );

    udp.beginPacket(
      PI_IP_ADDR,
      PI_UDP_PORT
    );

    udp.write(
      (const uint8_t*)buf,
      len
    );

    udp.endPacket();

    // ------------------------------------------------------------------------
    // EMG UDP packets
    // ------------------------------------------------------------------------

    flushEmgBuffer();
  }

  // --------------------------------------------------------------------------
  // Debug information
  // --------------------------------------------------------------------------

  printEmgStatsIfDue(
    nowMs
  );

  // --------------------------------------------------------------------------
  // Drain Nano UART again
  // --------------------------------------------------------------------------

  pollEmgUart();
}