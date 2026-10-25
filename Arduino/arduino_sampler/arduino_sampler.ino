/*
 * Nano (ATmega328P) EMG acquisition @ 2000 Hz
 * -> compact binary UART @ 1,000,000 baud
 *
 * Architecture:
 *   Timer1 CTC -> OCR1B ADC trigger -> ADC ISR -> ring buffer
 *   -> main loop -> CRC-protected binary UART frames -> ESP32
 *
 * ---------------------------------------------------------------------------
 * FRAME FORMAT (all multi-byte fields little-endian)
 *
 *   off  size  field
 *   0    1     0xA5                sync 0
 *   1    1     0x5A                sync 1
 *   2    1     type
 *                 0x01 = EMG batch
 *                 0x02 = STATS
 *   3    1     payload_len
 *   4    len   payload
 *   4+len 2    CRC16
 *
 * CRC:
 *   CRC-16/MCRF4XX
 *   polynomial 0x8408 reflected
 *   init 0xFFFF
 *   no xorout
 *   calculated over [type, payload_len, payload]
 *
 * EMG payload:
 *
 *   0    u32 seq
 *   4    u32 first_sample
 *   8    u8  count
 *   9    u8  flags
 *             bit 0 = ADC ISR dropped >=1 sample
 *   10   count x:
 *          u32 timestamp_us
 *          u16 adc
 *
 * STATS payload:
 *
 *   0    u32 uptime_ms
 *   4    u32 adc_total
 *   8    u32 tx_samples
 *   12   u32 tx_packets
 *   16   u32 dropped
 *   20   u16 ring_hwm
 *   22   u16 dt_min_us
 *   24   u16 dt_max_us
 *
 * IMPORTANT:
 *   NOTHING else is written to Serial.
 *   Do not add Serial.print() anywhere.
 * ---------------------------------------------------------------------------
 */

#include <Arduino.h>
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/atomic.h>
#include <util/crc16.h>

// ----------------------------- configuration -------------------------------

#define SAMPLE_RATE_HZ     2000UL
#define TIMER1_TOP         999
#define TIMER1_ADC_TRIG    499
#define UART_BAUD          1000000UL

#define RING_SIZE          128
#define RING_MASK          (RING_SIZE - 1)

#define PKT_SAMPLES        16

#define SYNC0              0xA5
#define SYNC1              0x5A

#define TYPE_EMG           0x01
#define TYPE_STATS         0x02

#define EMG_HDR_LEN        10
#define EMG_SAMPLE_LEN     6

#define STATS_LEN          26

#define STATS_INTERVAL_MS  1000UL

static_assert(
  (F_CPU / 8UL) / (TIMER1_TOP + 1UL) == SAMPLE_RATE_HZ,
  "Timer1 TOP does not give SAMPLE_RATE_HZ"
);

static_assert(
  (RING_SIZE & (RING_SIZE - 1)) == 0,
  "RING_SIZE must be a power of two"
);

// ------------------------------ ring buffer --------------------------------

struct RingEntry {
  uint32_t ts_us;
  uint16_t adc;

  // Low 16 bits of sample number.
  // Main loop reconstructs the full 32-bit number.
  uint16_t sn16;
};

static_assert(sizeof(RingEntry) == 8, "RingEntry size");

static RingEntry ring[RING_SIZE];

static volatile uint8_t ringHead = 0;
static volatile uint8_t ringTail = 0;

static volatile uint32_t adcTotal = 0;
static volatile uint32_t droppedTotal = 0;

// ----------------------------------- ISR ------------------------------------

ISR(ADC_vect) {

  // Clear Timer1 Compare Match B flag so the next trigger can occur.
  TIFR1 = _BV(OCF1B);

  const uint16_t v = ADC;
  const uint32_t ts = micros();

  const uint8_t h = ringHead;
  const uint8_t nx = (h + 1) & RING_MASK;

  const uint32_t n = adcTotal;

  // Sample number advances even if the sample has to be dropped.
  adcTotal = n + 1;

  // Ring full.
  if (nx == ringTail) {
    droppedTotal++;
    return;
  }

  ring[h].ts_us = ts;
  ring[h].adc   = v;
  ring[h].sn16  = (uint16_t)n;

  ringHead = nx;
}

// ------------------------------ TX-side state -------------------------------

static uint8_t txBuf[
  4 +
  EMG_HDR_LEN +
  PKT_SAMPLES * EMG_SAMPLE_LEN +
  2
];

static uint32_t txSeq = 0;

// Next expected sample number.
static uint32_t txNext = 0;

static uint32_t txSamples = 0;
static uint32_t txPackets = 0;

static uint32_t lastDroppedSeen = 0;

// Timestamp statistics.
static uint32_t prevTs = 0;
static uint32_t prevSn = 0;
static bool prevValid = false;

static uint16_t dtMin = 0xFFFF;
static uint16_t dtMax = 0;

static uint8_t ringHwm = 0;

static uint32_t nextStatsMs = 0;

// ------------------------------ serialization -------------------------------

static inline uint8_t* put16(uint8_t* p, uint16_t v) {

  *p++ = (uint8_t)v;
  *p++ = (uint8_t)(v >> 8);

  return p;
}

static inline uint8_t* put32(uint8_t* p, uint32_t v) {

  *p++ = (uint8_t)v;
  *p++ = (uint8_t)(v >> 8);
  *p++ = (uint8_t)(v >> 16);
  *p++ = (uint8_t)(v >> 24);

  return p;
}

// ------------------------------ frame sender --------------------------------

static void finishAndSend(uint8_t type, uint8_t len) {

  txBuf[0] = SYNC0;
  txBuf[1] = SYNC1;
  txBuf[2] = type;
  txBuf[3] = len;

  uint16_t crc = 0xFFFF;

  const uint8_t* d = txBuf + 2;

  for (uint8_t i = 0; i < (uint8_t)(len + 2); i++) {
    crc = _crc_ccitt_update(crc, d[i]);
  }

  put16(txBuf + 4 + len, crc);

  /*
   * Only the main loop writes to Serial.
   */
  Serial.write(
    txBuf,
    (size_t)len + 6
  );
}

// ------------------------------ EMG packet ----------------------------------

static void sendEmgPacket() {

  uint8_t t = ringTail;

  uint8_t avail =
    (uint8_t)(ringHead - t) & RING_MASK;

  if (avail > PKT_SAMPLES) {
    avail = PKT_SAMPLES;
  }

  if (avail == 0) {
    return;
  }

  uint8_t* p =
    txBuf + 4 + EMG_HDR_LEN;

  uint8_t n = 0;

  uint32_t firstSample = 0;

  /*
   * We build a packet containing consecutive samples.
   *
   * IMPORTANT:
   * If the Nano dropped a sample, there will be a gap in
   * the sample numbers.
   *
   * Previously, the code simply did:
   *
   *     break;
   *
   * and left the problematic sample in the ring.
   *
   * That could permanently stall transmission.
   *
   * Now we explicitly resynchronize txNext when a gap is
   * encountered.
   */

  while (n < avail) {

    const RingEntry& e = ring[t];

    /*
     * Reconstruct the 32-bit sample number from the
     * 16-bit value stored in the ring.
     *
     * Because the ring contains only a small number of
     * samples, the nearest value relative to txNext is
     * unambiguous.
     */
    const uint32_t full =
      txNext +
      (uint16_t)(e.sn16 - (uint16_t)txNext);

    /*
     * First sample in this packet.
     *
     * If txNext does not match this sample, a sample was
     * dropped before this entry.
     */
    if (n == 0) {

      if (full != txNext) {

        /*
         * RESYNC FIX
         *
         * A sample or samples were dropped.
         *
         * Do not get stuck waiting forever for the missing
         * sample. Treat the first currently available sample
         * as the new synchronization point.
         */
        txNext = full;

      }

      firstSample = full;

    } else {

      /*
       * After resynchronization, samples inside one packet
       * must still be consecutive.
       *
       * If another gap appears, stop this packet.
       * The next call will resynchronize again.
       */
      if (full != txNext) {
        break;
      }
    }

    /*
     * Update timestamp interval statistics.
     *
     * Only calculate dt when the sample number is actually
     * consecutive with the previous transmitted sample.
     */
    if (prevValid && full == prevSn + 1) {

      uint32_t dt = e.ts_us - prevTs;

      if (dt > 0xFFFF) {
        dt = 0xFFFF;
      }

      if ((uint16_t)dt < dtMin) {
        dtMin = (uint16_t)dt;
      }

      if ((uint16_t)dt > dtMax) {
        dtMax = (uint16_t)dt;
      }
    }

    prevTs = e.ts_us;
    prevSn = full;
    prevValid = true;

    /*
     * Add timestamp + ADC value to packet.
     */
    p = put32(p, e.ts_us);
    p = put16(p, e.adc);

    /*
     * Next expected sample.
     */
    txNext = full + 1;

    /*
     * Consume this ring entry.
     */
    t = (t + 1) & RING_MASK;

    n++;
  }

  /*
   * Safety check.
   */
  if (n == 0) {
    return;
  }

  /*
   * Release ring slots before the potentially blocking
   * Serial.write().
   */
  ringTail = t;

  /*
   * Check whether the ADC ISR has dropped anything since
   * the previous EMG packet.
   */
  uint32_t dr;

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    dr = droppedTotal;
  }

  uint8_t flags = 0;

  if (dr != lastDroppedSeen) {
    flags |= 0x01;
  }

  lastDroppedSeen = dr;

  /*
   * Construct EMG header.
   */
  uint8_t* h = txBuf + 4;

  h = put32(h, txSeq++);

  h = put32(h, firstSample);

  *h++ = n;

  *h++ = flags;

  /*
   * Statistics.
   */
  txSamples += n;
  txPackets++;

  /*
   * Send binary EMG frame.
   */
  finishAndSend(
    TYPE_EMG,
    EMG_HDR_LEN + n * EMG_SAMPLE_LEN
  );
}

// ------------------------------ stats packet --------------------------------

static void sendStatsPacket() {

  uint32_t adcT;
  uint32_t dr;

  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    adcT = adcTotal;
    dr = droppedTotal;
  }

  uint8_t* p = txBuf + 4;

  p = put32(p, millis());
  p = put32(p, adcT);
  p = put32(p, txSamples);
  p = put32(p, txPackets);
  p = put32(p, dr);
  p = put16(p, ringHwm);
  p = put16(p, dtMin);
  p = put16(p, dtMax);

  /*
   * Reset interval statistics.
   */
  ringHwm = 0;
  dtMin = 0xFFFF;
  dtMax = 0;

  finishAndSend(
    TYPE_STATS,
    STATS_LEN
  );
}

// ---------------------------------- setup -----------------------------------

void setup() {

  /*
   * UART:
   *
   * 1,000,000 baud
   */
  Serial.begin(UART_BAUD);

  /*
   * ADC:
   *
   * A0
   * AVcc reference
   * right-adjusted
   * ADC clock = 16 MHz / 64 = 250 kHz
   *
   * Timer1 Compare Match B triggers ADC conversion.
   */
  DIDR0 |= _BV(ADC0D);

  ADMUX = _BV(REFS0);

  /*
   * ADTS = 101
   *
   * Timer/Counter1 Compare Match B
   */
  ADCSRB =
    _BV(ADTS2) |
    _BV(ADTS0);

  /*
   * Enable:
   * ADC
   * Auto trigger
   * ADC interrupt
   * ADC prescaler /64
   */
  ADCSRA =
    _BV(ADEN)  |
    _BV(ADATE) |
    _BV(ADIE)  |
    _BV(ADPS2) |
    _BV(ADPS1);

  /*
   * Timer1 setup.
   */
  noInterrupts();

  TCCR1A = 0;
  TCCR1B = 0;

  TCNT1 = 0;

  /*
   * CTC mode 12:
   * TOP = ICR1
   */
  ICR1 = TIMER1_TOP;

  /*
   * ADC trigger point.
   */
  OCR1B = TIMER1_ADC_TRIG;

  /*
   * Clear pending Timer1 flags.
   */
  TIFR1 = 0xFF;

  /*
   * Timer1:
   *
   * WGM13 + WGM12
   * CTC mode 12
   *
   * CS11
   * prescaler /8
   */
  TCCR1B =
    _BV(WGM13) |
    _BV(WGM12) |
    _BV(CS11);

  interrupts();

  nextStatsMs =
    millis() + STATS_INTERVAL_MS;
}

// ---------------------------------- loop ------------------------------------

void loop() {

  /*
   * Determine current ring-buffer occupancy.
   */
  const uint8_t avail =
    (uint8_t)(ringHead - ringTail) & RING_MASK;

  /*
   * High-water mark.
   */
  if (avail > ringHwm) {
    ringHwm = avail;
  }

  /*
   * EMG has priority.
   *
   * Send whenever at least one packet's worth of samples
   * is available.
   */
  if (avail >= PKT_SAMPLES) {

    sendEmgPacket();

    return;
  }

  /*
   * Send statistics once per second.
   */
  if ((int32_t)(millis() - nextStatsMs) >= 0) {

    nextStatsMs += STATS_INTERVAL_MS;

    sendStatsPacket();
  }
}