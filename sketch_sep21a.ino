/*
  ESP32 + MAX30102
  Stream raw IR PPG (single column) at ~200 Hz to Serial

  CSV format:
    timestamp_ms,ppg

  Notes:
  - Sampling rate set in particleSensor.setup(...) to 200 Hz.
  - Timing loop uses micros() to produce stable ~200 Hz sample rate.
  - Output: raw IR counts (uint32). Do preprocessing / resampling on the PC.
*/

#include <Wire.h>
#include <MAX30105.h>

MAX30105 particleSensor;

// I2C pins (ESP32 typical)
#define SDA_PIN 21
#define SCL_PIN 22

// Desired output rate (Hz)
const uint16_t OUTPUT_HZ = 200;
const unsigned long SAMPLE_US = 1000000UL / OUTPUT_HZ; // microseconds per sample

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(1); }

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found. Check wiring.");
    while (1) { delay(1000); }
  }

  // Configure sensor:
  // particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange)
  // ledBrightness: 0-255 (LED current)
  // sampleAverage: 1,2,4,8,16,32 (use 4 to smooth)
  // ledMode: 1=Red only, 2=Red+IR, 3=Red+IR+Green (we keep Red+IR)
  // sampleRate: 50, 100, 200, 400, 800, 1000 (Hz)
  // pulseWidth: 69, 118, 215, 411 (ns) -> affects ADC resolution & SNR
  // adcRange: 2048, 4096, 8192, 16384 (depending on library build) - keep 4096 for common use
  particleSensor.setup(60, 4, 2, 200, 411, 4096); // 60 LED, avg 4, Red+IR, 200Hz, 411us PW, 4096 range

  // Give sensor a short warm-up
  delay(200);

  // Print CSV header
  Serial.println("timestamp_ms,ppg");
}

void loop() {
  static unsigned long nextMicros = 0;
  unsigned long now = micros();

  if (nextMicros == 0) {
    nextMicros = now;
  }

  if (now >= nextMicros) {
    // Read IR raw value
    uint32_t irRaw = particleSensor.getIR();

    // Timestamp in milliseconds (uint32)
    unsigned long tms = millis();

    // Print CSV line: timestamp_ms,ppg
    Serial.print(tms);
    Serial.print(",");
    Serial.println(irRaw);

    // Schedule next
    nextMicros += SAMPLE_US;

    // If the loop fell behind (e.g., after pause), avoid many rapid reads to catch up:
    // keep nextMicros near now
    if (nextMicros + SAMPLE_US < micros()) {
      nextMicros = micros() + SAMPLE_US;
    }
  }

  // small yield to let background tasks run
  // no blocking delay here; keep this tiny
  yield();
}
