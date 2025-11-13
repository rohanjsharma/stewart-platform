#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// ---- CONFIG ----
const uint8_t SERVO_CH[3] = {0, 1, 2};   // channels 0..2
const float   MIN_ANGLE   = 0.0;
const float   MAX_ANGLE   = 50.0;

// Per-servo pulse limits (tune to your servos/mechanics)
const int SERVO_US_MIN[3] = {500, 500, 500};
const int SERVO_US_MAX[3] = {2600, 2600, 2600};

// Electrical range mapping (common: 180 or 120)
const float SERVO_FULL_RANGE_DEG = 180.0;

// Initial neutral angle to park at on boot (your Python uses 15° neutral)
const float BOOT_NEUTRAL_DEG = 15.0;
// ----------------

static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void setup() {
  Serial.begin(9600);

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  delay(10);

  // Park all three servos at neutral
  writeAngleToServo(0, 15);
  writeAngleToServo(1, 25);
  writeAngleToServo(2, 11);
}

// Map angle (deg) -> microseconds for servo idx
void writeAngleToServo(uint8_t idx, float angleDeg) {
  angleDeg = clampf(angleDeg, MIN_ANGLE, MAX_ANGLE);
  float usPerDeg = (SERVO_US_MAX[idx] - SERVO_US_MIN[idx]) / SERVO_FULL_RANGE_DEG;
  int targetUs = SERVO_US_MIN[idx] + (int)(angleDeg * usPerDeg + 0.5f);
  pwm.writeMicroseconds(SERVO_CH[idx], targetUs);
}

void loop() {
  // Process the *latest* full triple if multiple have arrived
  while (Serial.available() >= 3) {
    uint8_t a0 = Serial.read();
    uint8_t a1 = Serial.read();
    uint8_t a2 = Serial.read();

    writeAngleToServo(0, (float)a0);
    writeAngleToServo(1, (float)a1);
    writeAngleToServo(2, (float)a2);
  }
}
