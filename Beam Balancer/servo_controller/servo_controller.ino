#include <Servo.h>

// ----- Servo configuration -----
Servo servo1;
Servo servo2;
Servo servo3;

const int servoPin1 = 9;
const int servoPin2 = 10;
const int servoPin3 = 11;

const int MIN_ANGLE = 0;
const int MAX_ANGLE = 30;
const int NEUTRAL_ANGLE = 15;

// Communication variables
int targetAngle = NEUTRAL_ANGLE;
bool newCommand = false;

void setup() {
  Serial.begin(9600);

  servo1.attach(servoPin1);
  servo2.attach(servoPin2);
  servo3.attach(servoPin3);

  // Move all servos to neutral
  servo1.write(NEUTRAL_ANGLE);
  servo2.write(NEUTRAL_ANGLE);
  servo3.write(NEUTRAL_ANGLE);

  Serial.println("3-servo controller ready");
}

void loop() {

  if (Serial.available() > 0) {
    int receivedAngle = Serial.read();

    if (receivedAngle >= MIN_ANGLE && receivedAngle <= MAX_ANGLE) {
      targetAngle = receivedAngle;
      newCommand = true;
    }
  }

  if (newCommand) {
    // Move servo 1
    servo1.write(targetAngle);
    delay(200); // wait 0.2s

    // Move servo 2
    servo2.write(targetAngle);
    delay(200);

    // Move servo 3
    servo3.write(targetAngle);
    delay(200);

    newCommand = false;
  }

  delay(10);
}
