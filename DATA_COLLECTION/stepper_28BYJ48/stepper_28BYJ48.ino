/*
  28BYJ-48 stepper motor with ULN2003 driver
  Wiring: IN1->D8, IN2->D9, IN3->D10, IN4->D11
*/

#include <Stepper.h>

// 28BYJ-48: 64 steps/rev internal, 1/64 gear ratio -> 2048 steps/full rev
const int STEPS_PER_REV = 2048;

// Pin order matches the ULN2003 driver coil firing sequence
Stepper stepper(STEPS_PER_REV, 8, 10, 9, 11);

void setup() {
  stepper.setSpeed(10);  // RPM (max reliable ~15 for 28BYJ-48)
  Serial.begin(9600);
}

void loop() {
  Serial.println("CW one full revolution");
  stepper.step(STEPS_PER_REV);
  delay(1000);

  Serial.println("CCW one full revolution");
  stepper.step(-STEPS_PER_REV);
  delay(1000);
}
