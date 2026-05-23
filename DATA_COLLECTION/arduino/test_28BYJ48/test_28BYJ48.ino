/*
  28BYJ-48 + ULN2003 basic rotation test.
  One full rotation forward, pause, one full rotation backward, repeat.

  Wiring (ULN2003 driver):
    IN1 → D8
    IN2 → D9
    IN3 → D10
    IN4 → D11
    Power → external 5V supply (NOT Arduino 5V)
    GND  → Arduino GND + supply GND
*/

#include <Stepper.h>

const int STEPS_PER_REV = 2048;

// Pin order corrects ULN2003 coil sequence
Stepper stepper(STEPS_PER_REV, 8, 10, 9, 11);

void setup() {
  stepper.setSpeed(10);   // 10 RPM — slow and reliable
}

void loop() {
  stepper.step(STEPS_PER_REV);    // forward 360°
  delay(1000);
  stepper.step(-STEPS_PER_REV);   // backward 360°
  delay(1000);
}
