/*
  NEMA 17 + DRV8825 basic rotation test.
  One full rotation forward, pause, one full rotation backward, repeat.

  Wiring:
    STEP  → D3
    DIR   → D4
    SLEEP → D5 (also connect RST to D5)
    EN    → GND
    M0/M1/M2 → GND (or leave unconnected)  full step
*/

#define STEP_PIN   3
#define DIR_PIN    4
#define SLEEP_PIN  5

const int    STEPS_PER_REV  = 200;    // full step, no microstepping
const unsigned int DELAY_US = 500;    // half-pulse width

void stepMotor(int steps, bool forward) {
  digitalWrite(DIR_PIN, forward ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(DELAY_US);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(DELAY_US);
  }
}

void setup() {
  pinMode(STEP_PIN,  OUTPUT);
  pinMode(DIR_PIN,   OUTPUT);
  pinMode(SLEEP_PIN, OUTPUT);
  digitalWrite(SLEEP_PIN, HIGH);
  delayMicroseconds(2000);
}

void loop() {
  stepMotor(STEPS_PER_REV, true);   // forward
  delay(1000);
  stepMotor(STEPS_PER_REV, false);  // backward
  delay(1000);
}
