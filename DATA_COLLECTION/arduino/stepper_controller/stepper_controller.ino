/*
  28BYJ-48 + ULN2003 stepper controller for DoA data collection.

  Protocol (newline-terminated):
    PC → "ROTATE"    advance one 5° step, reply "READY"
    PC → "RESET"     step backward to position 0 (emergency use), reply "READY"
    PC → "STATUS"    reply "POS <n>"  (0-71)
    PC → "STEP <n>"  arbitrary step count (+forward / -backward), reply "READY"
    PC → "ZERO"      set current_position = 0 without moving, reply "READY"
    PC → "DEENERGIZE" cut all coil current, reply "READY"

  Wiring (ULN2003 driver):
    IN1 → D8,  IN2 → D9,  IN3 → D10,  IN4 → D11
    VCC → external 5V supply
    GND → external 5V supply GND + Arduino GND

  Coil policy:
    Coils stay ENERGISED during recording — DC hold causes no switching EMI
    and prevents position slip from cable/gravity torque between steps.
    Coils are cut only via explicit DEENERGIZE (called from Python after
    each completed sweep for the 10 s inter-sweep cool-down).

  Step math:
    28BYJ-48: 2048 steps = 360°.
    72 positions × 5° = 360°.
    stepsAtPosition(n) = round(n × 2048 / 72)  →  error < 0.09°
*/

#include <Stepper.h>

const int STEPS_PER_REV   = 2048;
const int TOTAL_POSITIONS = 72;

// Pin order corrects ULN2003 coil firing sequence
Stepper stepper(STEPS_PER_REV, 8, 10, 9, 11);

int current_position = 0;

int stepsAtPosition(int pos) {
  return (int)roundf((float)pos * STEPS_PER_REV / TOTAL_POSITIONS);
}

void deenergizeCoils() {
  digitalWrite(8, LOW);
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
}

void setup() {
  stepper.setSpeed(10);   // 10 RPM — conservative for reliability
  Serial.begin(9600);
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "ROTATE") {
    int next  = (current_position + 1) % TOTAL_POSITIONS;
    int from  = stepsAtPosition(current_position);
    int to    = stepsAtPosition(next);
    int delta = (next > current_position)
              ? to - from
              : STEPS_PER_REV - from + to;
    stepper.step(delta);
    delay(200);   // mechanical settling; coils stay ON (DC hold, not EMI)
    current_position = next;
    Serial.println("READY");

  } else if (cmd == "RESET") {
    // Single-shot backward — used for emergency recovery only.
    // Normal backward path uses individual STEP commands from Python.
    if (current_position > 0) {
      stepper.step(-stepsAtPosition(current_position));
      current_position = 0;
    }
    delay(200);
    deenergizeCoils();
    Serial.println("READY");

  } else if (cmd == "STATUS") {
    Serial.print("POS ");
    Serial.println(current_position);

  } else if (cmd.startsWith("STEP ")) {
    // Arbitrary steps for calibration jogging and step-by-step backward reset.
    // Does NOT update current_position (caller manages position tracking).
    long n = cmd.substring(5).toInt();
    stepper.step(n);
    delay(200);
    Serial.println("READY");

  } else if (cmd == "ZERO") {
    current_position = 0;
    Serial.println("READY");

  } else if (cmd == "DEENERGIZE") {
    deenergizeCoils();
    Serial.println("READY");
  }
}
