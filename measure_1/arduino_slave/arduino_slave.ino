#include <Servo.h>

// CONFIGURATION
const int SERVO_PIN = 9;

// Settling delay after movement (ms).
// Increase this if the external speaker mount causes vibrations.
const int SETTLE_DELAY_MS = 500;

Servo myServo;
int currentAngle = 45;

void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(45);
  delay(1000);
  Serial.println("READY");
}


void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // GOTO command: G### (e.g., G045)
    if (cmd == 'G') {
      delay(10);
      String angleStr = "";
      while (Serial.available() > 0 && angleStr.length() < 3) {
        char c = Serial.read();
        if (c >= '0' && c <= '9') angleStr += c;
      }
      int targetAngle = angleStr.toInt();
      if (targetAngle >= 0 && targetAngle <= 180) {
        // Smooth movement
        int step = (targetAngle > currentAngle) ? 1 : -1;
        for (int a = currentAngle; a != targetAngle; a += step) {
          myServo.write(a);
          delay(10);
        }
        myServo.write(targetAngle);
        currentAngle = targetAngle;
        // Wait for mechanical settling (important with external speaker mount)
        delay(SETTLE_DELAY_MS);
        Serial.println("READY");
      }
    }


    // Reset to 0
    if (cmd == 'R') {
      for (int a = currentAngle; a >= 0; a--) {
        myServo.write(a);
        delay(10);
      }
      currentAngle = 0;
      delay(SETTLE_DELAY_MS);
      Serial.println("RESET_DONE");
    }

    // Query angle
    if (cmd == 'Q') {
      Serial.println(currentAngle);
    }
  }
}