#include <Servo.h>

// CONFIGURATION
const int SERVO_PIN = 9;   // Servo Signal Wire (Orange/Yellow)
const int SPEAKER_PIN = 3; // Buzzer/Speaker Pin
const int STEP_SIZE = 5;   // How many degrees to move per 'M' command (used for step mode)

Servo myServo;
int currentAngle = 0;

void setup() {
  Serial.begin(9600);
  
  // Attach Servo and move to default position (45°)
  myServo.attach(SERVO_PIN);
  myServo.write(45);
  currentAngle = 45;
  
  // Give it time to get there
  delay(1000);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // --- COMMAND: GOTO (G) - Move directly to angle ---
    // Format: G followed by 3-digit angle (e.g., G045 for 45°)
    if (cmd == 'G') {
      // Read the next 3 characters as angle
      delay(10); // Wait for remaining chars
      String angleStr = "";
      while (Serial.available() > 0 && angleStr.length() < 3) {
        char c = Serial.read();
        if (c >= '0' && c <= '9') {
          angleStr += c;
        }
      }
      
      int targetAngle = angleStr.toInt();
      if (targetAngle >= 0 && targetAngle <= 180) {
        // Smooth movement
        if (targetAngle > currentAngle) {
          for (int a = currentAngle; a <= targetAngle; a++) {
            myServo.write(a);
            delay(10); // Smooth movement speed
          }
        } else {
          for (int a = currentAngle; a >= targetAngle; a--) {
            myServo.write(a);
            delay(10);
          }
        }
        currentAngle = targetAngle;
        delay(200); // Settling time
        Serial.println("READY");
      } else {
        Serial.println("ERROR: Angle out of range");
      }
    }

    // --- COMMAND: MOVE STEP (M) ---
    if (cmd == 'M') { 
       if (currentAngle + STEP_SIZE <= 180) {
         // Smooth step movement
         int targetAngle = currentAngle + STEP_SIZE;
         for (int a = currentAngle; a <= targetAngle; a++) {
           myServo.write(a);
           delay(10);
         }
         currentAngle = targetAngle;
         delay(200);
         Serial.println("READY");
       } 
       else {
         Serial.println("READY"); 
       }
    }
    
    // --- COMMAND: PLAY (P) ---
    if (cmd == 'P') { 
       // Play chirp sweep (500Hz to 3kHz) - best for GCC-PHAT
       for (int i=500; i<3000; i+=100) {
         tone(SPEAKER_PIN, i, 10);
         delay(5); 
       }
       noTone(SPEAKER_PIN);
       Serial.println("DONE");
    }
    
    // --- COMMAND: RESET (R) ---
    if (cmd == 'R') {
      // Smooth return to 0
      for (int a = currentAngle; a >= 0; a--) {
        myServo.write(a);
        delay(10);
      }
      currentAngle = 0;
      delay(300);
      Serial.println("RESET_DONE");
    }
    
    // --- COMMAND: QUERY ANGLE (Q) ---
    if (cmd == 'Q') {
      Serial.println(currentAngle);
    }
  }
}