#include <Servo.h>

// CONFIGURATION
const int SERVO_PIN = 9;   // Servo Signal Wire (Orange/Yellow)
const int SPEAKER_PIN = 3; // Buzzer/Speaker Pin
const int STEP_SIZE = 5;   // How many degrees to move per 'M' command

Servo myServo;
int currentAngle = 0;

void setup() {
  Serial.begin(9600);
  
  // Attach Servo and move to start position
  myServo.attach(SERVO_PIN);
  myServo.write(0);
  
  // Give it time to get there
  delay(1000);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // --- COMMAND: MOVE ---
    if (cmd == 'M') { 
       // Check if we have room to move
       if (currentAngle + STEP_SIZE <= 180) {
         currentAngle += STEP_SIZE;
         myServo.write(currentAngle);
         
         // Servos are fast but jerky. Wait for mechanical settling.
         delay(600); 
         Serial.println("READY"); // Tell PC we are stable
       } 
       else {
         // If we hit the limit, just stay there and report ready
         Serial.println("READY"); 
       }
    }
    
    // --- COMMAND: PLAY ---
    if (cmd == 'P') { 
       // Play the "Chirp" sweep (500Hz to 3kHz)
       // This broadband noise is best for GCC-PHAT
       for (int i=500; i<3000; i+=100) {
         tone(SPEAKER_PIN, i, 10);
         delay(5); 
       }
       noTone(SPEAKER_PIN); // Silence
       Serial.println("DONE");
    }
    
    // --- COMMAND: RESET (Optional) ---
    if (cmd == 'R') {
      currentAngle = 0;
      myServo.write(0);
      delay(1000);
      Serial.println("RESET_DONE");
    }
  }
}