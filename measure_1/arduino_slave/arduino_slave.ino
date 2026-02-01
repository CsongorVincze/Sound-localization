#include <Servo.h>

// CONFIGURATION
const int SERVO_PIN = 9;
const int SPEAKER_PIN = 3;

Servo myServo;
int currentAngle = 45;

void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(45);
  delay(1000);
  Serial.println("READY");
}

// Play a pure tone at specified frequency for duration (ms)
void playTone(int freq, int duration) {
  tone(SPEAKER_PIN, freq, duration);
  delay(duration);
  noTone(SPEAKER_PIN);
}

// Play white noise simulation (random frequencies)
void playNoise(int duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    int freq = random(200, 8000);
    tone(SPEAKER_PIN, freq, 2);
    delay(2);
  }
  noTone(SPEAKER_PIN);
}

// Play chirp sweep (broadband)
void playChirp() {
  for (int i = 500; i < 3000; i += 100) {
    tone(SPEAKER_PIN, i, 10);
    delay(5);
  }
  noTone(SPEAKER_PIN);
}

// Play voice-like sound (formant simulation)
void playVoice(int duration) {
  // Simulate vowel formants with frequency modulation
  unsigned long startTime = millis();
  int baseFreq = 150; // F0 (fundamental)
  while (millis() - startTime < duration) {
    // Modulate between formant frequencies
    int t = millis() - startTime;
    int freq = baseFreq + (t % 200) * 2; // Slight pitch variation
    tone(SPEAKER_PIN, freq, 10);
    delay(10);
  }
  noTone(SPEAKER_PIN);
}

// Play click (impulsive)
void playClick() {
  tone(SPEAKER_PIN, 4000, 5);
  delay(5);
  noTone(SPEAKER_PIN);
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
        delay(200);
        Serial.println("READY");
      }
    }

    // SOUND commands: S followed by type code
    // S0 = Chirp (broadband)
    // S1 = 500Hz pure tone
    // S2 = 1000Hz pure tone
    // S3 = 2000Hz pure tone
    // S4 = 4000Hz pure tone
    // S5 = White noise
    // S6 = Voice-like
    // S7 = Click/impulse
    if (cmd == 'S') {
      delay(10);
      if (Serial.available() > 0) {
        char type = Serial.read();
        switch (type) {
          case '0': playChirp(); break;
          case '1': playTone(500, 200); break;
          case '2': playTone(1000, 200); break;
          case '3': playTone(2000, 200); break;
          case '4': playTone(4000, 200); break;
          case '5': playNoise(300); break;
          case '6': playVoice(300); break;
          case '7': playClick(); break;
          default: playChirp(); break;
        }
        Serial.println("DONE");
      }
    }

    // Legacy play command (chirp)
    if (cmd == 'P') {
      playChirp();
      Serial.println("DONE");
    }

    // Reset to 0
    if (cmd == 'R') {
      for (int a = currentAngle; a >= 0; a--) {
        myServo.write(a);
        delay(10);
      }
      currentAngle = 0;
      delay(300);
      Serial.println("RESET_DONE");
    }

    // Query angle
    if (cmd == 'Q') {
      Serial.println(currentAngle);
    }
  }
}