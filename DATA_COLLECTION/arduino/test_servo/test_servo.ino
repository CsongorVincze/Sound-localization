/*
  MG996R + PCA9685 basic test.
  Sweeps from 0° to 180° then back, repeating.

  Wiring:
    PCA9685 SDA → A4,  SCL → A5,  VCC → 5V,  GND → GND
    PCA9685 V+  → separate 5-6V servo power supply
    Servo       → PCA9685 channel 0

  Library: Adafruit PWM Servo Driver (install via Library Manager)

  If servo does not reach true 0°/180°, adjust SERVOMIN/SERVOMAX.
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVO_CH   0
#define SERVOMIN   150    // pulse count ≈ 0°   — increase if servo overshoots 0°
#define SERVOMAX   600    // pulse count ≈ 180° — decrease if servo overshoots 180°

int angleToPulse(int deg) {
  return SERVOMIN + (long)(SERVOMAX - SERVOMIN) * deg / 180;
}

void setAngle(int deg) {
  pwm.setPWM(SERVO_CH, 0, angleToPulse(deg));
}

void setup() {
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);
  setAngle(0);
  delay(1000);
}

void loop() {
  for (int deg = 0; deg <= 180; deg += 5) {
    setAngle(deg);
    delay(300);
  }
  delay(500);
  for (int deg = 180; deg >= 0; deg -= 5) {
    setAngle(deg);
    delay(300);
  }
  delay(500);
}
