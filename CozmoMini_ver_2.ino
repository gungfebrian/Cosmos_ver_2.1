#include <ESP32Servo.h>

// Define Servo Objects
Servo leftLeg0;
Servo leftLeg1;
Servo servo2;
Servo servo3;

// Pin Assignments
const int leftLeg0Pin = 0;
const int leftLeg1Pin = 1;
const int servo2Pin   = 2;
const int servo3Pin   = 3;

void setup() {
  Serial.begin(115200);
  
  // Allow allocation of all timers
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  // Set standard 50Hz frequency for RC servos
  leftLeg0.setPeriodHertz(50);
  leftLeg1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo3.setPeriodHertz(50);

  // Attach the servos to their respective pins
  leftLeg0.attach(leftLeg0Pin, 500, 2400); // standard min/max pulse widths
  leftLeg1.attach(leftLeg1Pin, 500, 2400);
  servo2.attach(servo2Pin, 500, 2400);
  servo3.attach(servo3Pin, 500, 2400);
}

void loop() {
  // --- SWEEPING TO 0 DEGREES ---
  Serial.println("Moving all servos to 0 degrees...");
  leftLeg0.write(0);
  leftLeg1.write(0);
  servo2.write(0);
  servo3.write(0);
  delay(2000); // Wait 2 seconds to see the position

  // --- SWEEPING TO THE REST / MID POINT (90 DEGREES) ---
  Serial.println("Moving all servos to 90 degrees (Standard Rest/Center Position)...");
  leftLeg0.write(90);
  leftLeg1.write(90);
  servo2.write(90);
  servo3.write(90);
  delay(2000);

  // --- SWEEPING TO 180 DEGREES ---
  Serial.println("Moving all servos to 180 degrees...");
  leftLeg0.write(180);
  leftLeg1.write(180);
  servo2.write(180);
  servo3.write(180);
  delay(2000);
}