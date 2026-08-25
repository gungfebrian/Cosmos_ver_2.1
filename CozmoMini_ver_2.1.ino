/* -------------------------------------------------
   Desk Pet Robot — ESP32-C3 Super Mini
   RoboEyes face + autonomous "drives" behavior engine
   ---------------------------------------------------
   HARDWARE (confirmed working):
     OLED (I2C, drive as SSD1306) -> SDA GPIO4, SCL GPIO5
     Servo LEFT  (360° cont.)     -> GPIO 10
     Servo RIGHT (360° cont.)     -> GPIO 6
     Capacitive touch sensor      -> GPIO 3
   LIBRARIES: FluxGarage RoboEyes, Adafruit GFX, Adafruit SSD1306, ESP32Servo
   Board: "ESP32C3 Dev Module"

   ---------------------------------------------------
   HOW IT "THINKS"
   Four hidden drives update continuously and DRIVE the behavior:
     energy    : drains while awake, only recharges by SLEEPING
                 -> it gets tired on its own and WAKES on its own
     mood       : rises with petting, falls with anger, drifts to neutral
     annoyance : each poke adds to it, decays over time
     boredom    : climbs when ignored -> it comes looking for attention

   GESTURES
     single tap            -> ALERT (surprised boop + hop)
     hold 0.4-2.5s         -> HAPPY (laugh + wiggle)
     hold > 2.5s           -> LOVE  (blissful squint)
     3 rapid taps          -> ANNOYED
     ~3 taps x3 in 5s (8+) -> ANGRY / rage, then sulks (GRUMPY)
     touch while asleep    -> STARTLED

   AUTONOMOUS (no touch needed)
     wakes itself once rested (or after max sleep)
     gets bored -> ATTENTION (hops/spins to get noticed) -> gives up
     high mood + energy    -> PLAYFUL bursts
     too much spinning     -> DIZZY
     runs out of energy    -> SLEEPY -> ASLEEP
-------------------------------------------------*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <FluxGarage_RoboEyes.h>
#include <ESP32Servo.h>
#include "ml/behavior_model.h"

/* -------- DISPLAY -------- */
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_ADDR     0x3C
#define I2C_SDA 4
#define I2C_SCL 5
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
RoboEyes<Adafruit_SSD1306> roboEyes(display);

/* -------- SERVOS -------- */
#define SERVO_L 10
#define SERVO_R 6
Servo servoL, servoR;
const int   SERVO_STOP = 1500;
const int   LEFT_TRIM  = 0;
const int   RIGHT_TRIM = 0;
const int   SPEED_US   = 220;
const int   LEFT_DIR   = +1;
const int   RIGHT_DIR  = -1;
const float MOTOR_SPEED = 0.60;
const float TURN_SPEED  = 0.60;

/* -------- TOUCH -------- */
#define TOUCH_PIN 3

/* -------- MODES -------- */
#define MODE_IDLE     0
#define MODE_ROAM     1
#define MODE_ALERT    2
#define MODE_HAPPY    3
#define MODE_LOVE     4
#define MODE_ANNOYED  5
#define MODE_ANGRY    6
#define MODE_GRUMPY   7
#define MODE_ATTENTION 8
#define MODE_SLEEPY   9
#define MODE_ASLEEP   10
#define MODE_WAKING   11
#define MODE_STARTLED 12
#define MODE_PLAYFUL  13
#define MODE_DIZZY    14

int mode = MODE_IDLE;
unsigned long modeSince = 0;
unsigned long modeUntil = 0;

/* -------- DRIVES -------- */
float energy    = 80;
float mood      = 55;
float annoyance = 0;
float boredom   = 0;
unsigned long lastTick = 0;
unsigned long lastInteraction = 0;

/* thresholds (tune to taste) */
const float ANNOY_TH      = 40;     // annoyance -> ANNOYED
const float RAGE_TH       = 75;     // annoyance -> ANGRY
const float ENERGY_SLEEPY = 18;     // energy    -> SLEEPY
const float BORED_TH      = 70;     // boredom   -> ATTENTION
const float WAKE_ENERGY   = 90;     // rested enough to self-wake
const unsigned long MAX_SLEEP_MS = 30000;  // wake anyway after this

/* -------- TAP / GESTURE -------- */
bool touchActive = false;
unsigned long touchStart = 0;
int  tapCount = 0;                   // taps in current burst
unsigned long lastTapTime = 0;
#define TAP_HIST 10
unsigned long tapTimes[TAP_HIST] = {0};
int  tapWrite = 0;
const float TAP_ANNOY = 15;          // annoyance added per poke

/* -------- TIMERS -------- */
unsigned long lastMove = 0, moveInterval = 0, roamUntil = 0;
unsigned long nextRoamExcursion = 0, nextQuirk = 0;
unsigned long attnNextHop = 0, angryNextShake = 0, angryNextMove = 0;
unsigned long playNext = 0, nextPlay = 0;
unsigned long dizzyNextConfused = 0;
const float MODEL_CONFIDENCE = 0.78;
const unsigned long MODEL_DECISION_INTERVAL_MS = 1000;
unsigned long lastModelDecision = 0;

/* -------- DIZZY -------- */
int spinLoad = 0;
const int DIZZY_THRESHOLD = 12;
unsigned long lastSpinDecay = 0;

/* -------- SERVO ACTION SCHEDULER -------- */
enum Act { ACT_STOP, ACT_FWD, ACT_TURNL, ACT_TURNR, ACT_WIGGLE, ACT_SPIN, ACT_BACK, ACT_SHUFFLE,
           ACT_CHARGE, ACT_RSPIN };   // CHARGE/RSPIN = full-speed angry moves
Act curAct = ACT_STOP;
unsigned long actUntil = 0, actPhase = 0;
bool actPhaseState = false;

/* -------- HELPERS -------- */
float clampf(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }

void drive(float l, float r) {
  int lus = SERVO_STOP + LEFT_TRIM  + (int)(LEFT_DIR  * l * SPEED_US);
  int rus = SERVO_STOP + RIGHT_TRIM + (int)(RIGHT_DIR * r * SPEED_US);
  servoL.writeMicroseconds(constrain(lus, 1000, 2000));
  servoR.writeMicroseconds(constrain(rus, 1000, 2000));
}
void motorStop()    { drive(0, 0); }
void motorForward() { drive(MOTOR_SPEED,  MOTOR_SPEED); }
void rotateCW()     { drive(TURN_SPEED,  -TURN_SPEED); }
void rotateCCW()    { drive(-TURN_SPEED,  TURN_SPEED); }

void startAct(Act a, unsigned long dur) {
  curAct = a; actUntil = millis() + dur; actPhase = millis(); actPhaseState = false;
}

void updateActions() {
  unsigned long now = millis();
  if (curAct != ACT_STOP && actUntil != 0 && now > actUntil) { curAct = ACT_STOP; motorStop(); return; }
  switch (curAct) {
    case ACT_STOP:  motorStop(); break;
    case ACT_FWD:   motorForward(); break;
    case ACT_TURNL: rotateCCW(); break;
    case ACT_TURNR: rotateCW(); break;
    case ACT_BACK:  drive(-MOTOR_SPEED, -MOTOR_SPEED); break;
    case ACT_SPIN:  rotateCW(); break;
    case ACT_CHARGE: drive(1.0, 1.0); break;    // full-speed forward stomp
    case ACT_RSPIN:  drive(1.0, -1.0); break;   // full-speed furious spin
    case ACT_WIGGLE:
      if (now - actPhase > 140) { actPhase = now; actPhaseState = !actPhaseState; }
      actPhaseState ? rotateCW() : rotateCCW(); break;
    case ACT_SHUFFLE:
      if (now - actPhase > 90) { actPhase = now; actPhaseState = !actPhaseState; }
      actPhaseState ? drive(0.85, 0.85) : drive(-0.85, -0.85); break;
  }
}

int tapsInWindow(unsigned long now, unsigned long win) {
  int c = 0;
  for (int i = 0; i < TAP_HIST; i++) if (tapTimes[i] != 0 && now - tapTimes[i] <= win) c++;
  return c;
}
void pokeRegister(unsigned long now) {
  tapTimes[tapWrite] = now; tapWrite = (tapWrite + 1) % TAP_HIST;
  annoyance = clampf(annoyance + TAP_ANNOY, 0, 100);
}

/* -------- ENTER A MODE (sets full expression once) -------- */
void enterMode(int m) {
  mode = m; modeSince = millis(); modeUntil = 0;
  curAct = ACT_STOP; actUntil = 0; motorStop();

  roboEyes.setHFlicker(false);
  roboEyes.setVFlicker(false);
  roboEyes.setSweat(false);
  roboEyes.setCyclops(false);

  switch (m) {

    case MODE_IDLE:
      roboEyes.setMood(DEFAULT); roboEyes.setCuriosity(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setIdleMode(true,3,3); roboEyes.setAutoblinker(true,4,3);
      roboEyes.setPosition(DEFAULT);
      break;

    case MODE_ROAM:
      roboEyes.setMood(DEFAULT); roboEyes.setCuriosity(true);
      roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setIdleMode(true,2,2); roboEyes.setAutoblinker(true,3,3);
      roamUntil = millis() + random(6000,12000); lastMove = 0; moveInterval = 0;
      break;

    case MODE_ALERT:                                  // friendly boop
      roboEyes.setMood(DEFAULT); roboEyes.setIdleMode(false);
      roboEyes.setWidth(40,40); roboEyes.setHeight(40,40);
      roboEyes.setPosition(N); roboEyes.setAutoblinker(true,2,1);
      roboEyes.blink(); startAct(ACT_SHUFFLE,250);
      modeUntil = millis()+1400;
      break;

    case MODE_HAPPY:                                  // gentle pet
      roboEyes.setMood(HAPPY); roboEyes.setIdleMode(false);
      roboEyes.setPosition(DEFAULT); roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setAutoblinker(true,2,1); roboEyes.anim_laugh();
      startAct(ACT_WIGGLE,700);
      break;

    case MODE_LOVE:                                   // long content pet
      roboEyes.setMood(HAPPY); roboEyes.setIdleMode(false);
      roboEyes.setWidth(38,38); roboEyes.setHeight(16,16);
      roboEyes.setPosition(N); roboEyes.setAutoblinker(true,3,2);
      break;

    case MODE_ANNOYED:                                // mild irritation
      annoyance = clampf(annoyance-20,0,100);         // venting
      roboEyes.setMood(ANGRY); roboEyes.setIdleMode(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setPosition(DEFAULT); roboEyes.setSweat(true);
      roboEyes.setHFlicker(true,3); roboEyes.setAutoblinker(true,2,1);
      startAct(ACT_BACK,500);
      modeUntil = millis()+2200;
      break;

    case MODE_ANGRY:                                  // full rage
      annoyance = 0; mood = clampf(mood-25,0,100);
      roboEyes.setMood(ANGRY); roboEyes.setIdleMode(false);
      roboEyes.setWidth(38,38); roboEyes.setHeight(38,38);
      roboEyes.setPosition(DEFAULT); roboEyes.setSweat(true);
      roboEyes.setHFlicker(true,5); roboEyes.setAutoblinker(true,2,1);
      roboEyes.anim_confused();
      startAct(ACT_CHARGE,500);                       // storms off immediately
      angryNextShake = millis()+400; angryNextMove = millis()+500;
      modeUntil = millis()+5500;                      // long tantrum
      break;

    case MODE_GRUMPY:                                 // sulk after rage
      roboEyes.setMood(TIRED); roboEyes.setIdleMode(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(28,28);
      roboEyes.setPosition(SW); roboEyes.setAutoblinker(true,3,2);
      startAct(ACT_TURNR,400);                        // turn away
      modeUntil = millis()+4000;
      break;

    case MODE_ATTENTION:                              // bored, wants you
      roboEyes.setMood(DEFAULT); roboEyes.setCuriosity(true); roboEyes.setIdleMode(false);
      roboEyes.setWidth(38,38); roboEyes.setHeight(40,40);
      roboEyes.setAutoblinker(true,2,1); roboEyes.setPosition(N);
      attnNextHop = millis()+600;
      break;

    case MODE_SLEEPY:
      roboEyes.setMood(TIRED); roboEyes.setIdleMode(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(22,22);
      roboEyes.setPosition(S); roboEyes.setAutoblinker(true,2,2);
      break;

    case MODE_ASLEEP:
      roboEyes.setMood(TIRED); roboEyes.setIdleMode(false);
      roboEyes.setAutoblinker(false); roboEyes.setPosition(DEFAULT);
      roboEyes.close();
      break;

    case MODE_WAKING:                                 // yawn / stretch
      roboEyes.setMood(TIRED); roboEyes.setIdleMode(false);
      roboEyes.setAutoblinker(true,2,1);
      roboEyes.setWidth(40,40); roboEyes.setHeight(40,40);
      roboEyes.setPosition(N); roboEyes.open();
      startAct(ACT_SHUFFLE,300); boredom = 30;
      modeUntil = millis()+1600;
      break;

    case MODE_STARTLED:                               // woken by touch
      roboEyes.setMood(DEFAULT); roboEyes.setIdleMode(false);
      roboEyes.setWidth(44,44); roboEyes.setHeight(44,44);
      roboEyes.open(); roboEyes.setSweat(true);
      roboEyes.setHFlicker(true,4); roboEyes.setAutoblinker(true,1,1);
      startAct(ACT_SHUFFLE,400);
      modeUntil = millis()+1200;
      break;

    case MODE_PLAYFUL:
      roboEyes.setMood(HAPPY); roboEyes.setCuriosity(true); roboEyes.setIdleMode(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setAutoblinker(true,2,1); roboEyes.anim_laugh();
      playNext = millis()+300;
      modeUntil = millis()+5000;
      break;

    case MODE_DIZZY:
      roboEyes.setMood(DEFAULT); roboEyes.setIdleMode(false);
      roboEyes.setWidth(36,36); roboEyes.setHeight(36,36);
      roboEyes.setVFlicker(true,3); roboEyes.setAutoblinker(true,2,1);
      roboEyes.anim_confused(); dizzyNextConfused = millis()+700;
      modeUntil = millis()+2600;
      break;
  }
}

/* -------- RESOLVE A TIMED TRANSIENT -------- */
void resolveTransient() {
  modeUntil = 0;
  switch (mode) {
    case MODE_STARTLED: enterMode(MODE_ALERT); break;
    case MODE_HAPPY:
    case MODE_LOVE:     enterMode(MODE_ROAM); break;
    case MODE_ANGRY:    enterMode(MODE_GRUMPY); break;
    case MODE_PLAYFUL:  enterMode(MODE_IDLE); nextPlay = millis()+random(15000,30000); break;
    case MODE_ALERT:
    case MODE_ANNOYED:
    case MODE_GRUMPY:
    case MODE_WAKING:
    case MODE_DIZZY:
    default:
      enterMode(MODE_IDLE);
      nextRoamExcursion = millis()+random(6000,14000);
      break;
  }
}

/* -------- TOUCH + GESTURES -------- */
void updateTouch() {
  bool touched = digitalRead(TOUCH_PIN);
  unsigned long now = millis();

  if (touched && !touchActive) {                      // press
    touchActive = true; touchStart = now; lastInteraction = now;
    if (mode == MODE_ASLEEP || mode == MODE_SLEEPY) enterMode(MODE_STARTLED);
  }

  if (touched && touchActive) {                       // holding
    unsigned long held = now - touchStart; lastInteraction = now;
    if (mode != MODE_STARTLED) {
      if (held > 2500)      { if (mode != MODE_LOVE)  enterMode(MODE_LOVE); }
      else if (held > 400)  { if (mode != MODE_HAPPY && mode != MODE_LOVE) enterMode(MODE_HAPPY); }
    }
  }

  if (!touched && touchActive) {                      // release
    touchActive = false; lastInteraction = now;
    unsigned long held = now - touchStart;
    if (held < 400) { tapCount++; lastTapTime = now; pokeRegister(now); }
    else if (mode == MODE_HAPPY || mode == MODE_LOVE) modeUntil = now + 1200;
  }
}

/* -------- DRIVES UPDATE -------- */
void updateDrives() {
  unsigned long now = millis();
  float dt = (now - lastTick) / 1000.0;
  if (dt <= 0) return;
  lastTick = now;
  if (dt > 0.5) dt = 0.5;

  // energy: only sleep recharges
  float de = -0.5;
  if      (mode == MODE_ASLEEP) de = 7;
  else if (mode == MODE_SLEEPY) de = 0.5;
  else if (mode == MODE_ROAM || mode == MODE_PLAYFUL || mode == MODE_ANGRY ||
           mode == MODE_ATTENTION || mode == MODE_ANNOYED) de = -3;
  else if (mode == MODE_IDLE) de = -1.0;
  energy = clampf(energy + de*dt, 0, 100);

  // annoyance decays
  annoyance = clampf(annoyance - 6*dt, 0, 100);

  // mood drifts toward 50, shaped by mode
  float dm = (50 - mood) * 0.02;
  if (mode == MODE_HAPPY || mode == MODE_LOVE || mode == MODE_PLAYFUL) dm += 8;
  if (mode == MODE_ANGRY || mode == MODE_ANNOYED || mode == MODE_GRUMPY) dm -= 6;
  mood = clampf(mood + dm*dt, 0, 100);

  // boredom
  if (touchActive || (now - lastInteraction) < 2500) boredom = 0;
  else if (mode == MODE_ASLEEP) boredom = 0;
  else boredom = clampf(boredom + 2*dt, 0, 100);
}

/* -------- ROAM MOVEMENT -------- */
void driveRoam(unsigned long now) {
  if (curAct != ACT_STOP) return;
  if (now - lastMove < moveInterval) return;
  lastMove = now; moveInterval = random(1500,3500);
  switch (random(0,5)) {
    case 0: startAct(ACT_FWD,   random(800,1600)); break;
    case 1: startAct(ACT_TURNL, random(300,700)); spinLoad += 2; break;
    case 2: startAct(ACT_TURNR, random(300,700)); spinLoad += 2; break;
    case 3: motorStop(); break;
    case 4: startAct(ACT_SPIN,  random(500,1000)); spinLoad += 3; break;
  }
}

/* -------- BRAIN -------- */
void updateBrain() {
  unsigned long now = millis();

  // settle a tap burst -> boop / annoyed / angry
  if (tapCount > 0 && now - lastTapTime > 450) {
    int recent = tapsInWindow(now, 5000);
    if (annoyance >= RAGE_TH || recent >= 8)        enterMode(MODE_ANGRY);
    else if (annoyance >= ANNOY_TH || recent >= 3)  enterMode(MODE_ANNOYED);
    else                                            enterMode(MODE_ALERT);
    tapCount = 0;
  }

  // timed transient expiry
  if (modeUntil != 0 && now > modeUntil) resolveTransient();

  // self-wake from sleep
  if (mode == MODE_ASLEEP) {
    if (energy >= WAKE_ENERGY || (now - modeSince) > MAX_SLEEP_MS) enterMode(MODE_WAKING);
  }
  // drowsy -> asleep
  if (mode == MODE_SLEEPY && (now - modeSince) > 5000) enterMode(MODE_ASLEEP);

  // autonomous decisions from a free/interruptible mode
  bool freeMode = (mode == MODE_IDLE || mode == MODE_ROAM || mode == MODE_ATTENTION);
  if (freeMode) {
    if (annoyance >= RAGE_TH)               enterMode(MODE_ANGRY);
    else if (annoyance >= ANNOY_TH)         enterMode(MODE_ANNOYED);
    else if (energy <= ENERGY_SLEEPY)       enterMode(MODE_SLEEPY);
    else if (mode != MODE_ATTENTION && boredom >= BORED_TH) enterMode(MODE_ATTENTION);
    else if (mood >= 82 && energy >= 55 && now > nextPlay)  enterMode(MODE_PLAYFUL);
  }

  // TinyML advice is deliberately downstream of the safety-first thresholds.
  // The confidence gate and one-second cadence keep low-confidence predictions
  // from making the behavior jittery.
  if (freeMode && (mode == MODE_IDLE || mode == MODE_ROAM || mode == MODE_ATTENTION) &&
      now - lastModelDecision >= MODEL_DECISION_INTERVAL_MS) {
    lastModelDecision = now;
    const cozmo_ml::Features features = {{
      clampf(energy / 100.0, 0, 1),
      clampf(mood / 100.0, 0, 1),
      clampf(annoyance / 100.0, 0, 1),
      clampf(boredom / 100.0, 0, 1),
      touchActive ? 1.0f : 0.0f,
      clampf((now - lastInteraction) / 5000.0, 0, 1),
      clampf(tapsInWindow(now, 5000) / 8.0, 0, 1)
    }};
    const cozmo_ml::Prediction prediction = cozmo_ml::predict(features);
    if (prediction.confidence >= MODEL_CONFIDENCE) {
      switch (prediction.intent) {
        case cozmo_ml::Intent::INTENT_ANGRY:
          if (mode != MODE_ANGRY) enterMode(MODE_ANGRY);
          break;
        case cozmo_ml::Intent::INTENT_ANNOYED:
          if (mode != MODE_ANNOYED) enterMode(MODE_ANNOYED);
          break;
        case cozmo_ml::Intent::INTENT_SLEEPY:
          if (mode != MODE_SLEEPY) enterMode(MODE_SLEEPY);
          break;
        case cozmo_ml::Intent::INTENT_ATTENTION:
          if (mode != MODE_ATTENTION) enterMode(MODE_ATTENTION);
          break;
        case cozmo_ml::Intent::INTENT_PLAYFUL:
          if (mode != MODE_PLAYFUL) enterMode(MODE_PLAYFUL);
          break;
        case cozmo_ml::Intent::INTENT_IDLE:
        default:
          break;
      }
    }
  }

  // ANGRY: storm around like an angry child + furious head shakes
  if (mode == MODE_ANGRY) {
    if (now > angryNextShake) { roboEyes.anim_confused(); angryNextShake = now + 400; }
    if (curAct == ACT_STOP && now > angryNextMove) {
      angryNextMove = now + random(250, 650);         // erratic, restless
      switch (random(0,4)) {
        case 0: startAct(ACT_CHARGE, random(400,800)); break;   // stomp forward
        case 1: startAct(ACT_RSPIN,  random(400,800)); spinLoad += 3; break; // furious spin
        case 2: startAct(ACT_BACK,   random(300,500)); break;   // storm backward
        case 3: startAct(ACT_SHUFFLE, 300); break;              // stamp feet
      }
    }
  }

  // ATTENTION: try to get noticed, then give up
  if (mode == MODE_ATTENTION) {
    if (now > attnNextHop) {
      attnNextHop = now + random(1200,2500);
      switch (random(0,3)) {
        case 0: startAct(ACT_SHUFFLE,300); break;
        case 1: startAct(ACT_WIGGLE,400);  break;
        case 2: startAct(ACT_SPIN,500); roboEyes.setPosition(random(0,2)?E:W); break;
      }
      roboEyes.blink();
    }
    if ((now - modeSince) > 12000) { boredom = 25; enterMode(MODE_IDLE); }
  }

  // PLAYFUL: energetic bursts
  if (mode == MODE_PLAYFUL && curAct == ACT_STOP && now > playNext) {
    playNext = now + random(500,1000);
    switch (random(0,3)) {
      case 0: startAct(ACT_SPIN, random(500,900)); spinLoad += 2; break;
      case 1: startAct(ACT_FWD,  random(500,900)); break;
      case 2: startAct(ACT_WIGGLE,500); break;
    }
  }

  // DIZZY: keep reeling
  if (mode == MODE_DIZZY && now > dizzyNextConfused) {
    roboEyes.anim_confused(); dizzyNextConfused = now + 700;
  }

  // spontaneous roaming from idle
  if (mode == MODE_IDLE && now > nextRoamExcursion) enterMode(MODE_ROAM);

  // random idle quirks
  if (mode == MODE_IDLE && now > nextQuirk) {
    nextQuirk = now + random(5000,12000);
    switch (random(0,3)) {
      case 0: roboEyes.blink(); break;
      case 1: roboEyes.anim_confused(); break;
      case 2: startAct(ACT_WIGGLE,300); break;
    }
  }

  // roam driving + exit
  if (mode == MODE_ROAM) {
    driveRoam(now);
    if (now > roamUntil && curAct == ACT_STOP) {
      enterMode(MODE_IDLE); nextRoamExcursion = now + random(8000,18000);
    }
  }

  // spin load decay + dizzy trigger
  if (now - lastSpinDecay > 1000) { lastSpinDecay = now; if (spinLoad > 0) spinLoad--; }
  if (spinLoad >= DIZZY_THRESHOLD && (mode == MODE_ROAM || mode == MODE_PLAYFUL)) {
    spinLoad = 0; enterMode(MODE_DIZZY);
  }
}

/* -------- SETUP -------- */
void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR))
    Serial.println("SSD1306 not found - check wiring / address");

  pinMode(TOUCH_PIN, INPUT_PULLDOWN);

  ESP32PWM::allocateTimer(0); ESP32PWM::allocateTimer(1);
  servoL.setPeriodHertz(50); servoR.setPeriodHertz(50);
  servoL.attach(SERVO_L,500,2500); servoR.attach(SERVO_R,500,2500);
  motorStop();

  randomSeed(esp_random());
  roboEyes.begin(SCREEN_WIDTH, SCREEN_HEIGHT, 60);
  roboEyes.setBorderradius(8,8);
  roboEyes.setSpacebetween(10);

  unsigned long now = millis();
  lastTick = now; lastInteraction = now; lastSpinDecay = now;
  nextRoamExcursion = now + random(6000,12000);
  nextQuirk = now + random(5000,10000);
  nextPlay  = now + random(10000,20000);

  enterMode(MODE_IDLE);
}

/* -------- LOOP -------- */
void loop() {
  updateTouch();
  updateDrives();
  updateBrain();
  updateActions();
  roboEyes.update();
}
