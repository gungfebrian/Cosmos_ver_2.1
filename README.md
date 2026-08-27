# CozmoMini

A palm-sized desk pet robot built on an ESP32-C3. It has animated eyes on a small OLED screen, drives around on two wheels, and reacts to touch — but the part I'm most proud of is that it doesn't just follow a script. It runs a small "mood engine" of four hidden drives (energy, mood, annoyance, boredom) that update continuously and decide what the robot does next. Left alone it gets tired and falls asleep. Ignored too long, it comes looking for attention. Poke it too many times and it throws a tantrum.

No app, no Wi-Fi, no cloud. Everything runs on the microcontroller.

> **Status:** working prototype. `CozmoMini_ver_2.1.ino` is the current firmware.

<!-- Add a photo or GIF of the robot here — it makes a huge difference:
![CozmoMini](docs/demo.gif)
-->

---

## What it does

- **Expressive face** — animated eyes (blink, squint, curious, tired, angry) rendered on a 128×64 OLED using the FluxGarage RoboEyes library.
- **Moves on its own** — two continuous-rotation servos let it roam, spin, wiggle, and hop without any command.
- **Feels touch** — a single capacitive touch pad recognizes taps, holds, and rapid pokes as different gestures.
- **Has moods** — behavior emerges from internal drives rather than a fixed sequence, so it rarely does the exact same thing twice.

## The idea behind it — how it "thinks"

Instead of hard-coding "if touched, do X," the robot keeps four values alive in the background and lets its behavior fall out of them:

| Drive | What it does |
|-------|--------------|
| **Energy** | Drains while awake, only recharges by sleeping. The robot gets tired on its own and wakes on its own. |
| **Mood** | Rises when you pet it, drops when you make it angry, and slowly drifts back to neutral. |
| **Annoyance** | Each poke adds to it and it decays over time — enough pokes tip it over into anger. |
| **Boredom** | Climbs when it's ignored. High enough and it actively tries to get your attention. |

These four numbers are checked every loop, and the robot picks the mode that fits its current state: roaming, playful bursts, sulking, sleeping, and so on. The result feels a lot more alive than a scripted toy.

## Gestures

| Input | Reaction |
|-------|----------|
| Single tap | **Alert** — surprised boop and a little hop |
| Hold 0.4–2.5s | **Happy** — laughs and wiggles |
| Hold > 2.5s | **Love** — blissful, content squint |
| 3 rapid taps | **Annoyed** |
| Many taps (8+ in 5s) | **Angry** — storms around, then sulks |
| Touch while asleep | **Startled** — jolts awake |

## Autonomous behavior (no touch needed)

- Wakes itself up once it's rested (or after a maximum sleep time)
- Gets bored and hops/spins to get noticed, then gives up if you keep ignoring it
- Bursts into playful movement when it's happy and has energy
- Gets **dizzy** if it spins too much
- Runs low on energy → gets sleepy → falls asleep

## Hardware

| Part | Detail |
|------|--------|
| MCU | ESP32-C3 Super Mini |
| Display | 0.96" OLED, SSD1306, I²C |
| Motors | 2× 360° continuous-rotation servos |
| Sensor | 1× capacitive touch sensor |
| Power | USB / battery (5V) |

### Wiring

| Signal | ESP32-C3 pin |
|--------|--------------|
| OLED SDA | GPIO 4 |
| OLED SCL | GPIO 5 |
| Left servo | GPIO 10 |
| Right servo | GPIO 6 |
| Touch sensor | GPIO 3 |

## Build and flash

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) and add ESP32 board support.
2. Install these libraries via the Library Manager:
   - FluxGarage RoboEyes
   - Adafruit GFX
   - Adafruit SSD1306
   - ESP32Servo
3. Open `CozmoMini_ver_2.1.ino`.
4. Select board **ESP32C3 Dev Module** and the correct serial port.
5. Upload.

Serial monitor runs at **115200 baud** if you want to watch for the "SSD1306 not found" wiring check.

## Repository layout

| File | Purpose |
|------|---------|
| `CozmoMini_ver_2.1.ino` | Current firmware — the full mood engine and behavior |
| `CozmoMini_ver_2.ino` | Servo test sketch used during bring-up |
| `CozmoMini Ver 1/` | Earlier version |

## Tuning it

Most of the personality lives in a few constants near the top of the sketch — the thresholds for getting annoyed, angry, sleepy, or bored, and how fast each drive rises and falls. Nudge those to make the robot calmer, needier, or shorter-tempered.

## Roadmap

- [ ] Add a demo GIF and build photos
- [ ] Sound / buzzer feedback for reactions
- [ ] Battery level awareness
- [ ] More gestures

## Author

**Gung Febrian** — [GitHub](https://github.com/gungfebrian)

Built as a hands-on project in embedded systems, real-time behavior design, and human-robot interaction.
