# AllSensorWebViewCollector - How to Reproduce / Run This

This is the Android data-collection app used to produce sensor_log.jsonl and
key_log.jsonl for the modelling scripts in the other folders. It's an Android
Studio project (Java + Gradle).

## 1. What's in this project

```
AllSensorWebViewCollector/
  build.gradle, settings.gradle, gradle.properties   
  gradle/wrapper/                                     
  gradlew, gradlew.bat                                
  app/build.gradle                                     - app module config 
  app/src/main/AndroidManifest.xml                    - permissions, activity, FileProvider
  app/src/main/java/.../MainActivity.java             - hosts the WebView, writes/zips/shares the log files
  app/src/main/assets/collector.html                  - the actual sensor + typing-prompt collector UI
  app/src/main/res/                                    - layout, strings, theme, FileProvider paths
```

The data-collection logic lives in collector.html (JavaScript,
runs in the WebView) and MainActivity.java where/how files are saved or shared.

## 2. How to build and run it

1. Open the AllSensorWebViewCollector folder in Android Studio.
2. Let Gradle sync
3. Connect an Android phone with USB debugging enabled
4. Run the app (the green Run button, or `./gradlew installDebug` from a
   terminal in this folder).

## 3. Using the app to collect data

1. Tap **Start sensors**.
2. Choose **Prompts per key** (20, 50, or 100 repeats per letter).
3. Tap **New typing session**.
4. Type the big letter shown on screen into the text box below it, over
   and over as prompts advance.
5. When the session says "Done", tap **Share logs** to export a zip
   containing sensor_log.jsonl, key_log.jsonl, and event_log.jsonl via the
   Android share sheet (email, Drive, etc).
6. **Clear logs** wipes all three log files from app storage. Use this
   between participants/sessions, not mid-session.


## 4. What actually gets logged

Three JSONL files, one JSON object per line:

- **sensor_log.jsonl** - continuous sensor readings, `kind: "sensor"`,
  tagged with a `type` field for which sensor produced the row. Includes
  both the legacy Web APIs (`devicemotion`: acc_x/y/z, gacc_x/y/z,
  rot_alpha/beta/gamma; `deviceorientation`: ori_alpha/beta/gamma) and the
  newer Generic Sensor API classes (Accelerometer, LinearAccelerationSensor,
  GravitySensor, Gyroscope, Magnetometer, AmbientLightSensor,
  AbsoluteOrientationSensor, RelativeOrientationSensor), when the phone/
  browser actually exposes them.
- **key_log.jsonl** - one row per keypress during a typing session:
  `trial_id`, `target_key` (what was asked for), `actual_key` (what was
  typed), `correct`, and `attempt` (which retry this was for that prompt).
- **event_log.jsonl** - diagnostic events: `sensor_started`,
  `sensor_unavailable`, `sensor_start_failed`, `sensor_error`,
  `typing_session_start`/`_complete`, `prompt_shown`, `keydown`/`keyup`,
  and a `collector_start` row listing which Generic Sensor API classes
  were available on that device. 

## 5. Requirements

- Android Studio (any recent version compatible with Gradle 9.0.0 /
  Android Gradle Plugin 8.7.3, e.g. Android Studio Ladybug or newer)
- An Android device running the latest version of Android
