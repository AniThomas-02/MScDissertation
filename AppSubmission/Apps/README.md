# Sensor Access Reproduction Suite

This repository contains the code used to reproduce the sensor-access experiments described in the dissertation.

The same JavaScript sensor test is used in every experiment. The only difference between tests is the Android or browser context in which the page is executed.

## Repository structure

```text
Apps/
|-- server/
|   |-- server.py
|   |-- web/
|   |   |-- browser.html
|   |   |-- ad.html
|   |   |-- sensor.js
|   |   `-- style.css
|   `-- logs/
|
|-- Test2_DirectWebViewBaseline/
|-- Test3_SDKVisibleAd/
|-- Test4_InterstitialNeverShown/
`-- Test5_OverlayInterApp/
```

The Chrome baseline does not use an Android app. It uses `browser.html` directly in Chrome.

## Requirements

You will need:

- Python 3.8 or later
- Android Studio
- An Android phone with USB debugging enabled
- `adb` from the Android SDK platform tools
- A USB connection between the phone and computer

No additional Python packages are required for the collection server.

## Sensor interfaces tested

The shared `sensor.js` script tests the following interfaces.

Legacy event APIs:

- `devicemotion`
- `deviceorientation`

Generic Sensor API:

- `Accelerometer`
- `Gyroscope`
- `GravitySensor`
- `LinearAccelerationSensor`
- `Magnetometer`
- `AmbientLightSensor`
- `AbsoluteOrientationSensor`
- `RelativeOrientationSensor`

For each interface, the script records whether the interface exists, whether it can be started, and whether repeated readings are received.

The script also records document focus, visibility state, timestamps, and lifecycle events.

## Running the experiments

### 1. Start the server

From the `Apps/server` directory, run:

```bash
python server.py 8000
```

The server provides the HTML and JavaScript files and stores uploaded JSONL records in:

```text
Apps/server/logs/
```

Keep the server running while performing the tests.

### 2. Connect the Android device

Connect the phone by USB and confirm that it is visible to `adb`:

```bash
adb devices
```

The device should appear with the status:

```text
device
```

If it appears as `unauthorized`, accept the USB debugging prompt on the phone.

### 3. Forward port 8000

Run:

```bash
adb reverse tcp:8000 tcp:8000
```

This allows the phone to access the server using:

```text
http://localhost:8000
```

Run the command again if the phone is disconnected or `adb` is restarted.

## Test 1: Chrome baseline

Open Chrome on the Android phone and navigate to:

```text
http://localhost:8000/browser.html?context=chrome_baseline
```

Press `Start`, allow the test to run for the required duration, then press `Stop`.

The results are uploaded automatically to the server.

The `Export` button can also be used to save a local JSONL copy.

## Test 2: Direct WebView

Open:

```text
Test2_DirectWebViewBaseline/
```

as a project in Android Studio.

Build and run the application on the connected phone.

The application loads the sensor test directly inside a WebView and starts logging automatically.


## Test 3: SDK visible advertisement

Open:

```text
Test3_SDKVisibleAd/
```

as a project in Android Studio.

Build and run the application.

The host application requests an advertisement from the mock SDK. The SDK creates its own WebView, loads the remote advertisement page, and displays it in the application.

Logging begins when the advertisement page loads.


## Test 4: Preloaded interstitial advertisement

Open:

```text
Test4_InterstitialNeverShown/
```

as a project in Android Studio.

Build and run the application.

The mock interstitial component creates a WebView and loads the advertisement page, but the WebView is never added to the visible application layout.

Sensor logging begins when the page loads.

Use the host application's export control if a local copy of the data is required.


## Test 5: Overlay inter-app execution

Open:

```text
Test5_OverlayInterApp/
```

as a project in Android Studio.

Build and run the application.

First, grant the application permission to display over other applications.

Then start the overlay.

The application service creates a WebView using `TYPE_APPLICATION_OVERLAY`. The WebView loads the same remote advertisement page used in the other tests.

After the overlay has started, another application can be brought to the foreground while the sensor test remains active.


## Output files

Most of the Tests have an Export option built in to access the data locally.

Alternatively, each test run produces a JSONL file in:

```text
Apps/server/logs/
```


Each line contains one JSON record.

Typical record types include:

```text
api_probe
api_start
sensor_reading
api_error
marker
api_summary
```

Files are named using the execution context, timestamp, and a random suffix. 

This applies to all but Test 5, where the server logging can cut short. This data should be instead accessed using the command in the same directory where the rest of the data is to be stored:

```bash
adb pull /sdcard/Android/data/con.example.sensor.overlayinterapp/files/overlay_export.jsonl .
```

## Important notes

All experiments use the same `sensor.js` file.

The Android applications expect the server to be available at:

```text
http://localhost:8000
```

Each Android test directory is a separate Gradle project. Open the individual test directory in Android Studio rather than opening the parent `Apps` directory.
