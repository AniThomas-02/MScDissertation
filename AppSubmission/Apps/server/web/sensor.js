"use strict";

(() => {
    const params = new URLSearchParams(location.search);

    const context = params.get("context") || "browser_baseline";
    const defaultDuration = Number(params.get("duration") || 20);
    const autoStart = params.get("autostart") === "1";
    const postToServer = params.get("upload") !== "0";

    const SUCCESS_READING_COUNT = 2;

    const legacySensors = [
        "devicemotion",
        "deviceorientation"
    ];

    const genericSensors = [
        ["Accelerometer", ["x", "y", "z"]],
        ["Gyroscope", ["x", "y", "z"]],
        ["GravitySensor", ["x", "y", "z"]],
        ["LinearAccelerationSensor", ["x", "y", "z"]],
        ["Magnetometer", ["x", "y", "z"]],
        ["AmbientLightSensor", ["illuminance"]],
        ["AbsoluteOrientationSensor", ["quaternion"]],
        ["RelativeOrientationSensor", ["quaternion"]]
    ];

    let records = [];
    let pendingUpload = [];
    let stopCallbacks = [];
    let apiState = {};

    let running = false;
    let sessionId = null;
    let runTimer = null;
    let uploadTimer = null;

    function createBaseRecord(type) {
        return {
            type,
            session_id: sessionId,
            context,
            wall_time_ms: Date.now(),
            performance_time_ms: performance.now(),
            visibility_state: document.visibilityState,
            has_focus: document.hasFocus(),
            secure_context: window.isSecureContext,
            url: location.href
        };
    }

    function writeRecord(record) {
        records.push(record);
        pendingUpload.push(record);

        if (
            window.NativeBridge &&
            typeof window.NativeBridge.log === "function"
        ) {
            try {
                window.NativeBridge.log(JSON.stringify(record));
            } catch (_) {
            }
        }
    }

    function addMarker(event, extra = {}) {
        writeRecord({
            ...createBaseRecord("marker"),
            event,
            ...extra
        });
    }

    window.ExperimentLog = {
        marker: addMarker,
        nativeMarker: addMarker
    };

    function setStatus(text) {
        const status = document.getElementById("status");

        if (status) {
            status.textContent = text;
        }
    }

    function getApiState(name) {
        if (!apiState[name]) {
            apiState[name] = {
                exists: false,
                started: false,
                readings: 0,
                outcome: "NOT_TESTED",
                error: null
            };
        }

        return apiState[name];
    }

    function renderResults() {
        const body = document.getElementById("results-body");

        if (!body) {
            return;
        }

        body.innerHTML = "";

        const sensorNames = [
            ...legacySensors,
            ...genericSensors.map(([name]) => name)
        ];

        for (const name of sensorNames) {
            const state = getApiState(name);
            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${name}</td>
                <td>${state.exists ? "yes" : "no"}</td>
                <td>${state.started ? "yes" : "no"}</td>
                <td>${state.readings}</td>
                <td>${state.outcome}</td>
            `;

            body.appendChild(row);
        }
    }

    function classifyApi(name) {
        const state = getApiState(name);

        if (!state.exists) {
            state.outcome = "UNAVAILABLE";
        } else if (state.readings >= SUCCESS_READING_COUNT) {
            state.outcome = "SUCCESS_REPEATED_READINGS";
        } else {
            state.outcome = "EXPOSED_NO_REPEATED_READINGS";
        }

        writeRecord({
            ...createBaseRecord("api_summary"),
            api: name,
            ...state
        });
    }

    function serialiseValue(value) {
        if (value == null) {
            return null;
        }

        if (Array.isArray(value)) {
            return value.map(Number);
        }

        if (typeof value === "number") {
            return Number.isFinite(value) ? value : null;
        }

        return value;
    }

    function getXYZ(reading) {
        if (!reading) {
            return null;
        }
        return {
            x: reading.x,
            y: reading.y,
            z: reading.z
        };
    }

    function getRotation(reading) {
        if (!reading) {
            return null;
        }

        return {
            alpha: reading.alpha,
            beta: reading.beta,
            gamma: reading.gamma
        };
    }

    function startLegacySensor(name) {
        const state = getApiState(name);

        state.exists = `on${name}` in window;

        writeRecord({
            ...createBaseRecord("api_probe"),
            api: name,
            exists: state.exists
        });

        if (!state.exists) {
            return;
        }

        const handler = event => {
            state.started = true;
            state.readings++;

            let values;

            if (name === "devicemotion") {
                values = {
                    acceleration: getXYZ(event.acceleration),
                    accelerationIncludingGravity: getXYZ(
                        event.accelerationIncludingGravity
                    ),
                    rotationRate: getRotation(event.rotationRate),
                    interval: event.interval ?? null
                };
            } else {
                values = {
                    alpha: event.alpha,
                    beta: event.beta,
                    gamma: event.gamma,
                    absolute: event.absolute
                };
            }

            writeRecord({
                ...createBaseRecord("sensor_reading"),
                api: name,
                reading_index: state.readings,
                values
            });

            if (state.readings % 10 === 0) {
                renderResults();
            }
        };

        window.addEventListener(name, handler);

        stopCallbacks.push(() => {
            window.removeEventListener(name, handler);
        });
    }


    function startGenericSensor(name, fields) {
        const state = getApiState(name);
        const SensorConstructor = window[name];

        state.exists = typeof SensorConstructor === "function";

        writeRecord({
            ...createBaseRecord("api_probe"),
            api: name,
            exists: state.exists
        });

        if (!state.exists) {
            return;
        }

        try {
            const sensor = new SensorConstructor({
                frequency: 60
            });

            sensor.addEventListener("reading", () => {
                state.readings++;

                const values = {};

                for (const field of fields) {
                    values[field] = serialiseValue(sensor[field]);
                }

                writeRecord({
                    ...createBaseRecord("sensor_reading"),
                    api: name,
                    reading_index: state.readings,
                    sensor_timestamp: sensor.timestamp ?? null,
                    values
                });

                if (state.readings % 10 === 0) {
                    renderResults();
                }
            });

            sensor.addEventListener("error", event => {
                state.error = event.error
                    ? `${event.error.name}: ${event.error.message}`
                    : "Sensor error";

                writeRecord({
                    ...createBaseRecord("api_error"),
                    api: name,
                    error: state.error
                });

                renderResults();
            });

            sensor.start();
            state.started = true;

            writeRecord({
                ...createBaseRecord("api_start"),
                api: name,
                requested_frequency_hz: 60
            });

            stopCallbacks.push(() => {
                try {
                    sensor.stop();
                } catch (_) {
                    // Sensor may already have stopped.
                }
            });

        } catch (error) {
            state.error = `${error.name}: ${error.message}`;

            writeRecord({
                ...createBaseRecord("api_start_error"),
                api: name,
                error: state.error
            });
        }
    }


    async function flushUpload() {
        if (!postToServer || pendingUpload.length === 0) {
            return;
        }

        const batch = pendingUpload.splice(0, pendingUpload.length);

        try {
            const response = await fetch("/collect", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    records: batch
                }),
                keepalive: true
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

        } catch (_) {
            // Restore the batch so temporary network failures do not lose data.
            pendingUpload = batch.concat(pendingUpload);

            setStatus(
                `Server upload pending (${pendingUpload.length} records)`
            );
        }
    }


    async function startTest(durationOverride) {
        if (running) {
            return;
        }

        running = true;

        records = [];
        pendingUpload = [];
        stopCallbacks = [];
        apiState = {};

        const timestamp = new Date()
            .toISOString()
            .replace(/[:.]/g, "-");

        const suffix = Math.random()
            .toString(36)
            .slice(2, 7);

        sessionId = `${context}_${timestamp}_${suffix}`;

        const durationInput = document.getElementById("duration");

        const duration = Number(
            durationOverride ||
            durationInput?.value ||
            defaultDuration
        );

        addMarker("TEST_START", {
            duration_seconds: duration,
            user_agent: navigator.userAgent
        });

        setStatus(`Running ${context} (${duration}s)`);

        for (const sensor of legacySensors) {
            startLegacySensor(sensor);
        }

        for (const [name, fields] of genericSensors) {
            startGenericSensor(name, fields);
        }

        renderResults();

        uploadTimer = setInterval(flushUpload, 1000);
        runTimer = setTimeout(stopTest, duration * 1000);
    }


    async function stopTest() {
        if (!running) {
            return;
        }

        running = false;

        clearTimeout(runTimer);
        clearInterval(uploadTimer);

        for (const stop of stopCallbacks) {
            try {
                stop();
            } catch (_) {
            }
        }

        stopCallbacks = [];

        const sensorNames = [
            ...legacySensors,
            ...genericSensors.map(([name]) => name)
        ];

        for (const name of sensorNames) {
            classifyApi(name);
        }

        addMarker("TEST_STOP");

        await flushUpload();

        renderResults();
        setStatus(`Finished ${sessionId}`);
    }


    function exportJsonl() {
        const contents =
            records.map(record => JSON.stringify(record)).join("\n") + "\n";

        const filename = `${sessionId || context}.jsonl`;

        if (
            window.NativeBridge &&
            typeof window.NativeBridge.exportJsonl === "function"
        ) {
            window.NativeBridge.exportJsonl(filename, contents);
            return;
        }

        const blob = new Blob(
            [contents],
            { type: "application/x-ndjson" }
        );

        const link = document.createElement("a");

        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();

        setTimeout(() => {
            URL.revokeObjectURL(link.href);
        }, 1000);
    }


    document.addEventListener("visibilitychange", () => {
        addMarker("DOCUMENT_VISIBILITY", {
            state: document.visibilityState
        });
    });

    window.addEventListener("focus", () => {
        addMarker("WINDOW_FOCUS");
    });

    window.addEventListener("blur", () => {
        addMarker("WINDOW_BLUR");
    });

    window.addEventListener("pagehide", () => {
        addMarker("PAGE_HIDE");
        flushUpload();
    });


    window.SensorExperiment = {
        start: startTest,
        stop: stopTest,
        exportJsonl,
        marker: addMarker,
        getSessionId: () => sessionId,
        getRecordCount: () => records.length,
        getElapsedMs: () => performance.now()
    };


    window.addEventListener("DOMContentLoaded", () => {
        const durationInput = document.getElementById("duration");

        if (durationInput) {
            durationInput.value = defaultDuration;
        }

        document
            .getElementById("start")
            ?.addEventListener("click", () => startTest());

        document
            .getElementById("stop")
            ?.addEventListener("click", stopTest);

        document
            .getElementById("export")
            ?.addEventListener("click", exportJsonl);

        renderResults();

        setStatus(
            `Ready: context=${context}; ` +
            `secureContext=${window.isSecureContext}`
        );

        if (autoStart) {
            setTimeout(
                () => startTest(defaultDuration),
                300
            );
        }
    });

})();