package com.example.sensor.directwebview;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.webkit.JavascriptInterface;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class SensorBridge {
    private final Activity activity;
    private String pendingName = "sensor_log.jsonl";
    private String pendingText = "";
    public SensorBridge(Activity activity) { this.activity = activity; }

    @JavascriptInterface public void log(String ignored) { }

    @JavascriptInterface public void exportJsonl(String name, String text) {
        pendingName = name; pendingText = text;
        activity.runOnUiThread(() -> {
            Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            i.setType("application/x-ndjson");
            i.putExtra(Intent.EXTRA_TITLE, pendingName);
            activity.startActivityForResult(i, 900);
        });
    }
    public void writeExport(Uri uri) {
        try (OutputStream out = activity.getContentResolver().openOutputStream(uri)) {
            if (out != null) out.write(pendingText.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) { e.printStackTrace(); }
    }
}
