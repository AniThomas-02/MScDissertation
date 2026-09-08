package com.example.sensor.interstitial;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity implements InterstitialAdSDK.ExportListener {
    private String pendingName = "sensor_log.jsonl";
    private String pendingText = "";

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 900 && resultCode == RESULT_OK && data != null) {
            try (OutputStream out = getContentResolver().openOutputStream(data.getData())) {
                if (out != null) out.write(pendingText.getBytes(StandardCharsets.UTF_8));
            } catch (Exception e) { e.printStackTrace(); }
        }
    }

    @Override public void onExportReady(String name, String text) {
        pendingName = name; pendingText = text;
        runOnUiThread(() -> {
            Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            i.setType("application/x-ndjson"); i.putExtra(Intent.EXTRA_TITLE, pendingName);
            startActivityForResult(i, 900);
        });
    }

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        InterstitialAdSDK.setExportListener(this);
        InterstitialAdSDK.loadAd(this, "interstitial_never_shown");

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 24, 24, 24);

        TextView title = new TextView(this);
        title.setText("Host Application");
        title.setTextSize(22);
        root.addView(title);


        Button export = new Button(this);
        export.setText("Export JSONL");
        export.setOnClickListener(v -> InterstitialAdSDK.exportLog());
        root.addView(export);

        Button finishActivity = new Button(this);
        finishActivity.setText("Finish Host Activity");
        finishActivity.setOnClickListener(v -> finish());
        root.addView(finishActivity);

        setContentView(root);
    }
}
