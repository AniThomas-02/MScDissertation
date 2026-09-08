package com.example.sensor.overlayinterapp;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {
    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(30, 50, 30, 30);

        Button permission = new Button(this);
        permission.setText("Grant Overlay Permission");
        permission.setOnClickListener(v -> {
            if (!Settings.canDrawOverlays(this)) {
                startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + getPackageName())));
            }
        });
        root.addView(permission);

        Button start = new Button(this);
        start.setText("Start Overlay Ad");
        start.setOnClickListener(v -> { if (Settings.canDrawOverlays(this)) startService(new Intent(this, OverlayService.class)); });
        root.addView(start);

        Button stop = new Button(this);
        stop.setText("Stop Overlay Ad");
        stop.setOnClickListener(v -> stopService(new Intent(this, OverlayService.class)));
        root.addView(stop);

        setContentView(root);
    }
}
