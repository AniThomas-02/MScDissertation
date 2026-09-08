package com.example.sensor.sdkvisible;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {
    protected MockAdSDK sdk;

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 900 && resultCode == RESULT_OK && data != null) sdk.writeExport(data.getData());
    }

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 24, 24, 24);

        TextView title = new TextView(this);
        title.setText("Host Application");
        title.setTextSize(24);
        root.addView(title);

        Button export = new Button(this);
        export.setText("Export JSONL");
        export.setOnClickListener(v -> sdk.exportLog());
        root.addView(export);

        setContentView(root);

        sdk = new MockAdSDK(this);
        sdk.init();
        sdk.requestAd("sdk_visible", () -> runOnUiThread(() -> {
            root.addView(sdk.getAdView(), new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 320));
            sdk.showAd();
        }));
    }
}
