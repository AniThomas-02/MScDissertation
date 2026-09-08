package com.example.sensor.overlayinterapp;

import android.app.Service;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.view.Gravity;
import android.view.WindowManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.LinearLayout;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public class OverlayService extends Service {
    private WindowManager wm;
    private LinearLayout overlayRoot;
    private WebView adWebView;
    private String pendingText = "";

    @Override public void onCreate() {
        super.onCreate();
        wm = (WindowManager) getSystemService(WINDOW_SERVICE);

        overlayRoot = new LinearLayout(this);
        overlayRoot.setOrientation(LinearLayout.HORIZONTAL);
        overlayRoot.setBackgroundColor(Color.parseColor("#DDFFFFFF"));

        adWebView = new WebView(this);
        WebSettings s = adWebView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        adWebView.setWebViewClient(new WebViewClient());
        adWebView.addJavascriptInterface(new Bridge(), "NativeBridge");
        overlayRoot.addView(adWebView, new LinearLayout.LayoutParams(0, 200, 1));

        Button export = new Button(this);
        export.setText("Export");
        export.setOnClickListener(v -> adWebView.evaluateJavascript("window.SensorExperiment&&window.SensorExperiment.exportJsonl()", null));
        overlayRoot.addView(export);

        WindowManager.LayoutParams p = new WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT, 260,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        );
        p.gravity = Gravity.BOTTOM;
        wm.addView(overlayRoot, p);

        adWebView.loadUrl("http://localhost:8000/ad.html?context=overlay_inter_app&autostart=1&duration=300&upload=1");
    }

    private class Bridge {
        @JavascriptInterface public void log(String ignored) { }
        @JavascriptInterface public void exportJsonl(String name, String text) {
            pendingText = text;
            try (FileOutputStream out = openFileOutput("overlay_export.jsonl", MODE_PRIVATE)) {
                out.write(pendingText.getBytes(StandardCharsets.UTF_8));
            } catch (Exception e) { e.printStackTrace(); }
        }
    }
    @Override public void onDestroy() {
        if (adWebView != null) { adWebView.loadUrl("about:blank"); }
        if (overlayRoot != null) wm.removeView(overlayRoot);
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent i) { return null; }
}
