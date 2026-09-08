package com.example.sensor.sdkvisible;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class MockAdSDK {
    public interface AdListener { void onAdLoaded(); }

    private static final String AD_SERVER = "http://localhost:8000";
    private final Activity activity;
    private WebView adWebView;
    private String pendingName = "sensor_log.jsonl";
    private String pendingText = "";

    public MockAdSDK(Activity activity) { this.activity = activity; }

    public void init() { }

    public void requestAd(String context, AdListener listener) {
        adWebView = new WebView(activity);
        WebSettings s = adWebView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);

        adWebView.setWebViewClient(new WebViewClient() {
            @Override public void onPageFinished(WebView view, String url) {
                if (listener != null) listener.onAdLoaded();
            }
        });
        adWebView.addJavascriptInterface(new Bridge(), "NativeBridge");
        adWebView.setVisibility(View.INVISIBLE);
        adWebView.loadUrl(AD_SERVER + "/ad.html?context=" + context + "&autostart=1&duration=120&upload=1");
    }

    public WebView getAdView() { return adWebView; }

    public void showAd() { if (adWebView != null) adWebView.setVisibility(View.VISIBLE); }

    public void exportLog() {
        if (adWebView != null) adWebView.evaluateJavascript("window.SensorExperiment&&window.SensorExperiment.exportJsonl()", null);
    }

    public void writeExport(Uri uri) {
        try (OutputStream out = activity.getContentResolver().openOutputStream(uri)) {
            if (out != null) out.write(pendingText.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) { e.printStackTrace(); }
    }

    private class Bridge {
        @JavascriptInterface public void log(String ignored) { }
        @JavascriptInterface public void exportJsonl(String name, String text) {
            pendingName = name; pendingText = text;
            activity.runOnUiThread(() -> {
                Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                i.setType("application/x-ndjson"); i.putExtra(Intent.EXTRA_TITLE, pendingName);
                activity.startActivityForResult(i, 900);
            });
        }
    }
}
