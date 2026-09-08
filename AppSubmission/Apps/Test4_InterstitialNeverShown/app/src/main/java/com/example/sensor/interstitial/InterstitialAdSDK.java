package com.example.sensor.interstitial;

import android.content.Context;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
public class InterstitialAdSDK {
    private static final String AD_SERVER = "http://localhost:8000";
    private static WebView adWebView;
    private static String pendingText = "";
    public interface ExportListener { void onExportReady(String name, String text); }
    private static ExportListener exportListener;

    public static void loadAd(Context appContext, String context) {
        if (adWebView != null) return;
        adWebView = new WebView(appContext.getApplicationContext());
        WebSettings s = adWebView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        adWebView.setWebViewClient(new WebViewClient());
        adWebView.addJavascriptInterface(new Bridge(), "NativeBridge");
        adWebView.loadUrl(AD_SERVER + "/ad.html?context=" + context + "&autostart=1&duration=300&upload=1");
        // never attached to any layout
    }

    public static void setExportListener(ExportListener listener) { exportListener = listener; }

    public static void exportLog() {
        if (adWebView != null) adWebView.evaluateJavascript("window.SensorExperiment&&window.SensorExperiment.exportJsonl()", null);
    }

    private static class Bridge {
        @JavascriptInterface public void log(String ignored) { }
        @JavascriptInterface public void exportJsonl(String name, String text) {
            pendingText = text;
            if (exportListener != null) exportListener.onExportReady(name, text);
        }
    }
}
