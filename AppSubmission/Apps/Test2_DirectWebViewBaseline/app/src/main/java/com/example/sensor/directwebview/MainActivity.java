package com.example.sensor.directwebview;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends Activity {
    protected static final String SERVER = "http://localhost:8000";
    protected WebView webView;
    protected SensorBridge bridge;

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 900 && resultCode == RESULT_OK && data != null) bridge.writeExport(data.getData());
    }

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        webView = new WebView(this);
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        webView.setWebViewClient(new WebViewClient());
        bridge = new SensorBridge(this);
        webView.addJavascriptInterface(bridge, "NativeBridge");
        setContentView(webView);
        webView.loadUrl(SERVER + "/browser.html?context=direct_webview_baseline&upload=1");
    }
}
