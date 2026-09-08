package com.example.allsensorcollector;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.ConsoleMessage;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class MainActivity extends Activity {

    private static final String SENSOR_LOG = "sensor_log.jsonl";
    private static final String KEY_LOG = "key_log.jsonl";
    private static final String EVENT_LOG = "event_log.jsonl";

    private TextView statusText;
    private WebView webView;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // To keep the buttons below the status bar
        WindowCompat.setDecorFitsSystemWindows(getWindow(), true);
        setContentView(R.layout.activity_main);
        View rootLayout = findViewById(R.id.rootLayout);
        ViewCompat.setOnApplyWindowInsetsListener(rootLayout, (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        statusText = findViewById(R.id.statusText);
        webView = findViewById(R.id.webView);

        findViewById(R.id.reloadButton).setOnClickListener(v ->
                webView.loadUrl(getString(R.string.collector_url)));
        findViewById(R.id.clearButton).setOnClickListener(v -> clearLogs());
        findViewById(R.id.shareButton).setOnClickListener(v -> shareLogs());

        setUpWebView(webView);
        webView.loadUrl(getString(R.string.collector_url));
    }

    private void setUpWebView(WebView webView) {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage consoleMessage) {
                Log.d("WebViewConsole", consoleMessage.message());
                return true;
            }
        });

        webView.addJavascriptInterface(new LogBridge(), "AndroidLogger");
    }

    //Bridge exposed to collector.html

    public class LogBridge {
        @JavascriptInterface
        public void appendSensorLines(String lines) {
            appendLines(SENSOR_LOG, lines);
        }

        @JavascriptInterface
        public void appendKeyLines(String lines) {
            appendLines(KEY_LOG, lines);
        }

        @JavascriptInterface
        public void appendEventLines(String lines) {
            appendLines(EVENT_LOG, lines);
        }

        @JavascriptInterface
        public void updateStatus(String text) {
            runOnUiThread(() -> statusText.setText(text));
        }
    }

    //File Handlinf

    private synchronized void appendLines(String filename, String lines) {
        if (lines == null || lines.trim().isEmpty()) return;

        File file = new File(getFilesDir(), filename);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
            writer.write(lines);
            if (!lines.endsWith("\n")) writer.write("\n");
        } catch (IOException e) {
            Log.e("AllSensorCollector", "Failed writing " + filename, e);
        }
    }

    private void clearLogs() {
        // clears the JS-side buffers first before deleting to prevent the flushing of data to a deleted file
        webView.evaluateJavascript("window.clearBuffers && window.clearBuffers();", value -> {
            for (String name : new String[]{SENSOR_LOG, KEY_LOG, EVENT_LOG}) {
                File file = new File(getFilesDir(), name);
                if (file.exists()) file.delete();
            }
            statusText.setText(R.string.status_cleared);
            Toast.makeText(this, R.string.status_cleared, Toast.LENGTH_SHORT).show();
        });
    }

    private void shareLogs() {
        try {
            File zip = new File(getCacheDir(), "webview_sensor_logs.zip");
            zipFiles(zip, SENSOR_LOG, KEY_LOG, EVENT_LOG);

            Uri uri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", zip);

            Intent intent = new Intent(Intent.ACTION_SEND);
            intent.setType("application/zip");
            intent.putExtra(Intent.EXTRA_STREAM, uri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(intent, "Share sensor logs"));
        } catch (Exception e) {
            Toast.makeText(this, "Share failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void zipFiles(File zipFile, String... names) throws IOException {
        byte[] buffer = new byte[4096];

        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(zipFile))) {
            for (String name : names) {
                File file = new File(getFilesDir(), name);
                if (!file.exists()) continue;

                zos.putNextEntry(new ZipEntry(name));
                try (FileInputStream in = new FileInputStream(file)) {
                    int len;
                    while ((len = in.read(buffer)) > 0) {
                        zos.write(buffer, 0, len);
                    }
                }
                zos.closeEntry();
            }
        }
    }
}
