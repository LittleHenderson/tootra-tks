package com.hvrtwk.app;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends AppCompatActivity {

    private WebView webView;
    private EditText urlBar;
    private ProgressBar progressBar;
    private SwipeRefreshLayout swipeRefresh;
    private ImageButton btnPaste, btnBrowse, btnBack, btnForward;
    private View browserBar;
    private String hvrtwkJs;
    private boolean isPasteMode = false;
    private static final String PREFS = "hvrtwk_prefs";
    private static final String KEY_LAST_URL = "last_url";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load HvrTwk JS from assets
        hvrtwkJs = loadAsset("hvrtwk.js");

        // Bind views
        webView = findViewById(R.id.webview);
        urlBar = findViewById(R.id.url_bar);
        progressBar = findViewById(R.id.progress_bar);
        swipeRefresh = findViewById(R.id.swipe_refresh);
        btnPaste = findViewById(R.id.btn_paste);
        btnBrowse = findViewById(R.id.btn_browse);
        btnBack = findViewById(R.id.btn_back);
        btnForward = findViewById(R.id.btn_forward);
        browserBar = findViewById(R.id.browser_bar);

        setupWebView();
        setupUrlBar();
        setupButtons();

        // Handle shared text from other apps
        Intent intent = getIntent();
        if (Intent.ACTION_SEND.equals(intent.getAction()) && intent.getType() != null) {
            String shared = intent.getStringExtra(Intent.EXTRA_TEXT);
            if (shared != null) {
                if (shared.startsWith("http://") || shared.startsWith("https://")) {
                    loadUrl(shared);
                } else {
                    loadPasteMode(shared);
                }
                return;
            }
        }

        // Load last URL or default
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String lastUrl = prefs.getString(KEY_LAST_URL, null);
        if (lastUrl != null) {
            loadUrl(lastUrl);
        } else {
            loadPasteMode("Welcome to HvrTwk!\n\nThis app reads any text aloud.\n\n" +
                "HOW TO USE:\n\n" +
                "1. BROWSE MODE: Type a URL in the bar above to visit any website. " +
                "HvrTwk automatically activates on every page.\n\n" +
                "2. PASTE MODE: Tap the clipboard icon to paste any text -- " +
                "emails, messages, articles -- and read it here.\n\n" +
                "3. SHARE TO HVRTWK: From any app, tap Share and select HvrTwk. " +
                "URLs open in the browser. Text opens in paste mode.\n\n" +
                "Touch and hold, then draw a circle around any text you want read aloud. " +
                "Or turn on Continuous mode in the floating panel to read entire pages.\n\n" +
                "Tap the floating button on the right to adjust voice, speed, pitch, and more.");
        }
    }

    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        settings.setUserAgentString(settings.getUserAgentString() + " HvrTwk/1.0");

        // Cache for offline
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);
        settings.setDatabaseEnabled(true);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                super.onPageStarted(view, url, favicon);
                if (!isPasteMode) {
                    urlBar.setText(url);
                }
                progressBar.setVisibility(View.VISIBLE);
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                super.onPageFinished(view, url);
                progressBar.setVisibility(View.GONE);
                swipeRefresh.setRefreshing(false);

                // Inject HvrTwk into every page
                if (hvrtwkJs != null && !hvrtwkJs.isEmpty()) {
                    view.evaluateJavascript(hvrtwkJs, null);
                }

                // Save last URL
                if (!isPasteMode && url != null && !url.startsWith("data:")) {
                    getSharedPreferences(PREFS, MODE_PRIVATE)
                        .edit().putString(KEY_LAST_URL, url).apply();
                }

                updateNavButtons();
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                return false; // Load everything in our WebView
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                progressBar.setProgress(newProgress);
            }
        });

        swipeRefresh.setOnRefreshListener(() -> webView.reload());
        swipeRefresh.setColorSchemeColors(0xFFE94560);
    }

    private void setupUrlBar() {
        urlBar.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_GO ||
                (event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER)) {
                String input = urlBar.getText().toString().trim();
                if (!input.isEmpty()) {
                    isPasteMode = false;
                    browserBar.setVisibility(View.VISIBLE);
                    if (!input.startsWith("http://") && !input.startsWith("https://")) {
                        if (input.contains(".") && !input.contains(" ")) {
                            input = "https://" + input;
                        } else {
                            input = "https://www.google.com/search?q=" +
                                android.net.Uri.encode(input);
                        }
                    }
                    loadUrl(input);
                    hideKeyboard();
                }
                return true;
            }
            return false;
        });
    }

    private void setupButtons() {
        btnPaste.setOnClickListener(v -> {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null && clipboard.hasPrimaryClip()) {
                ClipData clip = clipboard.getPrimaryClip();
                if (clip != null && clip.getItemCount() > 0) {
                    CharSequence text = clip.getItemAt(0).getText();
                    if (text != null && text.length() > 0) {
                        String content = text.toString();
                        if (content.startsWith("http://") || content.startsWith("https://")) {
                            loadUrl(content);
                        } else {
                            loadPasteMode(content);
                        }
                        Toast.makeText(this, "Loaded from clipboard", Toast.LENGTH_SHORT).show();
                        return;
                    }
                }
            }
            Toast.makeText(this, "Nothing in clipboard", Toast.LENGTH_SHORT).show();
        });

        btnBrowse.setOnClickListener(v -> {
            isPasteMode = false;
            browserBar.setVisibility(View.VISIBLE);
            urlBar.requestFocus();
            urlBar.selectAll();
            showKeyboard();
        });

        btnBack.setOnClickListener(v -> {
            if (webView.canGoBack()) webView.goBack();
        });

        btnForward.setOnClickListener(v -> {
            if (webView.canGoForward()) webView.goForward();
        });
    }

    private void loadUrl(String url) {
        isPasteMode = false;
        browserBar.setVisibility(View.VISIBLE);
        urlBar.setText(url);
        webView.loadUrl(url);
        hideKeyboard();
    }

    private void loadPasteMode(String text) {
        isPasteMode = true;
        urlBar.setText("Paste Mode");
        browserBar.setVisibility(View.VISIBLE);

        // Build a simple HTML page with the pasted text
        String escaped = text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("\n\n", "</p><p>")
            .replace("\n", "<br>");

        String html = "<!DOCTYPE html><html><head>" +
            "<meta name='viewport' content='width=device-width,initial-scale=1'>" +
            "<style>" +
            "body{font-family:sans-serif;background:#f5f0eb;color:#222;padding:20px 16px;line-height:1.8;font-size:16px;}" +
            "p{margin-bottom:14px;}" +
            ".paste-label{font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}" +
            "</style></head><body>" +
            "<div class='paste-label'>Pasted Text</div>" +
            "<p>" + escaped + "</p>" +
            "</body></html>";

        webView.loadDataWithBaseURL("https://hvrtwk.local/", html, "text/html", "UTF-8", null);
    }

    private void updateNavButtons() {
        btnBack.setAlpha(webView.canGoBack() ? 1.0f : 0.3f);
        btnForward.setAlpha(webView.canGoForward() ? 1.0f : 0.3f);
    }

    private void hideKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null && getCurrentFocus() != null) {
            imm.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), 0);
        }
    }

    private void showKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        if (imm != null) {
            imm.showSoftInput(urlBar, InputMethodManager.SHOW_IMPLICIT);
        }
    }

    private String loadAsset(String filename) {
        try {
            InputStream is = getAssets().open(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            reader.close();
            return sb.toString();
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
