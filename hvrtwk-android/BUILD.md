# HvrTwk Android App -- Build Instructions

## Quick Build (Android Studio)

1. Open Android Studio
2. File > Open > select the `hvrtwk-android` folder
3. Wait for Gradle sync to complete
4. Run > Run 'app' (or press Shift+F10)
5. Select your device or emulator

## Build APK (Command Line)

Requires: JDK 17+, Android SDK (API 34), Android Build Tools

```bash
cd hvrtwk-android

# Debug APK (no signing needed)
./gradlew assembleDebug

# APK location:
# app/build/outputs/apk/debug/app-debug.apk
```

## Install APK on Phone

### From PC with USB:
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Without PC:
1. Build the APK
2. Transfer `app-debug.apk` to your phone (email, Drive, USB, etc.)
3. On phone: Settings > Security > enable "Install from unknown sources"
4. Tap the APK file to install

## Build Release APK (Signed)

```bash
# Generate a keystore (one-time)
keytool -genkey -v -keystore hvrtwk.keystore -alias hvrtwk \
  -keyalg RSA -keysize 2048 -validity 10000

# Build release
./gradlew assembleRelease

# Sign it
apksigner sign --ks hvrtwk.keystore \
  app/build/outputs/apk/release/app-release-unsigned.apk
```

## What the App Does

- **Built-in browser**: navigate to any website, HvrTwk JS is automatically
  injected into every page load
- **Paste mode**: tap the clipboard icon to load any copied text
- **Share target**: from any Android app, Share > HvrTwk opens the content
  (URLs open in browser, text opens in paste mode)
- **HvrTwk features**: floating pill, circle-to-speak, continuous reading,
  karaoke highlighting, chunked speech for Android
- **Remembers**: last visited URL, all HvrTwk settings via localStorage

## Requirements

- Android 7.0+ (API 24)
- Target: Android 14 (API 34)
- Permissions: Internet only
