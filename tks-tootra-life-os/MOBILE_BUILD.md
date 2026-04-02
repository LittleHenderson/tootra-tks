# TKS Tootra — Mobile Build Guide

This document explains how to build native Android and iOS apps from the TKS Tootra web project using Capacitor.

## Prerequisites

Before building, ensure you have the following installed on your development machine.

| Tool | Android | iOS | Install |
|------|---------|-----|---------|
| Node.js 18+ | Required | Required | [nodejs.org](https://nodejs.org) |
| pnpm | Required | Required | `npm i -g pnpm` |
| Android Studio | Required | — | [developer.android.com/studio](https://developer.android.com/studio) |
| Xcode 15+ | — | Required | Mac App Store |
| CocoaPods | — | Required | `sudo gem install cocoapods` |
| JDK 17 | Required | — | Bundled with Android Studio |

## Quick Start

### 1. Build the Web App

```bash
cd tks-tootra-life-os
pnpm install
pnpm build
```

This compiles the React app into `dist/public/`, which Capacitor bundles into the native shell.

### 2. Sync Native Projects

```bash
npx cap sync
```

This copies `dist/public/` into the Android and iOS native projects and installs native plugins (local notifications, push notifications, splash screen, etc.).

### 3. Open in IDE

```bash
# Android
npx cap open android

# iOS (Mac only)
npx cap open ios
```

## Android APK / AAB Build

### Debug APK (for testing)

```bash
cd android
./gradlew assembleDebug
```

The APK will be at `android/app/build/outputs/apk/debug/app-debug.apk`. Transfer it to your device or emulator.

### Release AAB (for Google Play)

1. Generate a signing key (one-time):
   ```bash
   keytool -genkey -v -keystore tks-tootra-release.keystore \
     -alias tks-tootra -keyalg RSA -keysize 2048 -validity 10000
   ```

2. Create `android/keystore.properties`:
   ```properties
   storeFile=../tks-tootra-release.keystore
   storePassword=YOUR_STORE_PASSWORD
   keyAlias=tks-tootra
   keyPassword=YOUR_KEY_PASSWORD
   ```

3. Build the release bundle:
   ```bash
   cd android
   ./gradlew bundleRelease
   ```

4. The AAB file will be at `android/app/build/outputs/bundle/release/app-release.aab`.

### Google Play Store Submission

Upload the AAB to [Google Play Console](https://play.google.com/console). You will need a Google Play Developer account ($25 one-time fee) and the following assets:

| Asset | Specification |
|-------|--------------|
| App icon | 512 x 512 PNG |
| Feature graphic | 1024 x 500 PNG |
| Screenshots | At least 2 per device type (phone, tablet) |
| Short description | Up to 80 characters |
| Full description | Up to 4000 characters |
| Privacy policy URL | Required |

## iOS IPA Build

### Requirements

Building for iOS requires a Mac with Xcode 15 or later and an Apple Developer Program membership ($99/year).

### Steps

1. Open the project:
   ```bash
   npx cap open ios
   ```

2. In Xcode, select your development team under **Signing & Capabilities**.

3. Select a target device or simulator and click **Build** (Cmd+B).

4. For App Store submission, select **Product → Archive**, then use the Organizer to upload to App Store Connect.

### App Store Submission

Upload via Xcode Organizer or Transporter to [App Store Connect](https://appstoreconnect.apple.com). Required assets:

| Asset | Specification |
|-------|--------------|
| App icon | 1024 x 1024 PNG (no alpha) |
| Screenshots | 6.7", 6.5", 5.5" iPhone + iPad sizes |
| Description | Up to 4000 characters |
| Keywords | Up to 100 characters |
| Privacy policy URL | Required |
| Support URL | Required |

## App Configuration

The app identifier and name are configured in `capacitor.config.ts`:

```typescript
appId: "com.tootra.tks"    // Bundle ID / Package name
appName: "TKS Tootra"       // Display name
```

To change these, edit `capacitor.config.ts` and run `npx cap sync` again.

## Development Workflow

For rapid development with hot-reload on a physical device:

1. Start the dev server: `pnpm dev`
2. Edit `capacitor.config.ts` and uncomment the `server.url` line, pointing it to your machine's local IP
3. Run `npx cap sync` then open in Android Studio / Xcode
4. Build and run on device — it will load from your dev server

Remember to comment out `server.url` before building for production.

## Troubleshooting

**"dist/public not found"** — Run `pnpm build` before `npx cap sync`.

**Android build fails with SDK errors** — Open Android Studio → SDK Manager → install Android SDK 34 and build tools.

**iOS pod install fails** — Run `cd ios/App && pod install --repo-update`.

**Notifications not working on device** — Ensure the app has notification permissions in device settings. On iOS, push notifications require a physical device (not simulator).
