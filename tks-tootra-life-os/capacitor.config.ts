import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.tootra.tks",
  appName: "TKS Tootra",
  webDir: "dist/public",
  server: {
    // In production, the app loads from the bundled web assets
    // For development, you can set this to your dev server URL:
    // url: "http://YOUR_LOCAL_IP:3000",
    // cleartext: true,
    androidScheme: "https",
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: "#0a0a0f",
      showSpinner: false,
      androidScaleType: "CENTER_CROP",
      splashFullScreen: true,
      splashImmersive: true,
    },
    StatusBar: {
      style: "DARK",
      backgroundColor: "#0a0a0f",
    },
    LocalNotifications: {
      smallIcon: "ic_notification",
      iconColor: "#d4a843",
      sound: "default",
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
  },
};

export default config;
