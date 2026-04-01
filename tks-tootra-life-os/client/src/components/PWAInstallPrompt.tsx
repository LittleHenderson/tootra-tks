import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { X, Download, Smartphone } from "lucide-react";

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

export default function PWAInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [showBanner, setShowBanner] = useState(false);
  const [isIOS, setIsIOS] = useState(false);
  const [showIOSGuide, setShowIOSGuide] = useState(false);

  useEffect(() => {
    // Check if already dismissed recently
    const dismissed = localStorage.getItem("tks-pwa-dismissed");
    if (dismissed) {
      const dismissedAt = parseInt(dismissed, 10);
      // Don't show again for 7 days
      if (Date.now() - dismissedAt < 7 * 24 * 60 * 60 * 1000) return;
    }

    // Check if already installed
    if (window.matchMedia("(display-mode: standalone)").matches) return;

    // Detect iOS
    const isIOSDevice = /iPad|iPhone|iPod/.test(navigator.userAgent) && !(window as any).MSStream;
    setIsIOS(isIOSDevice);

    if (isIOSDevice) {
      // On iOS, show the manual install guide after a delay
      const timer = setTimeout(() => setShowBanner(true), 3000);
      return () => clearTimeout(timer);
    }

    // For Chrome/Edge/etc, listen for the beforeinstallprompt event
    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      // Show after a short delay so it doesn't feel aggressive
      setTimeout(() => setShowBanner(true), 2000);
    };

    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === "accepted") {
      setShowBanner(false);
    }
    setDeferredPrompt(null);
  };

  const handleDismiss = () => {
    setShowBanner(false);
    localStorage.setItem("tks-pwa-dismissed", Date.now().toString());
  };

  if (!showBanner) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 z-[100] mx-auto max-w-md animate-in slide-in-from-bottom-4 duration-500">
      <div className="rounded-xl border border-[#d4a843]/30 bg-[#0f1320]/95 backdrop-blur-lg p-4 shadow-2xl shadow-[#d4a843]/10">
        <button
          onClick={handleDismiss}
          className="absolute top-3 right-3 text-[#8a8070] hover:text-[#e8e0d0] transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-[#d4a843] to-[#c49535] flex items-center justify-center">
            <Smartphone className="h-5 w-5 text-[#0a0e1a]" />
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-[#e8e0d0] mb-1">
              Install TKS Tootra
            </h3>
            <p className="text-xs text-[#8a8070] mb-3 leading-relaxed">
              Add to your home screen for a native app experience with quick access and offline support.
            </p>

            {isIOS ? (
              <>
                {!showIOSGuide ? (
                  <Button
                    onClick={() => setShowIOSGuide(true)}
                    size="sm"
                    className="bg-gradient-to-r from-[#d4a843] to-[#c49535] text-[#0a0e1a] font-semibold hover:opacity-90 h-8 text-xs"
                  >
                    <Download className="h-3.5 w-3.5 mr-1.5" />
                    Show Me How
                  </Button>
                ) : (
                  <div className="space-y-2 text-xs text-[#b0a090]">
                    <div className="flex items-center gap-2">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#1a2035] text-[#d4a843] flex items-center justify-center text-[10px] font-bold">1</span>
                      <span>Tap the <strong className="text-[#e8e0d0]">Share</strong> button in Safari</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#1a2035] text-[#d4a843] flex items-center justify-center text-[10px] font-bold">2</span>
                      <span>Scroll down and tap <strong className="text-[#e8e0d0]">"Add to Home Screen"</strong></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#1a2035] text-[#d4a843] flex items-center justify-center text-[10px] font-bold">3</span>
                      <span>Tap <strong className="text-[#e8e0d0]">"Add"</strong> to install</span>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <Button
                onClick={handleInstall}
                size="sm"
                className="bg-gradient-to-r from-[#d4a843] to-[#c49535] text-[#0a0e1a] font-semibold hover:opacity-90 h-8 text-xs"
              >
                <Download className="h-3.5 w-3.5 mr-1.5" />
                Install App
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
