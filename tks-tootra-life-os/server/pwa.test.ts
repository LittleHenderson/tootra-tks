import { describe, expect, it } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";

describe("PWA configuration", () => {
  describe("manifest.json", () => {
    const manifestPath = resolve(__dirname, "../client/public/manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));

    it("has correct app name", () => {
      expect(manifest.name).toBe("TKS Tootra — Life Execution System");
      expect(manifest.short_name).toBe("TKS Tootra");
    });

    it("has standalone display mode", () => {
      expect(manifest.display).toBe("standalone");
    });

    it("has TKS brand colors", () => {
      expect(manifest.theme_color).toBe("#d4a843");
      expect(manifest.background_color).toBe("#0a0e1a");
    });

    it("has required icon sizes", () => {
      const sizes = manifest.icons.map((i: any) => i.sizes);
      expect(sizes).toContain("192x192");
      expect(sizes).toContain("512x512");
    });

    it("all icons have valid URLs", () => {
      for (const icon of manifest.icons) {
        expect(icon.src).toMatch(/^https:\/\//);
        expect(icon.type).toBe("image/png");
      }
    });

    it("has start_url set to root", () => {
      expect(manifest.start_url).toBe("/");
    });

    it("includes productivity category", () => {
      expect(manifest.categories).toContain("productivity");
    });
  });

  describe("offline.html", () => {
    const offlinePath = resolve(__dirname, "../client/public/offline.html");
    const offlineHtml = readFileSync(offlinePath, "utf-8");

    it("exists and contains TKS branding", () => {
      expect(offlineHtml).toContain("TKS Tootra");
      expect(offlineHtml).toContain("Est Ars Celare Artem");
    });

    it("has a retry button", () => {
      expect(offlineHtml).toContain("Try Again");
      expect(offlineHtml).toContain("window.location.reload()");
    });

    it("uses TKS brand colors", () => {
      expect(offlineHtml).toContain("#d4a843");
      expect(offlineHtml).toContain("#0a0e1a");
    });
  });

  describe("service worker (sw.js)", () => {
    const swPath = resolve(__dirname, "../client/public/sw.js");
    const swCode = readFileSync(swPath, "utf-8");

    it("defines a cache name", () => {
      expect(swCode).toContain("CACHE_NAME");
      expect(swCode).toContain("tks-tootra-v1");
    });

    it("handles install event", () => {
      expect(swCode).toContain('addEventListener("install"');
    });

    it("handles activate event with cache cleanup", () => {
      expect(swCode).toContain('addEventListener("activate"');
      expect(swCode).toContain("caches.delete");
    });

    it("handles fetch event with network-first for navigation", () => {
      expect(swCode).toContain('addEventListener("fetch"');
      expect(swCode).toContain('request.mode === "navigate"');
    });

    it("skips API requests", () => {
      expect(swCode).toContain('url.pathname.startsWith("/api/")');
    });

    it("pre-caches offline page", () => {
      expect(swCode).toContain("/offline.html");
    });

    it("uses stale-while-revalidate for static assets", () => {
      expect(swCode).toContain("js|css|png|jpg|jpeg|svg|gif|webp");
    });
  });

  describe("index.html PWA integration", () => {
    const indexPath = resolve(__dirname, "../client/index.html");
    const indexHtml = readFileSync(indexPath, "utf-8");

    it("links to manifest.json", () => {
      expect(indexHtml).toContain('rel="manifest"');
      expect(indexHtml).toContain("manifest.json");
    });

    it("has theme-color meta tag", () => {
      expect(indexHtml).toContain('name="theme-color"');
      expect(indexHtml).toContain("#d4a843");
    });

    it("has iOS PWA meta tags", () => {
      expect(indexHtml).toContain("apple-mobile-web-app-capable");
      expect(indexHtml).toContain("apple-mobile-web-app-status-bar-style");
      expect(indexHtml).toContain("apple-mobile-web-app-title");
      expect(indexHtml).toContain("apple-touch-icon");
    });

    it("registers service worker", () => {
      expect(indexHtml).toContain("serviceWorker");
      expect(indexHtml).toContain("register('/sw.js')");
    });

    it("has Open Graph meta tags", () => {
      expect(indexHtml).toContain('property="og:title"');
      expect(indexHtml).toContain('property="og:description"');
      expect(indexHtml).toContain('property="og:image"');
    });
  });
});
