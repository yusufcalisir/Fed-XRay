import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  isThirdPartyExtensionError,
  installRuntimeShield,
  getRuntimeShieldTelemetry,
  resetRuntimeShieldTelemetry,
} from "../lib/runtimeShield";

describe("Runtime Shield and Extension Isolation Engine", () => {
  beforeEach(() => {
    resetRuntimeShieldTelemetry();
  });

  describe("isThirdPartyExtensionError detection", () => {
    it("detects chrome-extension and moz-extension URL schemes in stacks", () => {
      const chromeErr = new Error("Injection failed");
      chromeErr.stack = "Error: Injection failed\n    at chrome-extension://abcdefghijklm/content.js:12:34";
      const check1 = isThirdPartyExtensionError(chromeErr);
      expect(check1.isExtension).toBe(true);
      expect(check1.matchedPattern).toContain("chrome-extension://");

      const mozErr = new Error("Firefox extension crash");
      mozErr.stack = "Error: Firefox extension crash\n    at moz-extension://12345-6789/script.js:1:1";
      const check2 = isThirdPartyExtensionError(mozErr);
      expect(check2.isExtension).toBe(true);
      expect(check2.matchedPattern).toContain("moz-extension://");
    });

    it("detects known extension injected files like adblock and globals-front", () => {
      const adblockErr = {
        filename: "https://site.com/adblock-picreplacement.js",
        message: "DOM replacement failed",
      };
      const checkAdblock = isThirdPartyExtensionError(adblockErr);
      expect(checkAdblock.isExtension).toBe(true);
      expect(checkAdblock.matchedPattern).toContain("adblock");

      const globalsErr = {
        filename: "globals-front.js",
        message: "Script error",
      };
      const checkGlobals = isThirdPartyExtensionError(globalsErr);
      expect(checkGlobals.isExtension).toBe(true);
      expect(checkGlobals.matchedPattern).toContain("globals-front.js");
    });

    it("detects port disconnection message from dormant service workers", () => {
      const msg = "Could not establish connection. Receiving end does not exist.";
      const check = isThirdPartyExtensionError(msg);
      expect(check.isExtension).toBe(true);
      expect(check.matchedPattern).toContain("could not establish connection");
    });

    it("detects extension useCache undefined errors", () => {
      const msg = "Cannot read properties of undefined (reading 'useCache')";
      const check = isThirdPartyExtensionError(msg);
      expect(check.isExtension).toBe(true);
      expect(check.matchedPattern).toContain("usecache");
    });

    it("detects benign resize observer loops and DOM mutation collisions", () => {
      const resizeErr = new Error("ResizeObserver loop completed with undelivered notifications");
      expect(isThirdPartyExtensionError(resizeErr).isExtension).toBe(true);

      const domErr = new Error("Failed to execute 'removeChild' on 'Node': The node to be removed is not a child");
      expect(isThirdPartyExtensionError(domErr).isExtension).toBe(true);
    });

    it("does NOT flag genuine Fed-XRay application and API errors", () => {
      const appErr = new Error("Failed to fetch hospital cohorts from http://127.0.0.1:8000/api/cohorts");
      appErr.stack = "Error: Failed to fetch\n    at fetchHospitalCohorts (api.ts:45:10)\n    at DashboardPage.tsx:78:20";
      const check = isThirdPartyExtensionError(appErr);
      expect(check.isExtension).toBe(false);

      const typeErr = new TypeError("Cannot read properties of undefined (reading 'accuracy')");
      typeErr.stack = "TypeError: Cannot read properties of undefined\n    at TelemetryCockpit.tsx:112:30";
      expect(isThirdPartyExtensionError(typeErr).isExtension).toBe(false);
    });
  });

  describe("installRuntimeShield runtime interception", () => {
    let uninstall: (() => void) | null = null;

    afterEach(() => {
      if (uninstall) {
        uninstall();
        uninstall = null;
      }
    });

    it("intercepts extension error events and stops event propagation", () => {
      uninstall = installRuntimeShield();

      const preventDefault = vi.fn();
      const stopImmediatePropagation = vi.fn();

      const event = new ErrorEvent("error", {
        message: "Uncaught Error: Could not establish connection. Receiving end does not exist.",
        filename: "chrome-extension://xyz/content.js",
      });
      event.preventDefault = preventDefault;
      event.stopImmediatePropagation = stopImmediatePropagation;

      window.dispatchEvent(event);

      expect(preventDefault).toHaveBeenCalled();
      expect(stopImmediatePropagation).toHaveBeenCalled();

      const telemetry = getRuntimeShieldTelemetry();
      expect(telemetry.blockedCount).toBeGreaterThanOrEqual(1);
      expect(telemetry.isShieldActive).toBe(true);
    });

    it("intercepts unhandled promise rejections originating from extension port drops", () => {
      uninstall = installRuntimeShield();

      const preventDefault = vi.fn();
      const stopImmediatePropagation = vi.fn();

      const rejEvent = new Event("unhandledrejection") as PromiseRejectionEvent;
      Object.defineProperty(rejEvent, "reason", {
        value: new Error("Could not establish connection. Receiving end does not exist."),
      });
      rejEvent.preventDefault = preventDefault;
      rejEvent.stopImmediatePropagation = stopImmediatePropagation;

      window.dispatchEvent(rejEvent);

      expect(preventDefault).toHaveBeenCalled();
      expect(stopImmediatePropagation).toHaveBeenCalled();
    });

    it("intercepts and throttles console.error noise while preserving genuine app errors", () => {
      const originalConsoleError = console.error;
      const originalConsoleWarn = console.warn;
      const warnSpy = vi.fn();
      console.warn = warnSpy;

      uninstall = installRuntimeShield();

      // Log extension error to console
      console.error("adblock-picreplacement.js:813 Uncaught (in promise) Error: Could not establish connection.");

      // Telemetry should record it
      const tel = getRuntimeShieldTelemetry();
      expect(tel.blockedCount).toBeGreaterThanOrEqual(1);
      expect(warnSpy).toHaveBeenCalled();

      // Log genuine application error to console
      const realSpy = vi.fn();
      console.error = realSpy;
      console.error("Clinical audit log: CKKS decryption completed with epsilon 1.42");
      expect(realSpy).toHaveBeenCalledWith("Clinical audit log: CKKS decryption completed with epsilon 1.42");

      console.warn = originalConsoleWarn;
      console.error = originalConsoleError;
    });
  });
});
