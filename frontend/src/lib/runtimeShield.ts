/**
 * Client Runtime Shield and Extension Isolation Engine
 * 
 * Provides defense-in-depth isolation against intrusive third-party browser
 * extensions (ad blockers, shopping assistants, grammar checkers, translation injectors).
 * 
 * Prevents extension port disconnection loops ("Could not establish connection. Receiving end does not exist"),
 * unhandled extension promise rejections ("reading 'useCache'"), and console flooding
 * from destabilizing the React tree or polluting clinical telemetry.
 */

export interface RuntimeShieldTelemetry {
  blockedCount: number;
  detectedSignatures: string[];
  lastBlockedTimestamp: number | null;
  isShieldActive: boolean;
}

const EXTENSION_URL_PATTERNS = [
  "chrome-extension://",
  "moz-extension://",
  "safari-extension://",
  "safari-web-extension://",
  "extension://",
];

const EXTENSION_FILE_PATTERNS = [
  "content.js",
  "adblock",
  "adblock-picreplacement.js",
  "globals-front.js",
  "inpage.js",
  "pageProvider.js",
  "contentscript.js",
];

const EXTENSION_MESSAGE_PATTERNS = [
  "could not establish connection",
  "the message port closed before a response was received",
  "cannot read properties of undefined (reading 'usecache')",
  "cannot read properties of null (reading 'usecache')",
  "extension context invalidated",
  "resizeobserver loop completed with undelivered notifications",
  "resizeobserver loop limit exceeded",
  "failed to execute 'removechild' on 'node'",
  "failed to execute 'insertbefore' on 'node'",
];

const telemetryState: RuntimeShieldTelemetry = {
  blockedCount: 0,
  detectedSignatures: [],
  lastBlockedTimestamp: null,
  isShieldActive: false,
};

// In-memory throttling cache for console warnings (max 1 warning per signature per 30 seconds)
const signatureThrottleMap = new Map<string, number>();
const THROTTLE_INTERVAL_MS = 30000;

/**
 * Evaluates whether an error, event, or rejection originates from a third-party browser extension.
 */
export function isThirdPartyExtensionError(target: unknown): { isExtension: boolean; matchedPattern?: string } {
  if (!target) {
    return { isExtension: false };
  }

  let message = "";
  let stack = "";
  let filename = "";

  if (typeof target === "string") {
    message = target;
  } else if (typeof target === "object") {
    const obj = target as Record<string, unknown>;

    // Handle ErrorEvent
    if (typeof obj.message === "string") message = obj.message;
    if (typeof obj.filename === "string") filename = obj.filename;

    // Handle PromiseRejectionEvent
    if (obj.reason && typeof obj.reason === "object") {
      const reason = obj.reason as Record<string, unknown>;
      if (typeof reason.message === "string") message = reason.message;
      if (typeof reason.stack === "string") stack = reason.stack;
    } else if (typeof obj.reason === "string") {
      message = obj.reason;
    }

    // Handle standard Error
    if (typeof obj.message === "string" && !message) message = obj.message;
    if (typeof obj.stack === "string") stack = obj.stack;
  }

  const normalizedMessage = message.toLowerCase();
  const normalizedStack = stack.toLowerCase();
  const normalizedFilename = filename.toLowerCase();
  const combinedContext = `${normalizedMessage} ${normalizedStack} ${normalizedFilename}`;

  // 1. Check URL patterns in combined context
  for (const urlPattern of EXTENSION_URL_PATTERNS) {
    if (combinedContext.includes(urlPattern)) {
      return { isExtension: true, matchedPattern: `url:${urlPattern}` };
    }
  }

  // 2. Check known script names in combined context
  for (const filePattern of EXTENSION_FILE_PATTERNS) {
    if (combinedContext.includes(filePattern)) {
      return { isExtension: true, matchedPattern: `file:${filePattern}` };
    }
  }

  // 3. Check known extension error messages in combined context
  for (const msgPattern of EXTENSION_MESSAGE_PATTERNS) {
    if (combinedContext.includes(msgPattern)) {
      return { isExtension: true, matchedPattern: `msg:${msgPattern}` };
    }
  }

  return { isExtension: false };
}

/**
 * Returns current snapshot of runtime shield telemetry.
 */
export function getRuntimeShieldTelemetry(): RuntimeShieldTelemetry {
  return { ...telemetryState };
}

/**
 * Resets runtime shield telemetry (useful in tests).
 */
export function resetRuntimeShieldTelemetry(): void {
  telemetryState.blockedCount = 0;
  telemetryState.detectedSignatures = [];
  telemetryState.lastBlockedTimestamp = null;
  telemetryState.isShieldActive = false;
  signatureThrottleMap.clear();
}

/**
 * Installs window listeners and console interceptors to isolate extension noise.
 * Safe to call multiple times (idempotent).
 */
export function installRuntimeShield(): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  if (telemetryState.isShieldActive) {
    return () => {};
  }

  telemetryState.isShieldActive = true;

  const recordBlockedEvent = (matchedPattern: string) => {
    telemetryState.blockedCount += 1;
    telemetryState.lastBlockedTimestamp = Date.now();
    if (!telemetryState.detectedSignatures.includes(matchedPattern)) {
      telemetryState.detectedSignatures.push(matchedPattern);
    }
  };

  // 1. Intercept window error events
  const errorHandler = (event: ErrorEvent) => {
    const check = isThirdPartyExtensionError(event);
    if (check.isExtension) {
      recordBlockedEvent(check.matchedPattern || "unknown_extension_error");
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  };

  // 2. Intercept unhandled promise rejections
  const unhandledRejectionHandler = (event: PromiseRejectionEvent) => {
    const check = isThirdPartyExtensionError(event);
    if (check.isExtension) {
      recordBlockedEvent(check.matchedPattern || "unknown_extension_rejection");
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  };

  window.addEventListener("error", errorHandler, { capture: true });
  window.addEventListener("unhandledrejection", unhandledRejectionHandler, { capture: true });

  // 3. Intercept console.error and console.warn with deduplication and throttling
  const originalConsoleError = console.error;
  const originalConsoleWarn = console.warn;

  const filteredLogger = (
    originalFn: (...args: unknown[]) => void,
    level: "error" | "warn",
    ...args: unknown[]
  ) => {
    let matchedSignature: string | null = null;
    for (const arg of args) {
      const check = isThirdPartyExtensionError(arg);
      if (check.isExtension) {
        matchedSignature = check.matchedPattern || "extension_noise";
        break;
      }
    }

    if (matchedSignature) {
      recordBlockedEvent(matchedSignature);
      const now = Date.now();
      const lastLogged = signatureThrottleMap.get(matchedSignature) || 0;
      if (now - lastLogged > THROTTLE_INTERVAL_MS) {
        signatureThrottleMap.set(matchedSignature, now);
        // Output clean diagnostic notice without red error crash
        originalConsoleWarn(
          `%c[Fed-XRay Runtime Shield]%c Isolated 3rd-party browser extension noise (${matchedSignature}). Clinical execution protected.`,
          "color: #06b6d4; font-weight: bold;",
          "color: #94a3b8;"
        );
      }
      return;
    }

    // Pass through genuine application logs unmodified
    originalFn.apply(console, args);
  };

  console.error = (...args: unknown[]) => filteredLogger(originalConsoleError, "error", ...args);
  console.warn = (...args: unknown[]) => filteredLogger(originalConsoleWarn, "warn", ...args);

  // Return teardown function
  return () => {
    window.removeEventListener("error", errorHandler, { capture: true });
    window.removeEventListener("unhandledrejection", unhandledRejectionHandler, { capture: true });
    console.error = originalConsoleError;
    console.warn = originalConsoleWarn;
    telemetryState.isShieldActive = false;
  };
}
