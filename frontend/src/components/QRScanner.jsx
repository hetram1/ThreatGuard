import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Camera, ImagePlus, ShieldCheck } from "lucide-react";
import { Html5Qrcode } from "html5-qrcode";

export default function QRScanner({ onAnalyzeUrl }) {
  const scannerRef = useRef(null);
  const fileInputRef = useRef(null);

  const [scanning, setScanning] =
    useState(false);

  const [decodedValue, setDecodedValue] =
    useState("");

  const [error, setError] =
    useState("");

  async function stopScanner() {
    const scanner = scannerRef.current;

    if (!scanner) {
      setScanning(false);
      return;
    }

    try {
      await scanner.stop();
    } catch (err) {
      console.warn(
        "QR scanner stop:",
        err,
      );
    }

    try {
      await scanner.clear();
    } catch (err) {
      console.warn(
        "QR scanner clear:",
        err,
      );
    }

    scannerRef.current = null;
    setScanning(false);
  }

  async function handleFileSelect(event) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    setError("");
    setDecodedValue("");

    try {
      await stopScanner();

      const scanner =
        new Html5Qrcode(
          "threatguard-qr-reader",
        );

      scannerRef.current =
        scanner;

      const decodedText =
        await scanner.scanFile(
          file,
          true,
        );

      const value =
        decodedText.trim();

      setDecodedValue(value);

      try {
        await scanner.clear();
      } catch (err) {
        console.warn(
          "QR file scanner clear:",
          err,
        );
      }

      scannerRef.current = null;

      if (
        /^https?:\/\//i.test(value)
      ) {
        await onAnalyzeUrl(value);
      } else {
        setError(
          "QR code decoded successfully, but it does not contain a valid HTTP/HTTPS URL.",
        );
      }
    } catch (err) {
      console.error(
        "QR file scan error:",
        err,
      );

      scannerRef.current = null;

      setError(
        "Unable to decode a QR code from that image. Select a clear QR image and try again.",
      );
    } finally {
      event.target.value = "";
    }
  }


  async function startScanner() {
    setError("");
    setDecodedValue("");

    try {
      const scanner =
        new Html5Qrcode(
          "threatguard-qr-reader",
        );

      scannerRef.current =
        scanner;

      await scanner.start(
        {
          facingMode:
            "environment",
        },
        {
          fps: 10,
          qrbox: {
            width: 250,
            height: 250,
          },
          aspectRatio: 1,
        },
        async (decodedText) => {
          const value =
            decodedText.trim();

          setDecodedValue(value);

          await stopScanner();

          if (
            /^https?:\/\//i.test(value)
          ) {
            onAnalyzeUrl(value);
          } else {
            setError(
              "QR code decoded successfully, but it does not contain a valid HTTP/HTTPS URL.",
            );
          }
        },
        () => {},
      );

      setScanning(true);
    } catch (err) {
      console.error(
        "QR scanner error:",
        err,
      );

      scannerRef.current = null;
      setScanning(false);

      setError(
        "Unable to start the camera. Check browser camera permission and try again.",
      );
    }
  }

  useEffect(() => {
    return () => {
      const scanner =
        scannerRef.current;

      if (scanner) {
        scanner
          .stop()
          .catch(() => {});

        scanner
          .clear()
          .catch(() => {});

        scannerRef.current = null;
      }
    };
  }, []);

  return (
    <div className="qr-scanner-panel">

      <div className="qr-reader-shell">
        <div
          id="threatguard-qr-reader"
        />
      </div>


      <div className="qr-actions">

        <input
          ref={fileInputRef}
          className="qr-file-input"
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
        />

        <button
          className="analyze-button"
          onClick={() =>
            fileInputRef.current?.click()
          }
          disabled={scanning}
        >
          <ImagePlus size={17} />
          Select QR from Device
        </button>

        {!scanning ? (
          <button
            className="analyze-button"
            onClick={startScanner}
          >
            <Camera size={17} />
            Start Camera
          </button>
        ) : (
          <button
            className="analyze-button"
            onClick={stopScanner}
          >
            Stop Scanner
          </button>
        )}

      </div>


      {decodedValue && (
        <div className="qr-result">

          <div className="result-kicker">
            DECODED CONTENT
          </div>

          <code>
            {decodedValue}
          </code>

        </div>
      )}


      {error && (
        <div className="error-banner">

          <AlertTriangle size={18} />

          {error}

        </div>
      )}


      <div className="qr-safety-note">

        <ShieldCheck size={17} />

        <span>
          ThreatGuard can decode QR codes
          from your camera or an image file.
          It analyzes the URL as text and never
          opens or visits the destination.
        </span>

      </div>

    </div>
  );
}


