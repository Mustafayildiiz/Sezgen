"""Sezgen - Havalimanı Pistlerinde Yabancı Cisim (FOD) Tespiti Sistemi Ana Giriş Noktası."""
import os
import uvicorn


def main() -> None:
    host = os.getenv("SEZGEN_HOST", "0.0.0.0")
    port = int(os.getenv("SEZGEN_PORT", "8000"))
    reload_flag = os.getenv("SEZGEN_RELOAD", "false").lower() == "true"

    print(f"✈️  Sezgen FOD Tespit Sistemi başlatılıyor -> http://localhost:{port}")
    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level="info",
    )


if __name__ == "__main__":
    main()
