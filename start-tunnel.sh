#!/bin/bash
# ── ngrok Tunnel — Chainlit'i public URL ile paylas ─────────────────────────
#
# Kullanim:
#   ./start-tunnel.sh              → varsayilan port (APP_PORT veya 7860)
#   ./start-tunnel.sh 8000         → ozel port
#   TUNNEL_PROVIDER=localtunnel ./start-tunnel.sh   → SSH-tabanli, hesap gerektirmez
#
# Gereksinim:
#   ngrok (varsayilan)  — https://ngrok.com/signup adresinden ucretsiz hesap ac,
#                         NGROK_AUTHTOKEN=xxxx satirini .env dosyana ekle.
#   localtunnel         — Sadece SSH gerekli; herhangi bir kurulum yok.
#
# ngrok neden Cloudflare'den daha saglamdir?
#   - Persistent subdomain destegi (ucretli) veya sabit URL
#   - Dahili istek/yanit inceleme arayuzu: http://localhost:4040
#   - Daha kararsiz ag kosullarinda otomatik yeniden baglanti
#   - Kapsamli hata raporlama ve HTTP replay ozelligi
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# .env varsa yukle
if [[ -f .env ]]; then
  set -a
  source <(grep -v '^\s*#' .env | grep -v '^\s*$')
  set +a
fi

APP_PORT="${1:-${APP_PORT:-7860}}"
PROVIDER="${TUNNEL_PROVIDER:-ngrok}"

# ════════════════════════════════════════════════════════════════════════════
#  SAGLAYICI: localtunnel (SSH, sifir kurulum, sifir hesap)
# ════════════════════════════════════════════════════════════════════════════
if [[ "$PROVIDER" == "localtunnel" ]]; then
  echo "──────────────────────────────────────────────────────"
  echo "  localhost.run — SSH tabanli tunel (hesap gerekmez)"
  echo "  Local : http://localhost:${APP_PORT}"
  echo ""
  echo "  Birkaç saniye sonra asagidaki gibi bir URL goreceksin:"
  echo "  https://<random>.lhr.life"
  echo ""
  echo "  Durdurmak icin: Ctrl+C"
  echo "──────────────────────────────────────────────────────"
  exec ssh \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -R "80:localhost:${APP_PORT}" \
    localhost.run
fi

# ════════════════════════════════════════════════════════════════════════════
#  SAGLAYICI: ngrok (varsayilan)
# ════════════════════════════════════════════════════════════════════════════

NGROK_BIN="$(command -v ngrok 2>/dev/null || echo "")"

# ── ngrok yoksa indir ────────────────────────────────────────────────────────
if [[ -z "$NGROK_BIN" ]]; then
  echo "ngrok bulunamadi — indiriliyor..."

  ARCH="$(uname -m)"
  OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

  case "$ARCH" in
    x86_64)  NG_ARCH="amd64" ;;
    aarch64) NG_ARCH="arm64" ;;
    armv7l)  NG_ARCH="arm"   ;;
    *)
      echo "Desteklenmeyen mimari: $ARCH"
      echo "Manuel indirme: https://ngrok.com/download"
      exit 1
      ;;
  esac

  INSTALL_DIR="$HOME/.local/bin"
  mkdir -p "$INSTALL_DIR"
  NGROK_BIN="$INSTALL_DIR/ngrok"

  TMP_TGZ="$(mktemp /tmp/ngrok-XXXX.tgz)"
  NG_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-${OS}-${NG_ARCH}.tgz"
  echo "Indiriliyor: $NG_URL"
  curl -fsSL "$NG_URL" -o "$TMP_TGZ"
  tar -xzf "$TMP_TGZ" -C "$INSTALL_DIR" ngrok
  chmod +x "$NGROK_BIN"
  rm -f "$TMP_TGZ"
  echo "Kuruldu: $NGROK_BIN"
  echo ""
  echo "PATH'a eklemek icin:"
  echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
  echo ""
fi

# ── Auth token kontrolu ──────────────────────────────────────────────────────
if [[ -n "${NGROK_AUTHTOKEN:-}" ]]; then
  "$NGROK_BIN" config add-authtoken "$NGROK_AUTHTOKEN" --log=false 2>/dev/null || true
else
  echo "╔════════════════════════════════════════════════════════╗"
  echo "║  UYARI: NGROK_AUTHTOKEN ayarlanmamis.                  ║"
  echo "║                                                         ║"
  echo "║  1. https://ngrok.com/signup — ucretsiz hesap ac        ║"
  echo "║  2. Dashboard > Your Authtoken kopyala                  ║"
  echo "║  3. .env dosyasina ekle:  NGROK_AUTHTOKEN=xxxx          ║"
  echo "║                                                         ║"
  echo "║  Alternatif (hesap gerektirmez):                        ║"
  echo "║    TUNNEL_PROVIDER=localtunnel ./start-tunnel.sh        ║"
  echo "╚════════════════════════════════════════════════════════╝"
  echo ""
  echo "Token olmadan devam ediliyor (baglanti kisitli olabilir)..."
  echo ""
fi

echo "──────────────────────────────────────────────────────"
echo "  ngrok — Chainlit'i public URL ile paylasiyor..."
echo "  Local  : http://localhost:${APP_PORT}"
echo "  Binary : $NGROK_BIN"
echo ""
echo "  URL terminalde gorunecek. Ayrica:"
echo "  Inspector: http://localhost:4040  (istek/yanit kayitlari)"
echo ""
echo "  Durdurmak icin: Ctrl+C"
echo "──────────────────────────────────────────────────────"

exec "$NGROK_BIN" http "${APP_PORT}" \
  --log=stdout \
  --log-format=logfmt \
  --log-level=info
