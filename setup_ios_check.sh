#!/bin/bash
# setup_ios_check.sh
# iPhone セキュリティ診断ツール セットアップスクリプト
#
# 使い方:
#   1. このファイルと iphone_security_check.py、requirements_ios_check.txt を
#      ~/Downloads に置く
#   2. ターミナルで以下を実行:
#      cd ~/Downloads && bash setup_ios_check.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOADS="$HOME/Downloads"

echo ""
echo "========================================"
echo "  iPhone Security Check セットアップ"
echo "========================================"
echo ""

# ────────────────────────────────────────────
# ステップ 1: Homebrew
# ────────────────────────────────────────────
echo "[1/3] Homebrew を確認中..."

if ! command -v brew &>/dev/null; then
    echo "  Homebrew が見つかりません。インストールします..."
    echo "  ※ パスワードの入力を求められる場合があります"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Apple Silicon Mac の場合、brew のパスを通す
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    echo "  [OK] Homebrew インストール完了"
else
    echo "  [OK] Homebrew は既にインストール済みです ($(brew --version | head -1))"
fi

# ────────────────────────────────────────────
# ステップ 2: Python3 + pip3
# ────────────────────────────────────────────
echo ""
echo "[2/3] Python3 を確認中..."

if ! command -v python3 &>/dev/null || ! brew list python3 &>/dev/null 2>&1; then
    echo "  Python3 をインストールします..."
    brew install python3
    echo "  [OK] Python3 インストール完了"
else
    echo "  [OK] Python3 は既にインストール済みです ($(python3 --version))"
fi

# pip3 の確認
if ! command -v pip3 &>/dev/null; then
    echo "  [エラー] pip3 が見つかりません。"
    echo "  brew install python3 を再試行してください。"
    exit 1
fi
echo "  [OK] pip3 確認済み ($(pip3 --version))"

# ────────────────────────────────────────────
# ステップ 3: 依存ライブラリのインストール
# ────────────────────────────────────────────
echo ""
echo "[3/3] 依存ライブラリをインストール中..."

REQUIREMENTS="$SCRIPT_DIR/requirements_ios_check.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    echo "  requirements_ios_check.txt が見つかりません。"
    echo "  iphone_security_check.py と同じフォルダに置いてください。"
    exit 1
fi

pip3 install -r "$REQUIREMENTS"
echo "  [OK] ライブラリのインストール完了"

# ────────────────────────────────────────────
# 完了メッセージ
# ────────────────────────────────────────────
echo ""
echo "========================================"
echo "  セットアップ完了！"
echo "========================================"
echo ""
echo "  診断を開始するには以下を実行してください:"
echo ""
echo "    cd ~/Downloads"
echo "    python3 iphone_security_check.py"
echo ""
echo "  ※ iPhoneをUSB-Cケーブルで接続してから実行してください"
echo "  ※ iPhoneの画面で「このコンピュータを信頼」をタップしてください"
echo ""
