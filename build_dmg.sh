#!/bin/zsh
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$root/build/dmg-root"
ditto "$root/dist/AI Foto.app" "$root/build/dmg-root/AI Foto.app"
ln -sfn /Applications "$root/build/dmg-root/Applications"
hdiutil create \
  -volname "AI Foto" \
  -srcfolder "$root/build/dmg-root" \
  -format UDZO \
  -imagekey zlib-level=1 \
  -ov "$root/dist/AI Foto.dmg"
