#!/bin/bash
# Run this on your PC to push HvrTwk to its own repo.
# Usage: bash setup-hvrtwk-repo.sh

set -e

echo "Cloning HvrTwk repo..."
git clone https://github.com/LittleHenderson/HvrTwk.git /tmp/HvrTwk-push
cd /tmp/HvrTwk-push

echo "Fetching files from tootra-tks..."
git remote add tks https://github.com/LittleHenderson/tootra-tks.git
git fetch tks claude/text-to-speech-hover-y3QwC

# Checkout just the HvrTwk files
git checkout tks/claude/text-to-speech-hover-y3QwC -- \
  hvrtwk.js \
  hvrtwk.html \
  hvrtwk-install.html \
  hvrtwk-android/ \
  .github/workflows/build-hvrtwk-apk.yml

# Copy README and LICENSE (they're only in the standalone prep)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 LittleHenderson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
EOF

cat > README.md << 'READMEEOF'
# HvrTwk

Text-to-speech overlay that works on any webpage. Hover on desktop. Circle on phone. Android app included.

## What It Does

**Desktop:** Hover your pointer over any text for 1.5 seconds -- a progress ring fills, then the text is read aloud.

**Phone:** Touch and hold, then draw a circle around the text you want read. Lift your finger -- the circled text is spoken aloud.

**Continuous mode:** Toggle it on, select any paragraph, and HvrTwk reads from there through the rest of the page. Auto-scrolls, highlights the current paragraph, shows a progress bar.

## Android App (APK)

A standalone app with a built-in browser. HvrTwk is always active on every page you visit.

### Get the APK

1. Go to the **Actions** tab above
2. Click the latest **Build HvrTwk APK** run
3. Download the **HvrTwk-debug** artifact
4. Install the APK on your phone

### App Features

- **Browse any website** with HvrTwk always active
- **Paste mode** -- tap clipboard icon to read any copied text
- **Share to HvrTwk** -- from any app, Share > HvrTwk
- Circle-to-speak, continuous reading, karaoke highlighting
- Android chunked speech (avoids Chrome 15s cutoff)
- Zero dependencies, pure Web Speech API

## Requirements

- **Android app:** Android 7.0+
- **Browser:** Chrome, Edge, Safari, Firefox
- **Voices:** Settings > System > Text-to-speech output
READMEEOF

git add -A
git commit -m "HvrTwk v1.0 -- Text-to-speech overlay + Android app"
git push origin main

echo ""
echo "Done! HvrTwk is live at https://github.com/LittleHenderson/HvrTwk"
echo "GitHub Actions will now build the APK automatically."
echo "Go to Actions tab to download it."

# Cleanup
rm -rf /tmp/HvrTwk-push
