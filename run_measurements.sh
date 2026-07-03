#!/bin/bash
set -e

echo "Building release..."
swift build -c release

bitrates=(500 1000 1500 2500 3500 4500)

for br in "${bitrates[@]}"; do
    echo "======================================"
    echo "Running bitrate: ${br}"
    swift run -c release compare -y4m /Users/octu0/Downloads/ToS-4k-1080.y4m -quality -vevc-only -bitrate ${br} > "output_${br}.log" 2>&1
    grep -E "Size   :|SSIM   :" "output_${br}.log"
done
