#!/bin/bash

VIDEO=${VIDEO:-"ToS-4k-1080.y4m"}
BITRATE=500

echo "Sweep Tuning Started"
echo "Format: L0_LUMA, I_SCALE, Z_THRESH, L16, L32 | SSIM_AVG | SIZE_KB"
echo "Format: L0_LUMA, I_SCALE, Z_THRESH, L16, L32 | SSIM_AVG | SIZE_KB" > sweep_results.txt

# Initial build to avoid timing it
swift build -c release --product compare > /dev/null 2>&1

for L0 in 2 3 4; do
    for I_SCALE in 90 95 100; do
        for Z_TH in 5 6 7; do
            for L16_L32 in "3,4" "4,5" "4,6"; do
                L16=$(echo $L16_L32 | cut -d',' -f1)
                L32=$(echo $L16_L32 | cut -d',' -f2)

                export VEVC_TUNE_L0_LUMA=$L0
                export VEVC_TUNE_IFRAME_SCALE=$I_SCALE
                export VEVC_TUNE_L16_LUMA=$L16
                export VEVC_TUNE_L32_LUMA=$L32
                
                # Run and capture output
                OUT=$(swift run -c release compare -y4m "$VIDEO" -quality -bitrate $BITRATE -zeroThreshold $Z_TH)
                
                # Extract Size and SSIM for VEVC
                SIZE=$(echo "$OUT" | grep -A 5 "\[VEVC" | grep "Size" | awk '{print $3}')
                SSIM=$(echo "$OUT" | grep -A 5 "\[VEVC" | grep "SSIM" | awk -F'Avg: ' '{print $2}' | awk '{print $1}')
                
                echo "$L0, $I_SCALE, $Z_TH, $L16, $L32 | SSIM: $SSIM | Size: $SIZE"
                echo "$L0, $I_SCALE, $Z_TH, $L16, $L32 | SSIM: $SSIM | Size: $SIZE" >> sweep_results.txt
            done
        done
    done
done
