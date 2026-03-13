import re

with open("Sources/vevc/Encode.swift", "r") as f:
    content = f.read()

old_log = """            appendUInt32BE(&out, UInt32(mvOut.count))
            out.append(contentsOf: mvOut)

            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            debugLog("[Frame \(i)] P-Frame: \(bytes.count) bytes (\(String(format: "%.2f", Double(bytes.count) / 1024.0)) KB) MVs=\(mvs.dx.count) meanSAD=\(meanSAD)")"""

new_log = """            appendUInt32BE(&out, UInt32(mvOut.count))
            out.append(contentsOf: mvOut)

            appendUInt32BE(&out, UInt32(bytes.count))
            out.append(contentsOf: bytes)
            let totalBytes = bytes.count + mvOut.count
            debugLog("[Frame \(i)] P-Frame: \(totalBytes) bytes (MV: \(mvOut.count) bytes, Data: \(bytes.count) bytes) MVs=\(mvs.dx.count) meanSAD=\(meanSAD) [PMV & LSCP applied]")"""

content = content.replace(old_log, new_log)

with open("Sources/vevc/Encode.swift", "w") as f:
    f.write(content)
