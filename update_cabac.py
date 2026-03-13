import re

with open("Sources/vevc/CABAC.swift", "r") as f:
    content = f.read()

# Replace rangeLPS_table
old_table = """let rangeLPS_table: [[UInt32]] = [
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    }
]"""

new_table = """let rangeLPS_table: [UInt32] = {
    var table = [UInt32](repeating: 0, count: 512)
    for q in 0..<4 {
        for s in 0..<128 {
            let p = pow(0.5, Double(s) / 10.0)
            table[q * 128 + s] = UInt32(max(1, min(255, Int(p * 256.0))))
        }
    }
    return table
}()"""

content = content.replace(old_table, new_table)

# Replace table accesses
content = content.replace("rangeLPS_table[Int(qIdx)][Int(ctx.pStateIdx)]", "rangeLPS_table[Int(qIdx) * 128 + Int(ctx.pStateIdx)]")

with open("Sources/vevc/CABAC.swift", "w") as f:
    f.write(content)
