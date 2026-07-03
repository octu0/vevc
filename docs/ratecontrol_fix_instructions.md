# RateControl 修正指示書

## 0. 背景とゴール

背景分析は `docs/ratecontrol_analysis.md` を必ず先に読むこと。要旨:

- レート制御が open-loop（実出力サイズが次の量子化判断に反映されない）のため、出力サイズが目標の約 2.4×（高ビットレートでは 3.4×）に恒常的にオーバーシュートしている。
- I-frame の量子化ステップ `q` が整数（1,2,3,...）のため、高ビットレートでは q の整数遷移がサイズの階段（プラトーと崖）を作っている。P-frame の QStep が `[baseStep, 4×baseStep]` にクランプされているため、同一 q 区間内ではビットレートを変えても出力がバイト単位で同一になる。

**ゴール**: 目標ビットレートに対して出力サイズがほぼ線形（±20% 以内）に追従し、サイズの階段（プラトー）が消えること。**同一の実出力サイズにおける SSIM を悪化させないこと**（後述の公平な比較ルール参照）。

## 1. 評価方法（全 Phase 共通）

### ビルドと実行

```sh
swift build -c release
swift run -c release compare -y4m /Users/octu0/Downloads/ToS-4k-1080.y4m -quality -vevc-only -bitrate 1500
```

出力の以下の行を記録する:

```
  Size   : xxxx.xx KB (...)
  SSIM   : Avg: 0.xxxx | ...
```

### 目標サイズの計算

テストクリップは 1802 フレーム。compare のデフォルト `framerate=60`, `keyint=30` が使われるため、RateController 上の動画長は 1802/60 ≈ 30.03 秒。

```
目標サイズ [KB] = bitrate[kbps] × 1000 × (1802/60) / 8 / 1024 ≈ bitrate × 3.667
```

### 測定ポイントとベースライン

各 Phase 完了ごとに `-bitrate` を変えて 6 点測定する（他の引数は同一）:

| bitrate (kbps) | 目標サイズ (KB) | 修正前 Size (KB) | 修正前 SSIM |
|---|---|---|---|
| 500  | 1,833  | 4,125  | 0.9107 |
| 1000 | 3,667  | 8,901  | 0.9359 |
| 1500 | 5,500  | 14,400 | 0.9445 |
| 2500 | 9,168  | 24,308 | 0.9622 |
| 3500 | 12,834 | 34,403 | 0.9689 |
| 4500 | 16,500 | 55,479 | 0.9773 |

（修正前の値は `docs/bitrate_sweep.csv` より。手元の実行値と多少ずれる場合は、自分で取り直したベースラインを基準にする）

### SSIM の公平な比較ルール（重要）

修正後は同じターゲットビットレートでの実サイズが約 1/2.5 になるため、**同一ターゲットでの SSIM は必ず下がる。これは正しい挙動であり、リグレッションではない。** SSIM を理由にサイズを再び膨らませる方向の「修正」をしてはならない。

正しい判定は「同一の**実サイズ**での SSIM 比較」で行う:

1. 修正後の bitrate B での実サイズ S を得る
2. 上のベースライン表（または `docs/bitrate_sweep.csv` の VEVC 行）から実サイズが S に最も近い行を探し、その SSIM と比較する
3. 修正後 SSIM がその値より 0.003 以上低ければ品質リグレッションとみなし原因を調査する

例: 修正後 bitrate 4000 で実サイズ ≈ 14,500 KB になった場合、ベースラインの 1500 kbps 行（14,400 KB / SSIM 0.9445）と比較する。

### 合格基準（最終）

1. `swift build -c release` が警告なくビルドでき、`swift test` が全て通る（Phase 2 で意図的に仕様変更するテストは修正してよい。詳細は Phase 2 参照）
2. 6 測定点すべてで実サイズが目標の ±20% 以内。特に 1500 kbps は 5,500 KB ±15%
3. サイズが bitrate に対して単調増加で、隣接測定点（500 kbps 差）のサイズ差が 2% 未満となる「プラトー」が存在しない
4. SSIM が上記の同一実サイズ比較で悪化していない
5. compare がエラーなく完走する（SSIM 計算はデコード結果を使うため、デコード互換の破壊は SSIM 崩壊として現れる。SSIM が 0.5 以下などの異常値になったらエンコーダとデコーダの逆量子化の不一致を疑う）

## 2. 作業の進め方

- Phase 1 → 2 → 3 の順に実施。**各 Phase を独立したコミットにし、Phase ごとに上記 6 点測定を実施して結果をコミットメッセージまたは PR 説明に記録する**
- Phase 1 だけでもオーバーシュートは大きく改善するはず。Phase 2 が最も変更範囲が広いので、Phase 1 の結果を確認してから着手する
- 迷ったら「エンコーダの再構成パスとデコーダは同一の関数を共有しており、絶対に挙動を分岐させない」を原則にする

### やってはいけないこと

- `Quant.swift` の帯域比率（qMidNum/qHighNum 等）、deadZoneBias 値、AQTable のスケール係数の変更（品質チューニングは本タスクのスコープ外）
- エントロピーコーデック（rANS.swift, EntropyCodec.swift, EncodeTransform.swift）の変更
- 浮動小数点の導入（このプロジェクトのコーデックパスは整数演算のみで統一されている。固定小数点を使うこと）
- SSIM を上げる目的でサイズを増やす方向の調整

---

## Phase 1: closed-loop フィードバック（エンコーダのみ、ビットストリーム変更なし）

### 目的

GOP ごとに「実際に消費したビット / 予算」の比を EMA で追跡し、次の I-frame の量子化ステップ推定に乗算補正する。これにより R-Q モデル誤差・サンプリングバイアス由来の恒常的オーバーシュート（約 2.4×）が収束する。

### 変更 1: `Sources/vevc/RateController.swift`

状態を追加:

```swift
// Closed-loop rate correction gain in Q8 fixed-point (256 = 1.0).
// actual/budget consumption ratio of past GOPs, tracked by EMA.
private(set) var rateGainQ8: Int = 256
```

`beginGOP()` の**先頭**（`gopTargetBits` を上書きする前）に追加:

```swift
// Update closed-loop gain from the previous GOP's actual consumption.
// Scene-change may cut a GOP short, so compare against the budget
// prorated by the number of frames actually consumed.
let framesUsed = self.keyint - self.gopRemainingFrames
if 0 < self.gopTargetBits && 0 < framesUsed {
    let expected = max(1, (self.gopTargetBits * framesUsed) / self.keyint)
    let consumed = self.gopTargetBits - self.gopRemainingBits
    if 0 < consumed {
        let instantQ8 = (consumed << 8) / expected
        let clamped = max(64, min(2048, instantQ8))   // [0.25x, 8.0x]
        // EMA: gain = gain * 0.75 + instant * 0.25
        self.rateGainQ8 = max(64, min(2048, ((self.rateGainQ8 * 3) + clamped) / 4))
    }
}
```

注意: 既存の `carryOver` 計算は `gopRemainingBits` を参照するので、上記コードは既存処理より**前**に置き、既存処理は変更しない。

### 変更 2: `Sources/vevc/Encoder.swift` の `estimateQuantization`

シグネチャに補正ゲインを追加:

```swift
private func estimateQuantization(img: YCbCrImage, targetBits: Int, rateGainQ8: Int = 256) -> QuantizationTable
```

`predictedStep64` の計算直後（line 421 付近）、clamp の前に補正を挿入:

```swift
let correctedStep64 = (predictedStep64 * Int64(rateGainQ8)) >> 8
let q = min(256, Int(max(1, correctedStep64)))
```

呼び出し側（`encodeFrame` 内 line 183）:

```swift
let baseQt = estimateQuantization(img: image, targetBits: targetBits, rateGainQ8: rateController.rateGainQ8)
```

### 変更 3（推奨・デバッグ用）: GOP ログ

`EncoderTuning`（Encoder.swift line 548 付近）の env 機構に倣い、環境変数 `VEVC_RC_LOG=1` のとき GOP ごとに 1 行出力する:

```
[RC] gop=12 q=48 gain=612 target=750000 consumed=1830000 (244%)
```

出力箇所は `beginGOP` 呼び出しの直後と直前が実装しやすい。検証が終わっても残してよい（デフォルト無効）。

### Phase 1 の期待結果

- 6 点すべてでサイズが目標に近づく（±30% 程度まで収束すれば成功。整数 q の粒度が残るため完全には収束しない）
- 低〜中ビットレート（500〜1500）はほぼ目標付近に乗るはず
- 高ビットレート（3500〜4500）は q の整数粒度により依然として階段が残る → Phase 2 で解消
- rateGainQ8 が発振（GOP ごとに大きく往復）していないか `VEVC_RC_LOG=1` で確認する。発振する場合は EMA の重みを 7/8 に強める

---

## Phase 2: 量子化ステップの Q4 固定小数点化（ビットストリーム変更あり）

### 目的

baseStep を Q4 固定小数点（値 16 = 1.0、最小刻み 1/16）にし、q=3→2→1 のような 50〜100% の不連続を 6% 程度の連続的な変化にする。これで高ビットレート側の階段・プラトーが消える。

### 設計原則

- **「step」の値はすべて Q4 に統一する**（部分的に Q4/実数が混在するのが最悪のバグ源）。`Quantizer.step`、`QuantizationTable.step`、ビットストリーム上の 2 バイト step フィールド、`RateController` が扱う qStep、すべて Q4
- 例外: DeblockingFilter の `qStep` 引数と、閾値ヒューリスティクスに使う箇所は「実数値スケール」の整数を期待しているため、`(stepQ4 + 8) >> 4` で変換して渡す（下のチェックリストに全箇所を列挙）
- baseStep の Q4 での範囲: `[16, 4096]`（実数 1.0〜256.0。現行の `[1, 256]` と同じ範囲）

### 変更チェックリスト（全箇所。漏れると P-frame ドリフトまたはデコード破壊になる）

#### (a) `Sources/vevc/Quant.swift` — `Quantizer`

```swift
init(step: Int, ...) {          // step は Q4 で受ける
    self.step = Int16(step)     // Q4 のまま保持
    self.mul = Int32((1 << 20) / step)   // 旧: (1<<16)/step。Q4 なので 4bit 追加
    ...bias は変更しない
}
```

量子化カーネル `quantizeSIMD*` / `quantizeSIMDSignedMapping*`（全変種）: **変更不要**。`(absVal * mul + bias) >> 16` の mul の意味が自動的に正しくなる（mul = 2^20/stepQ4 = 2^16/実step）。bias は Q16 出力単位なので不変。

#### (b) `Sources/vevc/Quant.swift` — 逆量子化カーネル（全 10 変種）

`dequantizeSIMD{4,8,16,32,Generic}` と `dequantizeSIMDSignedMapping{4,8,16,32,Generic}` の乗算をすべて:

```swift
// 旧: Int16(clamping: Int32(v) &* step)
// 新: Q4 step との積を +8 で丸めて 4bit 右シフト
Int16(clamping: (Int32(v) &* step &+ 8) >> 4)
```

- 丸め方式（`&+ 8` してから算術右シフト）は**全変種で完全に同一**にすること。エンコーダの再構成とデコーダがこの関数群を共有しているため、1 箇所でも違うと P-frame 予測がドリフトする
- SignedMapping 変種は zigzag デコード後の各レーンの乗算部分だけを変える。SIMD ロード/ストア部分は触らない
- オーバーフロー確認: |v| ≤ 32767, stepQ4 ≤ 4096 → 積 ≤ 1.35×10^8 < Int32.max ✓

#### (c) `Sources/vevc/Quant.swift` — `QuantizationTable`

`init(baseStep:)` の baseStep は Q4 で受ける。定数を 16 倍する:

```swift
let s = max(16, min(baseStep, 4096))    // 旧: max(1, min(baseStep, 32767))
```

派生ステップ（すべて Q4 のまま計算されるので比率計算のロジックは不変、**上下限の定数のみ 16 倍**）:

- Luma: `lLow = min(256, max(16, baseStep / qLowDivisor))`, `lMid = min(768, max(16, (baseStep * qMidNum) / qMidDen))`, `lHigh = min(1024, max(16, (baseStep * qHighNum) / qHighDen))`（旧 16/48/64 → 256/768/1024）
- Chroma: `cLow = min(256, ...)`, `cMid = min(384, ...)`, `cHigh = min(768, ...)`（旧 16/24/48 → 256/384/768）

`AQTable`: baseStep が Q4 になるだけで比率スケールのロジックは変更不要。

#### (d) `Sources/vevc/Encoder.swift`

- `estimateQuantization`: `probeStep = 1024`（旧 64、Q4 で 64.0）。clamp を `min(4096, Int(max(16, correctedStep64)))` に（旧 `min(256, ..., max(1,`）。predictedStep64 の式自体は比率計算なので不変
- line 188-189, 241-242 の `max(1, baseStep)` → `max(16, baseStep)`
- `RateController` に渡す `qStep: Int(qtY.step)` は Q4 のまま渡す（RateController 内は比率計算が主なので単位が揃っていればよい）

#### (e) `Sources/vevc/RateController.swift`

比率計算（SADRatio, multiplier, EMA）は単位に依存しないので不変。**絶対値の定数のみ修正**:

- `let maxStep = max(baseStep * 2, min(512, baseStep * 4))` → `min(8192, baseStep * 4)`（512 = 実数512 → Q4 で 8192）
- `let minStep = max(1, baseStep)` → `max(16, baseStep)`
- `isDriftAccelerating` の `32 < lastDistortion` は distortion（画素 SAD）であって step ではないので**変更しない**

#### (f) 実数スケールを期待する箇所（Q4 → 実数変換を挟む）

`grep -rn "Int(qt[YC]*\.step)"` で全箇所を確認し、以下の分類で対応:

1. **DeblockingFilter 呼び出し**（`applyDeblockingFilter32/16/Chroma16` の `qStep:` 引数）: `qStep: (Int(qtY2.step) + 8) >> 4` に変換。該当: `EncodeSpatial.swift` line 52-54, 142-144, 239-241 / `Decode.swift` line 621-627。**エンコード側とデコード側で同じ変換式を使うこと**
2. **ゼロ閾値ヒューリスティクス**: `EncodePlane.swift` line 718-719, 747-748, 1224, 1244, 1263 の `Int(qtY.step) / 4` 等 → `Int(qtY.step) / 64`（/4 の実数スケール ≒ Q4 の /64）。line 1212 の `scaledSADThreshold(150, step: Int(qtY.step))` → `step: (Int(qtY.step) + 8) >> 4`。`scaledSADThreshold` の定義も確認し、実数 step を期待しているならこの変換で正しい
3. **QuantizationTable/AQTable の再構築**（`EncodeSpatial.swift` line 10-17 等の `QuantizationTable(baseStep: Int(qtY.step), ...)`）: baseStep は Q4 のまま渡す（**変換しない**）

#### (g) シリアライズ / デシリアライズ

`DataLayout.swift`:

- `VEVCLayerData.serialize` の `qtYStep/qtCStep`（UInt16BE）は Q4 値をそのまま書く（最大 4096 なので UInt16 に収まる）。コード変更は不要だが意味が変わる
- `VEVCLayerData.deserialize`（line 236-237）も Q4 として `QuantizationTable(baseStep:)` に渡すので変更不要（意味だけ変わる）
- **profile バージョンは変更しない（`0x01` のまま）。** `VEVCFileHeader` とデシリアライズ側の guard には一切手を付けないこと。step フィールドの意味（整数 → Q4）だけが変わるため、旧フォーマットで生成された既存の .vevc ファイル（リポジトリ直下の a.vevc 等）は今後、無エラーで誤った量子化ステップとしてデコードされる。これは了承済みの仕様であり、互換検出の仕組みを追加してはならない
- `docs/DataLayout.md` の Quantization Step フィールドの記述を「Q4 固定小数点（16 = 1.0）」に更新する（profile の記述は変更しない）

#### (h) テスト

- `Tests/vevcSpecV1/` はフォーマット仕様のテストなので、step の意味変更（Q4 化）に伴い期待値の更新が必要になる可能性が高い。**期待値を新仕様に合わせて更新することは許可**（テストの意図・網羅性を弱める変更は不可。profile に関する期待値は 0x01 のまま）。`Tests/vevcSpecV1/testdata` に旧 step 意味のバイナリがある場合は新しい意味で再生成する
- `Tests/vevcTests/BlockRoundtripTests.swift` 等の量子化往復テストが Q4 前提に合うよう更新
- 新規テストを 1 本追加: 「Quantizer(stepQ4) で quantize→dequantize した往復誤差が step/2 以下（Q4 換算 stepQ4/32 以下）」を stepQ4 ∈ {16, 24, 40, 64, 1024, 4096} で検証

### Phase 2 の期待結果

- 3500〜4500 kbps のプラトー/崖が消え、6 点すべてでサイズが目標 ±20% に入る
- `VEVC_RC_LOG=1` で q（Q4）が bitrate に対して滑らかに変化していることを確認
- SSIM が同一実サイズ比較で Phase 1 時点から悪化していないこと

---

## Phase 3: P-frame QStep クランプの予算駆動化

### 目的

P-frame の QStep 下限が `baseStep`（= I-frame と同じ細かさまで）に固定されているため、GOP 予算が余っていても使えず、逆に不足していても 4×baseStep までしか粗くできない。これを予算に連動させる。

### 変更: `Sources/vevc/RateController.swift` の `calculatePFrameQStep`

（Q4 前提。Phase 2 未実施ならば定数を 1/16 で読み替える）

```swift
// 旧:
// let maxStep = max(baseStep * 2, min(512, baseStep * 4))
// let minStep = max(1, baseStep)

// 新: 予算の残り具合でクランプ幅を動かす
// budgetRatioQ8 = 残予算が計画通りなら 256 (1.0)
let plannedRemaining = max(1, (gopTargetBits * gopRemainingFrames) / max(1, keyint))
let budgetRatioQ8 = max(64, min(1024, (gopRemainingBits << 8) / plannedRemaining))

var maxStep = max(baseStep * 2, min(8192, baseStep * 4))
var minStep = max(16, baseStep)
if budgetRatioQ8 < 192 {
    // 予算逼迫 (<0.75): さらに粗くすることを許可
    maxStep = min(8192, baseStep * 8)
} else if 320 < budgetRatioQ8 {
    // 予算余剰 (>1.25): I-frame より一段細かくすることを許可
    minStep = max(16, (baseStep * 3) / 4)
}
```

`gopRemainingBits` が負になり得る点に注意（`<< 8` の前に負値チェック: 負なら budgetRatioQ8 = 64 とする）。

### Phase 3 の期待結果

- 同一 q プラトー内でも bitrate 増加に応じて P-frame が細かくなり、サイズと SSIM が bitrate に単調追従する
- オーバーシュート方向に戻っていないこと（Phase 1 の closed-loop が吸収するはずだが、6 点測定で確認）

---

## Phase 4（任意・時間があれば）: 複雑度推定の改善

Phase 1〜3 で合格基準を満たせない場合のみ着手:

1. `estimateQuantization` のサンプル 8 点（四隅・辺中点のみ）に画面中心 4 点を追加（例: (w/2±w/4, h/2±h/4) 相当のブロック）。`samplePixels` の分母更新を忘れない
2. probeStep 2 点測定（Q4 で 1024 と 256）による冪則フィット: `bits(step) = A * step^(-α)` の α を 2 点から求め、`targetBits` に対応する step を解く。整数演算で行うため log2 近似（`63 - leadingZeroBitCount` とテーブル補間）を使う

---

## 5. 最終確認チェックリスト

- [ ] `swift build -c release` 警告なし
- [ ] `swift test` 全パス
- [ ] 6 点測定の表（bitrate / 目標KB / 実測KB / 乖離% / SSIM / 同一実サイズでのベースラインSSIM）を作成し PR に添付
- [ ] 1500 kbps: 5,500 KB ±15%
- [ ] プラトー（隣接点サイズ差 2% 未満）なし
- [ ] 同一実サイズ比較で SSIM 悪化 0.003 未満
- [ ] `VEVC_RC_LOG=1` で rateGainQ8 が発振していない
- [ ] Phase 2 実施時: `docs/DataLayout.md` の step 記述更新済み、profile は 0x01 のまま無変更、旧 step 意味のテストデータ再生成済み