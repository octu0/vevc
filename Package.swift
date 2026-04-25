// swift-tools-version: 6.0
import PackageDescription
import Foundation

let isWasmBuild = ProcessInfo.processInfo.environment["WASM_BUILD"] == "1"

var packageProducts: [Product] = [
    .library(name: "vevc", targets: ["vevc"]),
]

var packageDeps: [Package.Dependency] = [
    .package(url: "https://github.com/tayloraswift/swift-png", from: "4.4.9")
]

var packageTargets: [Target] = [
    .target(
        name: "vevc",
        swiftSettings: [
            .unsafeFlags(["-Ounchecked", "-wmo", "-Xcc", "-msimd128"], .when(platforms: [.wasi]))
        ]
    ),
    .testTarget(
        name: "vevcTests",
        dependencies: ["vevc"]
    ),
    .executableTarget(
        name: "example-enc",
        dependencies: [
            "vevc",
            .product(name: "PNG", package: "swift-png")
        ],
        path: "Sources/example-enc"
    ),
    .executableTarget(
        name: "example-dec",
        dependencies: [
            "vevc",
            .product(name: "PNG", package: "swift-png")
        ],
        path: "Sources/example-dec"
    ),
    .executableTarget(
        name: "vevc-enc",
        dependencies: ["vevc"]
    ),
    .executableTarget(
        name: "vevc-dec",
        dependencies: ["vevc"]
    ),
    .executableTarget(
        name: "compare",
        dependencies: [
            "vevc",
            .product(name: "PNG", package: "swift-png")
        ],
        path: "Sources/example"
    ),
    .executableTarget(
        name: "vevc-splitter",
        dependencies: ["vevc"],
        path: "Sources/vevc-splitter"
    ),
    .executableTarget(
        name: "vevc-training",
        dependencies: ["vevc"],
        path: "Sources/vevc-training"
    )
]

if isWasmBuild {
    packageDeps.append(.package(url: "https://github.com/swiftwasm/JavaScriptKit", from: "0.45.0"))
    packageProducts.append(.executable(name: "wasm", targets: ["wasm"]))
    
    packageTargets.append(
        .executableTarget(
            name: "wasm",
            dependencies: [
                "vevc",
                .product(name: "JavaScriptKit", package: "JavaScriptKit"),
                .product(name: "JavaScriptEventLoop", package: "JavaScriptKit")
            ],
            swiftSettings: [
                .enableExperimentalFeature("Extern"),
                .unsafeFlags(["-Ounchecked", "-wmo", "-Xcc", "-msimd128"], .when(platforms: [.wasi]))
            ],
            plugins: [
                .plugin(name: "BridgeJS", package: "JavaScriptKit")
            ]
        )
    )
}

let package = Package(
    name: "vevc",
    platforms: [
        .macOS(.v15)
    ],
    products: packageProducts,
    dependencies: packageDeps,
    targets: packageTargets,
    swiftLanguageModes: [.v6]
)
