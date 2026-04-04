.PHONY: wasm
wasm:
	WASM_BUILD=1 swift package --swift-sdk swift-6.2.3-RELEASE_wasm -c release plugin --allow-writing-to-package-directory js --use-cdn --product wasm
	wasm-opt -O4 --fast-math --converge .build/plugins/PackageToJS/outputs/Package/wasm.wasm -o .build/plugins/PackageToJS/outputs/Package/wasm.wasm
	
.PHONY: build
build:
	swift build -c release