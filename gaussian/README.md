# Gaussian splats in HA++

A Gaussian splat renderer for phones, written in HA++. The same engine runs on
Android (Vulkan), iPhone (Metal) and PC (Vulkan or CPU).

| File | What |
|---|---|
| `splat.ha` | the engine: packs splats to 32 bytes, projects them (EWA), sorts them on the GPU (radix sort) and draws them with a 16×16-tile rasterizer |
| `demo_scene.ha` | a generated "galaxy" scene, so the app works without a data file |
| `render.py` | PC host: renders to a PNG on this computer's GPU; also loads `.ply` files from 3D Gaussian Splatting |
| `android/SplatRenderer.java` | Android host (Vulkan, through the generated JNI class) |
| `apple/SplatRenderer.swift` | Swift host for an ARKit app (needs Xcode or xtool) |
| `ios/main.c` + `build_ios.py` | a complete iPhone/iPad app, built into `Splats.ipa` **without a Mac, Xcode or the iOS SDK** |

How the pipeline works: [docs/SPLATS.md](../docs/SPLATS.md).

## Try it on a PC

```sh
python3 gaussian/render.py --out splats.png            # a generated demo scene
python3 gaussian/render.py --ply scene.ply --out s.png # a real 3DGS scene
```

## Build the iPhone app (on Windows, Linux or Mac)

You need Python 3 and LLVM (the same tools as the rest of HA++):

| Your computer | Install |
|---|---|
| Windows | `winget install LLVM.LLVM` and `winget install Python.Python.3.12` |
| Linux | `sudo apt install clang lld llvm python3` |
| macOS | `brew install llvm lld python` |

```sh
python3 gaussian/build_ios.py                    # -> gaussian/build/ios/Splats.ipa
python3 gaussian/build_ios.py --ply scene.ply    # put your own splat scene inside the app
```

The script:
1. compiles the HA++ engine for ARM64 iOS, with the Metal source of the GPU
   kernels embedded;
2. compiles `ios/main.c`, which talks to UIKit and Metal through the
   Objective-C runtime (no Apple headers);
3. links with `ld64.lld`. It gets the list of iOS system functions from small
   `.tbd` text files it writes itself, so no Apple files are copied or needed;
4. writes `Info.plist` and the app icon (rendered with the splat engine when
   this PC has a Vulkan GPU);
5. zips `Splats.ipa` and prints what it checked in the binary (arm64, iOS 15+,
   entry point, linked libraries).

## Install it on your iPhone (from Windows)

The `.ipa` is not signed, because only Apple's tools and your Apple ID can sign
apps. A sideloading tool signs it with your Apple ID while it installs it.

**With Sideloadly** (Windows or Mac, [sideloadly.io](https://sideloadly.io)):
1. On Windows, install **iTunes** and **iCloud** from Apple's website (not the
   Microsoft Store versions). Sideloadly uses their drivers to talk to the
   iPhone.
2. Connect the iPhone with a USB cable, unlock it, and tap **Trust**.
3. Open Sideloadly, drag `Splats.ipa` into it, enter your Apple ID and press
   **Start**.
4. On the iPhone (iOS 16 or newer), turn on **Settings ▸ Privacy & Security ▸
   Developer Mode** and restart when asked.
5. Go to **Settings ▸ General ▸ VPN & Device Management**, tap your Apple ID
   and choose **Trust**.
6. Open **Splats**.

**With AltStore** ([altstore.io](https://altstore.io)): install AltServer on
the PC and AltStore on the iPhone, then open `Splats.ipa` from **My Apps ▸ +**.

With a **free Apple ID**, a sideloaded app stops opening after 7 days. Install
it again, or refresh it in AltStore. A free Apple ID can also have only a few
sideloaded apps at once. A paid Apple Developer account removes these limits.
These tools change over time, so if a step looks different, follow the tool's
current instructions.

## Using the app

- **One finger:** orbit around the scene. After 2 seconds without touching,
  the camera orbits slowly by itself.
- **Pinch:** zoom. Very close views create millions of (splat, tile) pairs to
  sort. Above 8 million the app keeps the last image and asks you to pinch out,
  instead of running out of memory.
- The text at the top shows the number of splats, the frame time and the fps.
  If something goes wrong, for example a Metal compile error, the message
  appears there too.
- Rotation, iPad Split View and Stage Manager are handled: the image is
  re-rendered at the new size. The app pauses rendering in the background.

## What is tested, and what isn't

| | Status |
|---|---|
| The engine (`splat.ha`) on a Vulkan GPU: sort order identical to an independent NumPy renderer, image within 1–2/255 | ✅ tested (`tests/test_happ.py`) |
| The engine's Metal kernels: compiled and run through a C++ stand-in for Metal, match the CPU | ⚠️ emulated, not Apple's Metal compiler |
| `Splats.ipa` builds from LLVM alone; the Mach-O is arm64 iOS 15+, has an entry point and signature space, and every iOS function it imports is bound to the right system library | ✅ tested |
| The app actually opening and drawing on an iPhone | ❌ not yet: no iPhone was available. The first run on a real device may show problems that can't be seen here. The status text in the app is there to make them visible. |
| Android: the Java host renders the same bytes as the PC host (desktop JVM + Vulkan) | ✅ tested; not yet on a phone |

If the app shows an error, or closes on start, send the text shown at the top
of the screen or the crash log (**Settings ▸ Privacy & Security ▸ Analytics &
Improvements ▸ Analytics Data**, entries starting with `Splats`).
