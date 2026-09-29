"""Build Splats.ipa, the Gaussian splat viewer for iPhone/iPad, without Xcode or the iOS SDK.

    python3 gaussian/build_ios.py                    # -> gaussian/build/ios/Splats.ipa
    python3 gaussian/build_ios.py --ply scene.ply    # bundle your own 3D Gaussian Splatting scene

Needs only Python and LLVM (clang + ld64.lld), on Windows, Linux or macOS.

How it works without Apple's SDK:
  * the app (gaussian/ios/main.c) declares the few iOS functions it uses itself and
    talks to UIKit/Metal through the Objective-C runtime;
  * the linker only needs to know which system library exports each symbol. That
    is what the small .tbd text files written below say (install path + symbol
    names). No Apple files are copied or needed;
  * the HA++ engine (gaussian/splat.ha + demo_scene.ha) is compiled for ARM64 iOS
    with the Metal source of its kernels embedded.

The .ipa is not signed with an Apple certificate (only Apple's tools and your
Apple ID can do that). Install it with a sideloading tool that signs it with
your Apple ID, e.g. Sideloadly (Windows/macOS) or AltStore; see gaussian/README.md.
"""
import argparse
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from happ.driver import build  # noqa: E402
from happ.targets import Toolchain  # noqa: E402

MIN_IOS = "15.0"
SDK_VERSION = "17.0"      # recorded in the binary; iOS uses it for compatibility behaviour
BUNDLE_ID = "com.happ.splats"
APP_NAME = "Splats"

# Which iOS library exports each symbol the app (and the HA++ library) uses.
SYSTEM_SYMBOLS = {
    "/usr/lib/libSystem.B.dylib": [
        "_malloc", "_free", "_memcpy", "_memset", "_snprintf", "_fopen", "_fread", "_fseek", "_ftell",
        "_fclose", "_sin", "_cos", "_tan", "_sqrt", "___stack_chk_fail", "___stack_chk_guard"],
    "/usr/lib/libobjc.A.dylib": [
        "_objc_msgSend", "_objc_getClass", "_sel_registerName", "_objc_allocateClassPair",
        "_objc_registerClassPair", "_class_addMethod", "_objc_autoreleasePoolPush", "_objc_autoreleasePoolPop"],
    "/System/Library/Frameworks/UIKit.framework/UIKit": ["_UIApplicationMain"],
    "/System/Library/Frameworks/Metal.framework/Metal": ["_MTLCreateSystemDefaultDevice"],
    "/System/Library/Frameworks/QuartzCore.framework/QuartzCore": ["_CACurrentMediaTime"],
    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics": [
        "_CGColorSpaceCreateDeviceRGB", "_CGColorSpaceRelease", "_CGDataProviderCreateWithData",
        "_CGDataProviderRelease", "_CGImageCreate", "_CGImageRelease"],
    # Foundation is not called directly but must be loaded for NSString & co.
    "/System/Library/Frameworks/Foundation.framework/Foundation": [],
}


def run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{what} failed:\n  $ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout


def write_tbds(sdk_dir):
    os.makedirs(sdk_dir, exist_ok=True)
    paths = []
    for install, syms in SYSTEM_SYMBOLS.items():
        name = os.path.basename(install).split(".")[0]
        path = os.path.join(sdk_dir, f"{name}.tbd")
        sym_list = ", ".join(syms)
        with open(path, "w") as f:
            f.write("--- !tapi-tbd\n"
                    "tbd-version:     4\n"
                    "targets:         [ arm64-ios ]\n"
                    f"install-name:    '{install}'\n"
                    "current-version: 1\n")
            if syms:
                f.write("exports:\n"
                        "  - targets:         [ arm64-ios ]\n"
                        f"    symbols:         [ {sym_list} ]\n")
            f.write("...\n")
        paths.append(path)
    return paths


def info_plist(has_icon):
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.1",
        "CFBundleVersion": "1",
        "CFBundleSupportedPlatforms": ["iPhoneOS"],
        "DTPlatformName": "iphoneos",
        "DTPlatformVersion": SDK_VERSION,
        "DTSDKName": f"iphoneos{SDK_VERSION}",
        "LSRequiresIPhoneOS": True,
        "MinimumOSVersion": MIN_IOS,
        "UIDeviceFamily": [1, 2],
        "UIRequiredDeviceCapabilities": ["arm64", "metal"],
        "UILaunchScreen": {},
        "UIStatusBarHidden": True,
        "UISupportedInterfaceOrientations": ["UIInterfaceOrientationPortrait",
                                             "UIInterfaceOrientationLandscapeLeft",
                                             "UIInterfaceOrientationLandscapeRight"],
    }
    if has_icon:
        icons = {"CFBundlePrimaryIcon": {"CFBundleIconFiles": ["AppIcon60x60"], "CFBundleIconName": "AppIcon"}}
        info["CFBundleIcons"] = icons
        info["CFBundleIcons~ipad"] = {"CFBundlePrimaryIcon": {"CFBundleIconFiles": ["AppIcon60x60",
                                                                                       "AppIcon76x76"]}}
    return plistlib.dumps(info, fmt=plistlib.FMT_BINARY)


def make_icons(app_dir, work):
    """Render the app icon with the same HA++ pipeline on this PC's GPU (skipped if there is none)."""
    try:
        import numpy as np
        sys.path.insert(0, HERE)
        import render as R
        pipe = R.Pipeline(R.build(out_dir=os.path.join(work, "pc")))
        pipe.upload(R.synthetic_scene(8000, seed=2))
        for name, size in (("AppIcon60x60@2x.png", 120), ("AppIcon60x60@3x.png", 180),
                           ("AppIcon76x76@2x.png", 152)):
            cam = R.look_at([0, 3.4, -3.8], [0, 0, 0], size, size)
            img, _ = pipe.render(cam, size, size, background=(8, 10, 22))
            img = img | np.uint32(0xFF000000)
            R.write_png(os.path.join(app_dir, name), img)
        return True
    except Exception as e:           # no Vulkan GPU / numpy: the app just has no icon
        print(f"  note: no app icon ({e.__class__.__name__}: {e})")
        return False


def ply_to_scene(ply, out_path):
    sys.path.insert(0, HERE)
    import render as R
    raw = R.load_ply(ply)
    raw.astype("<f4").tofile(out_path)
    return len(raw)


def verify(tc, exe):
    """Print the facts iOS checks when it loads the app."""
    otool = tc.find("llvm-otool")
    if otool is None:
        return
    out = run([otool, "-l", exe], "llvm-otool")
    lines = [l.strip() for l in out.splitlines()]
    platform = next((l.split()[1] for l in lines if l.startswith("platform ")), "?")
    minos = next((l.split()[1] for l in lines if l.startswith("minos ")), "?")
    dylibs = [l.split()[1] for l in lines if l.startswith("name /") and "dyld" not in l]
    print(f"  checked: Mach-O arm64, platform {platform} (2 = iOS), minimum iOS {minos}")
    print(f"  checked: entry point LC_MAIN: {'cmd LC_MAIN' in lines}, code signature: "
          f"{'cmd LC_CODE_SIGNATURE' in lines} (ad-hoc; the sideloading tool re-signs)")
    for d in dylibs:
        print(f"           links {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", help="3D Gaussian Splatting .ply to bundle (default: generated galaxy)")
    ap.add_argument("--out", default=os.path.join(HERE, "build", "ios"))
    ap.add_argument("--no-icon", action="store_true")
    args = ap.parse_args()
    tc = Toolchain()
    clang = tc.require("clang", "compiling the app")
    lld = tc.require("ld64.lld", "linking the iOS app (part of LLVM's lld)")
    out = args.out
    work = os.path.join(out, "work")
    os.makedirs(work, exist_ok=True)

    print("1/5 HA++ engine for iPhone (ARM64 + embedded Metal kernels)")
    lib_out = os.path.join(work, "happ")
    produced = build(os.path.join(HERE, "demo_scene.ha"), ["ios-arm64"], lib_out, name="splat", quiet=True)
    lib = produced["ios-arm64"]

    print("2/5 app shell (C, UIKit + Metal through the Objective-C runtime)")
    obj = os.path.join(work, "main.o")
    run([clang, f"--target=arm64-apple-ios{MIN_IOS}", "-O2", "-ffreestanding", "-fno-stack-protector",
         "-Wall", "-Wextra", "-Werror", "-Wno-override-module", "-I", os.path.join(lib_out, "include"),
         "-c", os.path.join(HERE, "ios", "main.c"), "-o", obj], "clang")

    print("3/5 link (ld64.lld + symbol lists, no SDK)")
    tbds = write_tbds(os.path.join(work, "sdk"))
    app_dir = os.path.join(out, f"{APP_NAME}.app")
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    os.makedirs(app_dir)
    exe = os.path.join(app_dir, APP_NAME)
    run([lld, "-arch", "arm64", "-platform_version", "ios", MIN_IOS, SDK_VERSION, "-fixup_chains",
         "-adhoc_codesign", "-dead_strip", "-o", exe, obj, lib] + tbds, "ld64.lld")

    print("4/5 app bundle")
    has_icon = False if args.no_icon else make_icons(app_dir, work)
    if args.ply:
        n = ply_to_scene(args.ply, os.path.join(app_dir, "scene.bin"))
        print(f"  bundled {n} splats from {args.ply}")
    with open(os.path.join(app_dir, "Info.plist"), "wb") as f:
        f.write(info_plist(has_icon))
    with open(os.path.join(app_dir, "PkgInfo"), "w") as f:
        f.write("APPL????")

    print("5/5 installer (.ipa)")
    ipa = os.path.join(out, f"{APP_NAME}.ipa")
    with zipfile.ZipFile(ipa, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(app_dir):
            for name in sorted(files):
                path = os.path.join(base, name)
                arc = os.path.join("Payload", os.path.relpath(path, out))
                zi = zipfile.ZipInfo(arc)
                zi.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if name == APP_NAME else 0o644
                zi.external_attr = (0o100000 | mode) << 16
                with open(path, "rb") as fh:
                    z.writestr(zi, fh.read())
    verify(tc, exe)
    size = os.path.getsize(ipa)
    print(f"done: {ipa} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
