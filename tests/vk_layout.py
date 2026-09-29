"""Compare every struct offset/size and enum in runtime/gpu/ha_vk_min.h with the official vulkan.h."""
import os
import re
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_H = os.path.join(ROOT, "runtime", "gpu", "ha_vk_min.h")

# structs whose tail is intentionally reserved space (only leading fields are compared)
PARTIAL = {"VkPhysicalDeviceProperties": ["apiVersion", "driverVersion", "vendorID", "deviceID",
                                          "deviceType", "deviceName", "pipelineCacheUUID"],
           "VkPhysicalDeviceFeatures": []}


def parse_min_header():
    with open(MIN_H) as f:
        text = f.read()
    structs = {}
    for m in re.finditer(r"typedef struct (\w+) \{(.*?)\} \1;", text, re.S):
        name, body = m.group(1), m.group(2)
        fields = []
        for line in body.split(";"):
            line = line.strip()
            if not line:
                continue
            decl = line.split(",")[0]
            fm = re.search(r"(\w+)\s*(\[\d+\])?$", decl)
            if fm:
                fields.append(fm.group(1))
                for extra in line.split(",")[1:]:
                    fields.append(extra.strip())
        structs[name] = fields
    enums = re.findall(r"(VK_[A-Z0-9_]+) = (0x[0-9a-fA-F]+|\d+)", text)
    return structs, enums


def program(header, structs, enums):
    lines = [f"#include {header}", "#include <stdio.h>", "#include <stddef.h>", "int main(void) {"]
    for s, fields in sorted(structs.items()):
        comp = PARTIAL.get(s, fields)
        if s not in PARTIAL:
            lines.append(f'  printf("{s} size %zu\\n", sizeof({s}));')
        for f in comp:
            lines.append(f'  printf("{s}.{f} %zu\\n", offsetof({s}, {f}));')
    for e, _ in enums:
        lines.append(f'  printf("{e} %lld\\n", (long long){e});')
    lines.append('  printf("features size %zu\\n", sizeof(VkPhysicalDeviceFeatures));')
    lines += ["  return 0;", "}"]
    return "\n".join(lines) + "\n"


def run():
    structs, enums = parse_min_header()
    outs = []
    with tempfile.TemporaryDirectory() as tmp:
        for official in (True, False):
            src = os.path.join(tmp, f"p{int(official)}.c")
            exe = os.path.join(tmp, f"p{int(official)}")
            header = "<vulkan/vulkan.h>" if official else f'"{MIN_H}"'
            with open(src, "w") as f:
                f.write(program(header, structs, enums))
            subprocess.run(["clang", "-w", src, "-o", exe], check=True)
            outs.append(subprocess.run([exe], capture_output=True, text=True, check=True).stdout)
    a, b = outs[0].splitlines(), outs[1].splitlines()
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return len(a), diffs


if __name__ == "__main__":
    n, diffs = run()
    for x, y in diffs:
        print("MISMATCH official:", x, "| ours:", y)
    print(f"{n} layout/enum checks, {len(diffs)} mismatches")
