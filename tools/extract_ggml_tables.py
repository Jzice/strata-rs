#!/usr/bin/env python3
"""Extract the GGML_TABLE_BEGIN/END lookup tables from Strata's vendored
ggml-common.h into a Rust source file (const arrays), so the Rust replica's
dequantizers use byte-exact tables without hand transcription.

Usage: python tools/extract_ggml_tables.py <ggml-common.h> <out.rs>
"""
import re
import sys


def main(src: str, out: str) -> None:
    text = open(src, "r", encoding="utf-8", errors="replace").read()
    # Match: GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END()
    pat = re.compile(
        r"GGML_TABLE_BEGIN\(\s*([A-Za-z_0-9]+)\s*,\s*([A-Za-z_0-9]+)\s*,\s*([0-9]+)\s*\)\s*(.*?)\s*GGML_TABLE_END\(\)",
        re.S,
    )
    tables = []
    for m in pat.finditer(text):
        ctype, name, size, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        # Split body on commas that are not inside brackets (there are none here,
        # but keep it robust) and strip comments.
        body = re.sub(r"//[^\n]*", "", body)
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        vals = [v.strip() for v in body.split(",") if v.strip()]
        assert len(vals) == size, f"{name}: declared {size}, found {len(vals)}"
        tables.append((ctype, name, size, vals))

    rust_ctype = {
        "uint8_t": "u8",
        "int8_t": "i8",
        "uint16_t": "u16",
        "int16_t": "i16",
        "uint32_t": "u32",
        "int32_t": "i32",
        "uint64_t": "u64",
        "int64_t": "i64",
    }
    lines = [
        "//! GGML lookup tables, extracted verbatim from Strata's vendored",
        "//! third_party/ggml/ggml-common.h by tools/extract_ggml_tables.py.",
        "//! Do not edit by hand.",
        "",
        "#![allow(clippy::unreadable_literal)]",
        "",
    ]
    for ctype, name, size, vals in tables:
        rt = rust_ctype.get(ctype)
        assert rt, f"unknown C type {ctype} for table {name}"
        lines.append(f"pub const {name.upper()}: [{rt}; {size}] = [")
        # 12 values per line
        for i in range(0, size, 12):
            chunk = ", ".join(vals[i : i + 12])
            lines.append(f"    {chunk},")
        lines.append("];")
        lines.append("")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {len(tables)} tables to {out}:")
    for ctype, name, size, _ in tables:
        print(f"  {name.upper():24s} [{rust_ctype[ctype]}; {size}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
