"""Download the Super Mario Bros levels from the Video Game Level Corpus (TheVGLC).

Only the files this project needs are fetched (15 levels, the legend, the licence and
the README), from a pinned commit so results are reproducible. Nothing is committed to
this repository: data/ is in .gitignore.

    python scripts/download_data.py
"""
import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "TheVGLC/TheVGLC"
# Commit checked on 2026-11-11. Use --ref master to get the latest version instead.
DEFAULT_REF = "0e86a8f31f20ecad4eaa5741ff061af88767f7fb"

LEVELS = ["1-1", "1-2", "1-3", "2-1", "3-1", "3-3", "4-1", "4-2",
          "5-1", "5-3", "6-1", "6-2", "6-3", "7-1", "8-1"]
FILES = (["License.md", "README.md", "Super Mario Bros/smb.json"]
         + [f"Super Mario Bros/Processed/mario-{lv}.txt" for lv in LEVELS])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/TheVGLC")
    p.add_argument("--ref", default=DEFAULT_REF, help="git commit or branch of TheVGLC")
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = p.parse_args()

    out = Path(args.out)
    for rel in FILES:
        dest = out / rel
        if dest.exists() and not args.force:
            print(f"exists   {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPO}/{args.ref}/{urllib.parse.quote(rel)}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
        except Exception as e:  # network errors, 404, ...
            sys.exit(f"Failed to download {url}: {e}")
        dest.write_bytes(data)
        print(f"fetched  {rel} ({len(data)} bytes)")
    print(f"\nDone. Dataset is in {out.resolve()}")
    print("Licence: see License.md in that folder (MIT, (c) 2016 Summerville, Ontanon, Snodgrass).")


if __name__ == "__main__":
    main()
