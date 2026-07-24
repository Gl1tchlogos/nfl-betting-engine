from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    parts_dir = root / "comprehensive_parts"
    parts = sorted(parts_dir.glob("*.inc"))
    if not parts:
        raise RuntimeError("No comprehensive runner source fragments found")
    source = "".join(part.read_text(encoding="utf-8") for part in parts)
    output = root / "run_comprehensive_generated.py"
    output.write_text(source, encoding="utf-8")
    compile(source, str(output), "exec")
    print(f"Built {output} from {len(parts)} fragments ({len(source)} characters)")


if __name__ == "__main__":
    main()
