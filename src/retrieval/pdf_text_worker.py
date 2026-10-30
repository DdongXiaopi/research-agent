"""Optional PDF parser process, bounded by the parent deadline and text budget."""
import sys


def main():
    from pypdf import PdfReader
    reader = PdfReader(sys.argv[1], strict=True)
    if reader.is_encrypted or len(reader.pages) > 200:
        raise ValueError("encrypted or oversized PDF is unsupported")
    maximum = int(sys.argv[2])
    parts, total = [], 0
    for page in reader.pages:
        value = page.extract_text() or ""
        total += len(value)
        if total > maximum:
            raise ValueError("PDF text exceeds character budget")
        parts.append(value)
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
