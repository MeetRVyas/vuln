from __future__ import annotations


def line_window(source: str, line_start: int, line_end: int, context: int = 5) -> str:
    lines = source.splitlines()
    start = max(0, line_start - 1 - context)
    end = min(len(lines), line_end + context)
    numbered = [f"{i + 1:>5}  {lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)
