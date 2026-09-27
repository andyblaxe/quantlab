"""A tiny document model rendered to Markdown and HTML (reports are built once, rendered twice)."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    kind: str  # h1 h2 h3 p ul table chart banner code
    content: Any
    extra: dict = field(default_factory=dict)


class Doc:
    def __init__(self, title: str) -> None:
        self.title = title
        self.blocks: list[Block] = []

    def h1(self, t): self.blocks.append(Block("h1", t)); return self
    def h2(self, t): self.blocks.append(Block("h2", t)); return self
    def h3(self, t): self.blocks.append(Block("h3", t)); return self
    def p(self, t): self.blocks.append(Block("p", t)); return self
    def ul(self, items): self.blocks.append(Block("ul", list(items))); return self
    def code(self, t): self.blocks.append(Block("code", t)); return self
    def banner(self, t, level="warn"): self.blocks.append(Block("banner", t, {"level": level})); return self

    def table(self, header: list[str], rows: list[list[Any]]):
        self.blocks.append(Block("table", {"header": header, "rows": rows}))
        return self

    def kv(self, pairs: list[tuple[str, Any]]):
        return self.table(["Item", "Value"], [[k, v] for k, v in pairs])

    def chart(self, fig, caption: str = ""):
        self.blocks.append(Block("chart", fig, {"caption": caption}))
        return self

    # ---------------------------------------------------------------------------------------------
    def to_markdown(self) -> str:
        out = []
        for b in self.blocks:
            if b.kind in ("h1", "h2", "h3"):
                out.append("#" * int(b.kind[1]) + " " + str(b.content))
            elif b.kind == "p":
                out.append(str(b.content))
            elif b.kind == "banner":
                out.append(f"> **{b.content}**")
            elif b.kind == "ul":
                out.append("\n".join(f"- {x}" for x in b.content) if b.content else "- (none)")
            elif b.kind == "code":
                out.append("```\n" + str(b.content) + "\n```")
            elif b.kind == "table":
                h, rows = b.content["header"], b.content["rows"]
                lines = ["| " + " | ".join(map(str, h)) + " |", "|" + "---|" * len(h)]
                lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
                out.append("\n".join(lines) if rows else "_(no rows)_")
            elif b.kind == "chart":
                out.append(f"_[chart: {b.extra.get('caption', '')} — see HTML version]_")
        return "\n\n".join(out) + "\n"

    def to_html(self, plotly_src: str = "plotly.min.js") -> str:
        esc = lambda x: html.escape(str(x))
        parts = []
        first_chart = True
        for b in self.blocks:
            if b.kind in ("h1", "h2", "h3"):
                parts.append(f"<{b.kind}>{esc(b.content)}</{b.kind}>")
            elif b.kind == "p":
                parts.append(f"<p>{esc(b.content)}</p>")
            elif b.kind == "banner":
                parts.append(f'<div class="banner {b.extra.get("level", "warn")}">{esc(b.content)}</div>')
            elif b.kind == "ul":
                parts.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in b.content) + "</ul>" if b.content else "<p><em>(none)</em></p>")
            elif b.kind == "code":
                parts.append(f"<pre>{esc(b.content)}</pre>")
            elif b.kind == "table":
                h, rows = b.content["header"], b.content["rows"]
                head = "".join(f"<th>{esc(c)}</th>" for c in h)
                body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
                parts.append(f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            elif b.kind == "chart":
                parts.append('<figure>' + b.content.to_html(full_html=False, include_plotlyjs=False,
                                                              config={"displaylogo": False, "responsive": True})
                             + f"<figcaption>{esc(b.extra.get('caption', ''))}</figcaption></figure>")
                first_chart = False
        script = f'<script src="{plotly_src}"></script>' if not first_chart else ""
        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(self.title)}</title>{script}<style>{REPORT_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"""


REPORT_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f21;--muted:#5f6368;--line:#e2e2de;--warn:#fff4d6;--warnb:#c98a00;--bad:#fde8e8;--badb:#b3261e;--ok:#e6f4ea;--okb:#1e7e34}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--fg:#e8e8e6;--muted:#a0a3a7;--line:#2c2e31;--warn:#3a2f10;--warnb:#e0b040;--bad:#3b1715;--badb:#f28b82;--ok:#10301b;--okb:#81c995}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:1000px;margin:0 auto;padding:24px 16px}
h1{font-size:1.6rem;margin:.2em 0 .6em}h2{font-size:1.25rem;margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:.2em}h3{font-size:1.05rem;margin-top:1.4em}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6em 0}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.banner{padding:10px 14px;border-left:4px solid var(--warnb);background:var(--warn);margin:12px 0;font-weight:600}
.banner.bad{border-color:var(--badb);background:var(--bad)}.banner.ok{border-color:var(--okb);background:var(--ok)}
pre{background:rgba(127,127,127,.1);padding:12px;overflow-x:auto;font-size:.85rem}figure{margin:1em 0}figcaption{color:var(--muted);font-size:.85rem}
"""
