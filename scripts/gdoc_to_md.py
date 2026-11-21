"""Convert a Google Docs API JSON (read_doc output) into Markdown.

Used to refresh docs/progress/progress_update.md from the rolling Google Doc:
save the Google Docs `read_doc` result (JSON with a top-level "content" key) to a
file, then run:

    python scripts/gdoc_to_md.py read_doc.json > body.md

and put body.md under the header comment of the mirror file.
"""
import json
import sys

HEAD = {"TITLE": "#", "HEADING_1": "#", "HEADING_2": "##", "HEADING_3": "###",
        "HEADING_4": "####", "HEADING_5": "#####", "HEADING_6": "######", "SUBTITLE": "##"}


def runs(elements, plain=False):
    out = []
    for e in elements:
        tr = e.get("textRun")
        if not tr:
            if "inlineObjectElement" in e:
                out.append("[image]")
            elif "richLink" in e:
                rl = e["richLink"].get("richLinkProperties", {})
                out.append(f"[{rl.get('title', 'link')}]({rl.get('uri', '')})")
            continue
        text = tr["content"].replace("\n", "").replace("\u000b", "  \n")
        if not text:
            continue
        st = tr.get("textStyle", {})
        core = text.strip()
        if core and not plain:
            lead = text[: len(text) - len(text.lstrip())]
            trail = text[len(text.rstrip()):]
            if st.get("weightedFontFamily", {}).get("fontFamily", "").lower() in ("courier new", "roboto mono", "consolas"):
                core = f"`{core}`"
            if st.get("bold"):
                core = f"**{core}**"
            if st.get("italic"):
                core = f"*{core}*"
            if st.get("link", {}).get("url"):
                core = f"[{core}]({st['link']['url']})"
            text = lead + core + trail
        out.append(text)
    s = "".join(out)
    return s.replace("****", "")


def table(t):
    rows = []
    for r in t["tableRows"]:
        cells = []
        for c in r["tableCells"]:
            parts = [runs(x["paragraph"]["elements"]).strip() for x in c["content"] if "paragraph" in x]
            cells.append("<br>".join(p for p in parts if p).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    if rows:
        n = len(t["tableRows"][0]["tableCells"])
        rows.insert(1, "|" + "---|" * n)
    return "\n".join(rows)


def convert(doc):
    lists = doc.get("lists", {})
    lines = []
    prev_list = False
    for el in doc["body"]["content"]:
        if "paragraph" in el:
            p = el["paragraph"]
            style = p.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
            text = runs(p["elements"], plain=style in HEAD).rstrip()
            b = p.get("bullet")
            if b:
                lvl = b.get("nestingLevel", 0)
                glyph = lists.get(b["listId"], {}).get("listProperties", {}).get("nestingLevels", [{}])
                g = glyph[min(lvl, len(glyph) - 1)] if glyph else {}
                ordered = "glyphType" in g and g.get("glyphType") not in ("GLYPH_TYPE_UNSPECIFIED", None) and not g.get("glyphSymbol")
                marker = "1." if ordered else "-"
                lines.append("  " * lvl + f"{marker} {text}")
                prev_list = True
                continue
            if prev_list:
                lines.append("")
                prev_list = False
            if not text.strip():
                continue
            if style in HEAD:
                lines.append(f"{HEAD[style]} {text.strip()}\n")
            else:
                lines.append(text + "\n")
        elif "table" in el:
            if prev_list:
                lines.append("")
                prev_list = False
            lines.append(table(el["table"]) + "\n")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))["content"]
    tab = d["tabs"][0]["documentTab"]
    sys.stdout.write(convert(tab))
