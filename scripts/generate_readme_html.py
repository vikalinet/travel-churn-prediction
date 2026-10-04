"""Генерация README.html из README.md для GitHub Pages.

Разметка страницы — в шаблоне templates/reports/readme.html.

    python scripts/generate_readme_html.py
"""

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]

_templates = Environment(
    loader=FileSystemLoader(ROOT / "templates"),
    autoescape=select_autoescape(["html"]),
)


def markdown_to_html(md_text: str) -> str:
    """Простой конвертер Markdown → HTML."""
    html = md_text

    # Экранирование HTML
    html = html.replace("&", "&amp;")
    html = html.replace("<", "&lt;")
    html = html.replace(">", "&gt;")

    # Заголовки
    html = re.sub(r"^###### (.+)$", r"<h6>\1</h6>", html, flags=re.MULTILINE)
    html = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
    html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

    # Жирный текст
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html)

    # Курсив
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    html = re.sub(r"_(.+?)_", r"<em>\1</em>", html)

    # Код inline
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

    # Блоки кода
    html = re.sub(
        r"```(\w+)?\n(.*?)```",
        lambda m: (
            f'<pre><code class="language-{m.group(1) or "text"}">'
            f"{m.group(2)}</code></pre>"
        ),
        html,
        flags=re.DOTALL,
    )

    # Ссылки [text](url)
    html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)

    # Списки
    lines = html.split("\n")
    result = []
    in_list = False
    for line in lines:
        if line.strip().startswith(
            ("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")
        ):
            if not in_list:
                result.append("<ul>")
                in_list = True
            item = (
                line.strip()[2:]
                if line.strip().startswith(("- ", "* "))
                else line.strip()[3:]
            )
            result.append(f"<li>{item}</li>")
        else:
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(line)
    if in_list:
        result.append("</ul>")
    html = "\n".join(result)

    # Горизонтальная линия
    html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

    # Параграфы
    paragraphs = html.split("\n\n")
    new_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith(("<h", "<ul", "<pre", "<hr", "<li", "</ul>")):
            new_paragraphs.append(f"<p>{p}</p>")
        else:
            new_paragraphs.append(p)
    html = "\n\n".join(new_paragraphs)

    return html


def generate_readme_html(
    input_path: Path = ROOT / "README.md", output_path: Path = ROOT / "README.html"
):
    """Генерация README.html из README.md."""
    readme_path = Path(input_path)
    if not readme_path.exists():
        print(f"Файл {input_path} не найден")
        return

    md_text = readme_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_text)

    html = _templates.get_template("reports/readme.html").render(body_html=body_html)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"README.html сгенерирован: {output_path}")


if __name__ == "__main__":
    generate_readme_html()
