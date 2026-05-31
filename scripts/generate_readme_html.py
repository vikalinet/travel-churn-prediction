"""Генерация README.html из README.md для GitHub Pages."""

import re
from pathlib import Path


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
        lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
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
    input_path: str = "README.md", output_path: str = "reports/README.html"
):
    """Генерация README.html из README.md."""
    readme_path = Path(input_path)
    if not readme_path.exists():
        print(f"Файл {input_path} не найден")
        return

    md_text = readme_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_text)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Churn Prediction - README</title>
    <base href="/travel-churn-prediction/">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{ color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #764ba2; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
        h3 {{ color: #555; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }}
        pre code {{
            background: none;
            padding: 0;
        }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
        .nav {{
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 2px solid #eee;
        }}
        .nav a {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            margin-right: 10px;
            text-decoration: none;
        }}
        .nav a:hover {{ background: #5568d3; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="index.html">← Назад к отчётам</a>
            <a href="presentation.html">🎬 Презентация</a>
        </div>
        {body_html}
    </div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"README.html сгенерирован: {output_path}")


if __name__ == "__main__":
    generate_readme_html()
