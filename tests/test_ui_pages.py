"""Страница тестирования сценариев (/test).

В разметке оставался кусок старой версии секции положительных сценариев:
лишние закрывающие теги выводили следующие секции из контейнера (страница
растягивалась на весь экран), а старые карточки pos_002–pos_005 подписывали
одни данные, а по щелчку подставляли другие. Браузер такие ошибки прощает
молча — поэтому проверка здесь.
"""

import re
from collections import Counter

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module
from src.api.main import app


def page_html() -> str:
    with TestClient(app) as client:
        response = client.get("/test")
    assert response.status_code == 200
    return response.text


def test_div_tags_are_balanced():
    body = page_html().split("<body", 1)[1]
    depth = 0
    for line in body.splitlines():
        depth += len(re.findall(r"<div[\s>]", line)) - line.count("</div>")
        assert depth >= 0, f"лишний </div>: {line.strip()}"
    assert depth == 0


def test_each_scenario_appears_once():
    scenarios = re.findall(r"applyScenario\('(\w+)'\)", page_html())
    repeated = [name for name, count in Counter(scenarios).items() if count > 1]
    assert scenarios
    assert repeated == []


# ---- страницы с шаблонами ---------------------------------------------------
# Прежде /, /monitoring и /drift читали файл шаблона как текст: на /monitoring
# и /drift пользователь видел сырой код шаблона, а на главной оба значка
# состояния модели показывались одновременно.


@pytest.mark.parametrize("url", ["/", "/monitoring", "/drift", "/test"])
def test_pages_have_no_raw_template_code(url):
    with TestClient(app) as client:
        html = client.get(url).text
    assert "{%" not in html and "{{" not in html


@pytest.mark.parametrize("loaded", [True, False])
def test_home_shows_exactly_one_model_status(loaded, monkeypatch):
    with TestClient(app) as client:
        if not loaded:
            monkeypatch.setattr(main_module, "model", None)
        html = client.get("/").text
    assert ("Модель загружена" in html) is loaded
    assert ("Модель не загружена" in html) is not loaded
