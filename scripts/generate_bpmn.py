"""
BPMN-процессы вокруг travel-churn-model: docs/model_lifecycle.* и docs/model_retrain.*.

Каждый процесс описан один раз — лэйнами, узлами и переходами в PROCESSES ниже;
из них строится файл BPMN 2.0 (открывается в https://demo.bpmn.io и Camunda
Modeler) и SVG-схема для README и GitHub. Рендер один и тот же для обоих
процессов — чтобы добавить диаграмму, достаточно нового Process, без копирования
кода. Шаги жизненного цикла — docs/MODEL_LIFECYCLE.md, шаги переобучения —
docs/Model_retrain_BPMN.md.

    python scripts/generate_bpmn.py
"""

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

DOCS = Path(__file__).resolve().parents[1] / "docs"

Node = Tuple[
    str, str, str, int, int, str
]  # id, тип BPMN, дорожка, колонка, ряд, подпись
Flow = Tuple[str, str, str, object]  # источник, цель, подпись, маршрут (см. route)


@dataclass(frozen=True)
class Process:
    id: str  # префикс id элементов BPMN и имя файлов (<id>.bpmn, <id>.svg)
    pool_title: str
    lanes: List[Tuple[str, str]]
    nodes: List[Node]
    flows: List[Flow]
    label_above: Tuple[str, ...] = ()  # шлюзы, у которых подпись рисуется сверху


# ---- процесс 1: жизненный цикл модели (релиз + мониторинг) --------------

LIFECYCLE_LANES = [
    ("ds", "Data Scientist"),
    ("ci", "CI/CD (GitHub Actions)"),
    ("owner", "Владелец модели"),
    ("prod", "Эксплуатация"),
]

LIFECYCLE_NODES: List[Node] = [
    ("start", "startEvent", "ds", 0, 0, "Нужна новая\nверсия модели"),
    ("prep", "userTask", "ds", 1, 0, "Подготовить\nданные (ETL)"),
    ("train", "userTask", "ds", 2, 0, "Обучить и сравнить\nмодели, версия\nв MLflow"),
    ("gw_metrics", "exclusiveGateway", "ds", 3, 0, "Не хуже\nchampion?"),
    ("card", "userTask", "ds", 4, 0, "Обновить\nModel Card,\nоткрыть PR"),
    ("checks", "serviceTask", "ci", 4, 0, "Линтеры, mypy,\nтесты,\nпокрытие ≥ 60 %"),
    ("gw_ci", "exclusiveGateway", "ci", 5, 0, "CI\nзелёный?"),
    ("review", "userTask", "owner", 6, 0, "Ревью кода,\nметрик и\nModel Card"),
    ("gw_approve", "exclusiveGateway", "owner", 7, 0, "Одобрено?"),
    ("merge", "userTask", "owner", 8, 0, "Merge\nв main"),
    (
        "build",
        "serviceTask",
        "ci",
        9,
        0,
        "Сборка Docker-\nобраза; release: —\nпубликация",
    ),
    ("deploy", "userTask", "prod", 10, 0, "Развернуть,\nпроверить /health\nи версию"),
    ("gw_smoke", "exclusiveGateway", "prod", 11, 0, "Работает?"),
    ("rollback", "userTask", "prod", 11, 1, "Откат на\nпредыдущую\nверсию"),
    ("end_rollback", "endEvent", "prod", 12, 1, "Прежняя версия\nв работе"),
    ("serve", "serviceTask", "prod", 12, 0, "Прогнозы,\nжурнал\nзапросов"),
    ("timer", "timerEvent", "prod", 13, 0, "Ежедневно\n9:00 UTC"),
    ("drift", "serviceTask", "prod", 14, 0, "Проверка\nдрейфа:\nKS, JS, PSI"),
    ("gw_drift", "exclusiveGateway", "prod", 15, 0, "Дрейф?"),
    ("alert", "sendTask", "prod", 16, 0, "Telegram\nи письмо\nGitHub"),
    ("analyze", "userTask", "ds", 16, 0, "Разобрать\nпричину\nдрейфа"),
    ("gw_retrain", "exclusiveGateway", "ds", 17, 0, "Нужно\nпереобучение?"),
    ("end_keep", "endEvent", "ds", 18, 0, "Модель\nостаётся"),
]

LIFECYCLE_FLOWS: List[Flow] = [
    ("start", "prep", "", "h"),
    ("prep", "train", "", "h"),
    ("train", "gw_metrics", "", "h"),
    ("gw_metrics", "card", "да", "h"),
    ("gw_metrics", "train", "нет", ("over", "ds", 22)),
    ("card", "checks", "", "v"),
    ("checks", "gw_ci", "", "h"),
    ("gw_ci", "review", "да", "h"),
    ("gw_ci", "card", "нет", "up_to_side"),
    ("review", "gw_approve", "", "h"),
    ("gw_approve", "merge", "да", "h"),
    ("gw_approve", "card", "нет", ("over", "ds", 40)),
    ("merge", "build", "", "h"),
    ("build", "deploy", "", "h"),
    ("deploy", "gw_smoke", "", "h"),
    ("gw_smoke", "serve", "да", "h"),
    ("gw_smoke", "rollback", "нет", "v"),
    ("rollback", "end_rollback", "", "h"),
    ("serve", "timer", "", "h"),
    ("timer", "drift", "", "h"),
    ("drift", "gw_drift", "", "h"),
    ("gw_drift", "alert", "да", "h"),
    ("gw_drift", "serve", "нет", ("over", "prod", 22)),
    ("alert", "analyze", "", "v"),
    ("analyze", "gw_retrain", "", "h"),
    ("gw_retrain", "end_keep", "нет", "h"),
    ("gw_retrain", "prep", "да", ("over", "ds", 8)),
]

LIFECYCLE = Process(
    id="model_lifecycle",
    pool_title="Жизненный цикл travel-churn-model",
    lanes=LIFECYCLE_LANES,
    nodes=LIFECYCLE_NODES,
    flows=LIFECYCLE_FLOWS,
    label_above=("gw_smoke",),
)

# ---- процесс 2: автоматическое переобучение (триггер → деплой) ----------
#
# Тот же конвейер, что в LIFECYCLE (п. 1–13), но в рамке «от триггера до
# развёртывания новой версии»: без цикла обслуживания и ежедневного дрейфа
# (он описан один раз в model_lifecycle, здесь — только то, что запускает
# переобучение и что происходит, если проверки или деплой не прошли).

RETRAIN_NODES: List[Node] = [
    (
        "start",
        "startEvent",
        "ds",
        0,
        0,
        "Триггер: дрейф /\nновые данные /\nплановый запуск",
    ),
    ("prep", "userTask", "ds", 1, 0, "Получить и\nподготовить\nновые данные"),
    ("train", "userTask", "ds", 2, 0, "Обучить новую\nверсию модели"),
    ("gw_quality", "exclusiveGateway", "ds", 3, 0, "Качество не\nхуже champion?"),
    ("card", "userTask", "ds", 4, 0, "Обновить\nModel Card,\nоткрыть PR"),
    ("checks", "serviceTask", "ci", 4, 0, "Линтеры, тесты,\nпокрытие ≥ 60 %"),
    ("gw_ci", "exclusiveGateway", "ci", 5, 0, "CI\nзелёный?"),
    ("review", "userTask", "owner", 6, 0, "Ревью: код,\nметрики,\nModel Card"),
    ("gw_publish", "exclusiveGateway", "owner", 7, 0, "Публиковать\nновую версию?"),
    (
        "register",
        "userTask",
        "owner",
        8,
        0,
        "Merge: версия\nстановится\nchampion в MLflow",
    ),
    ("build", "serviceTask", "ci", 9, 0, "Сборка и\nпубликация\nDocker-образа"),
    ("deploy", "userTask", "prod", 10, 0, "Развернуть,\nпроверить /health\nи версию"),
    ("gw_health", "exclusiveGateway", "prod", 11, 0, "Работает?"),
    ("rollback", "userTask", "prod", 11, 1, "Откат на\nпредыдущую\nверсию"),
    ("end_fail", "endEvent", "prod", 12, 1, "Прежняя версия\nосталась; разбор\nпричин"),
    ("end_ok", "endEvent", "prod", 12, 0, "Новая версия\nв эксплуатации"),
]

RETRAIN_FLOWS: List[Flow] = [
    ("start", "prep", "", "h"),
    ("prep", "train", "", "h"),
    ("train", "gw_quality", "", "h"),
    ("gw_quality", "card", "да", "h"),
    ("gw_quality", "train", "нет, доработать", ("over", "ds", 22)),
    ("card", "checks", "", "v"),
    ("checks", "gw_ci", "", "h"),
    ("gw_ci", "review", "да", "h"),
    ("gw_ci", "card", "нет", "up_to_side"),
    ("review", "gw_publish", "", "h"),
    ("gw_publish", "register", "да", "h"),
    ("gw_publish", "card", "нет, доработка", ("over", "ds", 40)),
    ("register", "build", "", "h"),
    ("build", "deploy", "", "h"),
    ("deploy", "gw_health", "", "h"),
    ("gw_health", "end_ok", "да", "h"),
    ("gw_health", "rollback", "нет", "v"),
    ("rollback", "end_fail", "", "h"),
]

RETRAIN = Process(
    id="model_retrain",
    pool_title="Переобучение travel-churn-model",
    lanes=LIFECYCLE_LANES,
    nodes=RETRAIN_NODES,
    flows=RETRAIN_FLOWS,
    label_above=("gw_health",),
)

PROCESSES = [LIFECYCLE, RETRAIN]

# ---- разметка ------------------------------------------------------------

POOL_X, POOL_Y = 20, 20
POOL_LABEL_W, LANE_LABEL_W = 30, 34
CONTENT_X = POOL_X + POOL_LABEL_W + LANE_LABEL_W
COL_W = 140
LANE_H = 220
ROW_Y = (85, 160)  # центры рядов от верха дорожки
SIZES = {
    "task": (112, 66),
    "exclusiveGateway": (46, 46),
    "event": (36, 36),
}

Box = Tuple[float, float, float, float]  # x, y, w, h
Point = Tuple[float, float]


def kind(node_type: str) -> str:
    if node_type.endswith("Task"):
        return "task"
    if node_type == "exclusiveGateway":
        return node_type
    return "event"


def lane_top(proc: Process, lane: str) -> float:
    return POOL_Y + [lid for lid, _ in proc.lanes].index(lane) * LANE_H


def pool_size(proc: Process) -> Tuple[float, float]:
    width = (
        POOL_LABEL_W + LANE_LABEL_W + COL_W * (max(n[3] for n in proc.nodes) + 1) + 10
    )
    return width, LANE_H * len(proc.lanes)


def layout(proc: Process) -> Dict[str, Box]:
    boxes: Dict[str, Box] = {}
    for nid, ntype, lane, col, row, _ in proc.nodes:
        w, h = SIZES[kind(ntype)]
        cx = CONTENT_X + COL_W * col + COL_W / 2
        cy = lane_top(proc, lane) + ROW_Y[row]
        boxes[nid] = (cx - w / 2, cy - h / 2, w, h)
    return boxes


def route(proc: Process, src: Box, dst: Box, how) -> List[Point]:
    """Ломаная перехода.

    h — вправо (с изломом, если цель в другой дорожке); v — вертикально;
    up_to_side — вверх и в правый бок цели; ("over", дорожка, отступ) —
    петля назад поверх узлов на заданной высоте от верха дорожки.
    """
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    scx, scy, dcx, dcy = sx + sw / 2, sy + sh / 2, dx + dw / 2, dy + dh / 2
    if how == "h":
        if abs(scy - dcy) < 1:
            return [(sx + sw, scy), (dx, dcy)]
        mid = (sx + sw + dx) / 2
        return [(sx + sw, scy), (mid, scy), (mid, dcy), (dx, dcy)]
    if how == "v":
        if dcy > scy:
            return [(scx, sy + sh), (dcx, dy)]
        return [(scx, sy), (dcx, dy + dh)]
    if how == "up_to_side":
        return [(scx, sy), (scx, dcy), (dx + dw, dcy)]
    _, lane, offset = how
    top = lane_top(proc, lane) + offset
    return [(scx, sy), (scx, top), (dcx, top), (dcx, dy)]


def flow_points(proc: Process, boxes: Dict[str, Box]) -> List[List[Point]]:
    return [route(proc, boxes[s], boxes[t], how) for s, t, _, how in proc.flows]


# ---- BPMN 2.0 ------------------------------------------------------------


def tag(element: str, inner: str = "", **attrs) -> str:
    """XML-элемент; «__» в имени атрибута — двоеточие (xmlns__bpmn)."""
    text = "".join(f' {k.replace("__", ":")}="{v}"' for k, v in attrs.items())
    if not inner:
        return f"<{element}{text} />"
    return f"<{element}{text}>{inner}</{element}>"


def one_line(content: str) -> str:
    return escape(" ".join(content.split("\n")))


def bounds(x: float, y: float, w: float, h: float) -> str:
    return tag("dc:Bounds", x=x, y=y, width=w, height=h)


TIMER = tag(
    "bpmn:timerEventDefinition",
    tag("bpmn:timeCycle", "R/P1D", xsi__type="bpmn:tFormalExpression"),
    id="timer_def",
)


def process_xml(proc: Process) -> List[str]:
    incoming = {n[0]: "" for n in proc.nodes}
    outgoing = {n[0]: "" for n in proc.nodes}
    for i, (s, t, _, _) in enumerate(proc.flows):
        outgoing[s] += tag("bpmn:outgoing", f"flow_{i}")
        incoming[t] += tag("bpmn:incoming", f"flow_{i}")

    lanes = [
        tag(
            "bpmn:lane",
            "".join(tag("bpmn:flowNodeRef", n[0]) for n in proc.nodes if n[2] == lid),
            id=f"lane_{lid}",
            name=escape(lname),
        )
        for lid, lname in proc.lanes
    ]
    nodes = []
    for nid, ntype, *_, name in proc.nodes:
        inner = incoming[nid] + outgoing[nid]
        if ntype == "timerEvent":
            ntype, inner = "intermediateCatchEvent", inner + TIMER
        nodes.append(tag(f"bpmn:{ntype}", inner, id=nid, name=one_line(name)))
    sequence = []
    for i, (s, t, label, _) in enumerate(proc.flows):
        extra = {"name": label} if label else {}
        sequence.append(
            tag("bpmn:sequenceFlow", id=f"flow_{i}", sourceRef=s, targetRef=t, **extra)
        )
    return [tag("bpmn:laneSet", "".join(lanes), id="lanes"), *nodes, *sequence]


def diagram_xml(
    proc: Process, boxes: Dict[str, Box], points: List[List[Point]]
) -> List[str]:
    shape = "bpmndi:BPMNShape"
    pool_w, pool_h = pool_size(proc)
    out = [
        tag(
            shape,
            bounds(POOL_X, POOL_Y, pool_w, pool_h),
            id="pool_di",
            bpmnElement="pool",
            isHorizontal="true",
        )
    ]
    for lid, _ in proc.lanes:
        lane_box = bounds(
            POOL_X + POOL_LABEL_W, lane_top(proc, lid), pool_w - POOL_LABEL_W, LANE_H
        )
        out.append(
            tag(
                shape,
                lane_box,
                id=f"lane_{lid}_di",
                bpmnElement=f"lane_{lid}",
                isHorizontal="true",
            )
        )
    for nid, ntype, *_ in proc.nodes:
        x, y, w, h = boxes[nid]
        inner = bounds(x, y, w, h)
        if kind(ntype) != "task":  # подпись под событием или шлюзом
            label_y = y - 32 if nid in proc.label_above else y + h + 4
            inner += tag("bpmndi:BPMNLabel", bounds(x + w / 2 - 50, label_y, 100, 28))
        out.append(tag(shape, inner, id=f"{nid}_di", bpmnElement=nid))
    for i, pts in enumerate(points):
        waypoints = "".join(tag("di:waypoint", x=x, y=y) for x, y in pts)
        out.append(
            tag(
                "bpmndi:BPMNEdge", waypoints, id=f"flow_{i}_di", bpmnElement=f"flow_{i}"
            )
        )
    return out


NAMESPACES = {
    "xmlns__bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "xmlns__bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "xmlns__dc": "http://www.omg.org/spec/DD/20100524/DC",
    "xmlns__di": "http://www.omg.org/spec/DD/20100524/DI",
    "xmlns__xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def block(lines: List[str]) -> str:
    return "\n" + "\n".join(lines) + "\n"


def bpmn(proc: Process, boxes: Dict[str, Box], points: List[List[Point]]) -> str:
    process_id = f"{proc.id}_process"
    participant = tag(
        "bpmn:participant",
        id="pool",
        name=proc.pool_title,
        processRef=process_id,
    )
    process = tag(
        "bpmn:process",
        block(process_xml(proc)),
        id=process_id,
        name=proc.pool_title,
        isExecutable="false",
    )
    plane = tag(
        "bpmndi:BPMNPlane",
        block(diagram_xml(proc, boxes, points)),
        id="plane",
        bpmnElement="collaboration",
    )
    definitions = tag(
        "bpmn:definitions",
        block(
            [
                tag("bpmn:collaboration", participant, id="collaboration"),
                process,
                tag("bpmndi:BPMNDiagram", plane, id="diagram"),
            ]
        ),
        id=proc.id,
        targetNamespace="https://github.com/vikalinet/travel-churn-prediction",
        **NAMESPACES,
    )
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + definitions + "\n"


# ---- SVG -----------------------------------------------------------------

LANE_FILL = {"ds": "#eef4fb", "ci": "#f3f3f3", "owner": "#fbf3e8", "prod": "#edf7ef"}
INK = "#2b2f36"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
TASK_ICON = {"serviceTask": "⚙", "userTask": "👤", "sendTask": "✉"}


def text(x: float, y: float, content: str, size: int = 11, **attrs) -> str:
    """Текст SVG; «_» в имени атрибута — дефис (text_anchor → text-anchor)."""
    attrs.setdefault("text_anchor", "middle")
    attrs.setdefault("fill", INK)
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="{FONT}"'
        f"{extra}>{escape(content)}</text>"
    )


def text_lines(cx: float, cy: float, content: str, size: int = 11) -> str:
    lines = content.split("\n")
    top = cy - (len(lines) - 1) * size * 0.6 + size * 0.35
    return "".join(
        text(cx, top + i * size * 1.2, line, size) for i, line in enumerate(lines)
    )


def svg_lanes(proc: Process, pool_w: float) -> List[str]:
    out = []
    for lid, lname in proc.lanes:
        y = lane_top(proc, lid)
        out.append(
            f'<rect x="{POOL_X + POOL_LABEL_W}" y="{y}" '
            f'width="{pool_w - POOL_LABEL_W}" height="{LANE_H}" '
            f'fill="{LANE_FILL[lid]}" stroke="{INK}"/>'
        )
        lx, ly = POOL_X + POOL_LABEL_W + LANE_LABEL_W / 2, y + LANE_H / 2
        out.append(
            text(lx, ly, lname, 13, transform=f"rotate(-90 {lx} {ly})", dy="0.35em")
        )
        out.append(
            f'<line x1="{CONTENT_X}" y1="{y}" x2="{CONTENT_X}" y2="{y + LANE_H}" '
            f'stroke="{INK}"/>'
        )
    return out


def svg_flow(points: List[Point], label: str) -> str:
    path = " ".join(f"{x},{y}" for x, y in points)
    out = (
        f'<polyline points="{path}" fill="none" stroke="{INK}" stroke-width="1.3" '
        'marker-end="url(#arrow)"/>'
    )
    if label:
        (x0, y0), (_, y1) = points[0], points[1]
        if abs(y1 - y0) < 1:  # горизонтальный выход — подпись над линией
            ly = y0 - 6
        else:  # вертикальный выход — подпись справа от линии
            ly = y0 + (-10 if y1 < y0 else 16)
        out += text(x0 + 6, ly, label, 11, text_anchor="start", fill="#555")
    return out


def svg_node(box: Box, ntype: str, name: str, label_above: bool) -> str:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    shape = f'fill="#ffffff" stroke="{INK}" stroke-width="1.5"'
    if kind(ntype) == "task":
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" {shape}/>'
            + text(x + 6, y + 14, TASK_ICON[ntype], 11, text_anchor="start")
            + text_lines(cx, cy + 3, name)
        )
    if ntype == "exclusiveGateway":
        diamond = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
        out = f'<polygon points="{diamond}" {shape}/>' + text(cx, cy + 6, "×", 18)
    else:
        border = 3.5 if ntype == "endEvent" else 1.5
        out = (
            f'<circle cx="{cx}" cy="{cy}" r="{w / 2}" fill="#ffffff" '
            f'stroke="{INK}" stroke-width="{border}"/>'
        )
        if ntype == "timerEvent":
            out += (
                f'<circle cx="{cx}" cy="{cy}" r="{w / 2 - 4}" fill="none" '
                f'stroke="{INK}"/><polyline points="{cx},{cy - 9} {cx},{cy} '
                f'{cx + 7},{cy}" fill="none" stroke="{INK}"/>'
            )
    return out + text_lines(cx, y - 24 if label_above else y + h + 16, name)


def svg(proc: Process, boxes: Dict[str, Box], points: List[List[Point]]) -> str:
    pool_w, pool_h = pool_size(proc)
    w, h = POOL_X * 2 + pool_w, POOL_Y * 2 + pool_h
    pcx, pcy = POOL_X + POOL_LABEL_W / 2, POOL_Y + pool_h / 2
    pool_title = text(
        pcx,
        pcy,
        proc.pool_title,
        14,
        font_weight="600",
        transform=f"rotate(-90 {pcx} {pcy})",
        dy="0.35em",
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<rect x="{POOL_X}" y="{POOL_Y}" width="{pool_w}" height="{pool_h}" '
        f'fill="none" stroke="{INK}" stroke-width="1.5"/>',
        pool_title,
        *svg_lanes(proc, pool_w),
        *(svg_flow(pts, flow[2]) for flow, pts in zip(proc.flows, points)),
        *(
            svg_node(boxes[nid], ntype, name, nid in proc.label_above)
            for nid, ntype, *_, name in proc.nodes
        ),
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def generate(proc: Process) -> None:
    boxes = layout(proc)
    points = flow_points(proc, boxes)
    (DOCS / f"{proc.id}.bpmn").write_text(bpmn(proc, boxes, points), encoding="utf-8")
    (DOCS / f"{proc.id}.svg").write_text(svg(proc, boxes, points), encoding="utf-8")
    print(f"docs/{proc.id}.bpmn и docs/{proc.id}.svg обновлены")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for proc in PROCESSES:
        generate(proc)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
