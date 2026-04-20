from __future__ import annotations

import math
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "ppt_designed_assets"
OUTPUT_MD = DOCS_DIR / "组会汇报研究进度_设计增强版.md"
OUTPUT_PPTX = DOCS_DIR / "组会汇报研究进度_设计增强版.pptx"

WIDTH = 1920
HEIGHT = 1080

COLORS = {
    "navy": "#0E223D",
    "navy_2": "#17375E",
    "ink": "#1E2430",
    "paper": "#F7F3EC",
    "paper_2": "#EFE8DD",
    "sky": "#9FC4E9",
    "blue": "#5A8FC8",
    "blue_dark": "#2F5F98",
    "orange": "#C8783B",
    "gold": "#D8B26A",
    "green": "#5E8B6B",
    "line": "#D7CCBC",
    "muted": "#6B7280",
    "white": "#FFFFFF",
    "soft_blue": "#E7EEF7",
    "soft_orange": "#FBF1E5",
    "soft_green": "#EAF2EA",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    preferred = [
        (r"C:\Windows\Fonts\STZHONGS.TTF", False),
        (r"C:\Windows\Fonts\msyhbd.ttc", True),
        (r"C:\Windows\Fonts\simhei.ttf", True),
        (r"C:\Windows\Fonts\msyh.ttc", False),
        (r"C:\Windows\Fonts\simsun.ttc", False),
    ]
    for path, is_bold in preferred:
        if bold and not is_bold and "STZHONGS" not in path.upper():
            continue
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def paragraph_lines(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        for ch in para:
            probe = current + ch
            if draw.textlength(probe, font=fnt) <= max_width or not current:
                current = probe
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 10,
) -> int:
    lines = paragraph_lines(draw, text, fnt, max_width)
    cy = y
    for line in lines:
        draw.text((x, cy), line, font=fnt, fill=fill)
        bbox = draw.textbbox((x, cy), line or "国", font=fnt)
        cy += (bbox[3] - bbox[1]) + line_gap
    return cy


def draw_bullets(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    bullets: list[str],
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    bullet_color: str = COLORS["orange"],
) -> int:
    cy = y
    bullet_gap = 22
    line_gap = 8
    for item in bullets:
        bx = x
        by = cy + 14
        draw.ellipse((bx, by, bx + 12, by + 12), fill=bullet_color)
        cy = draw_text_block(draw, x + 28, cy, item, fnt, fill, max_width - 28, line_gap=line_gap)
        cy += bullet_gap
    return cy


def add_texture(img: Image.Image, strength: int = 18) -> None:
    px = img.load()
    for x in range(0, img.size[0], 8):
        for y in range(0, img.size[1], 8):
            noise = int(((x * 17 + y * 13) % 21) - 10)
            r, g, b, a = px[x, y]
            v = max(-strength, min(strength, noise))
            px[x, y] = (
                max(0, min(255, r + v)),
                max(0, min(255, g + v)),
                max(0, min(255, b + v)),
                a,
            )


def canvas(dark: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    bg = COLORS["navy"] if dark else COLORS["paper"]
    img = Image.new("RGBA", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)
    if dark:
        for radius, color in [(740, "#123056"), (520, "#183E6B"), (340, "#20558A")]:
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.ellipse((WIDTH - radius - 180, -120, WIDTH + 140, radius + 200), fill=color + "55")
            img.alpha_composite(overlay)
    else:
        draw.rectangle((0, 0, 110, HEIGHT), fill=COLORS["navy"])
        draw.rectangle((110, 0, 124, HEIGHT), fill=COLORS["gold"])
        draw.line((150, 110, WIDTH - 120, 110), fill=COLORS["line"], width=2)
        add_texture(img)
    return img, draw


def page_footer(draw: ImageDraw.ImageDraw, page: int, tag: str) -> None:
    tag_font = font(22, bold=True)
    page_font = font(22)
    draw.text((155, HEIGHT - 74), tag.upper(), font=tag_font, fill=COLORS["muted"])
    page_str = f"{page:02d}"
    bbox = draw.textbbox((0, 0), page_str, font=page_font)
    draw.text((WIDTH - 110 - (bbox[2] - bbox[0]), HEIGHT - 74), page_str, font=page_font, fill=COLORS["muted"])


def left_label(draw: ImageDraw.ImageDraw, text: str) -> None:
    lbl = Image.new("RGBA", (600, 110), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lbl)
    ld.text((0, 8), text, font=font(44, bold=True), fill=COLORS["white"])
    rotated = lbl.rotate(90, expand=True)
    draw.bitmap((16, 250), rotated, fill=None)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    x1, y1, x2, y2 = box
    shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x1 + 14, y1 + 14, x2 + 14, y2 + 14), radius=34, fill=(12, 18, 28, 28))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    base = draw._image  # type: ignore[attr-defined]
    base.alpha_composite(shadow)
    draw.rounded_rectangle(box, radius=34, fill=fill, outline=outline or fill, width=2)


def fit_image(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ratio = min(bw / img.size[0], bh / img.size[1])
    resized = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    canvas_img = Image.new("RGB", (bw, bh), COLORS["white"])
    ox = (bw - resized.size[0]) // 2
    oy = (bh - resized.size[1]) // 2
    canvas_img.paste(resized, (ox, oy))
    return canvas_img


def slide_cover(path: Path) -> None:
    img, draw = canvas(dark=True)
    title_font = font(64, bold=True)
    subtitle_font = font(28)
    meta_font = font(24)
    small_font = font(20, bold=True)

    draw.text((160, 150), "PROGRESS REPORT", font=small_font, fill=COLORS["sky"])
    draw.text((160, 230), "基于 CNN 和归一化流的", font=title_font, fill=COLORS["white"])
    draw.text((160, 320), "随机引力波背景参数估计方法", font=title_font, fill=COLORS["white"])
    draw.text((160, 435), "组会汇报研究进度", font=subtitle_font, fill="#DCE8F6")

    draw.line((160, 500, 980, 500), fill=COLORS["gold"], width=3)
    draw.text((160, 560), "导师｜孙翠敏", font=meta_font, fill=COLORS["white"])
    draw.text((160, 610), "汇报时间｜2026 年 4 月 16 日", font=meta_font, fill=COLORS["white"])
    draw.text((160, 660), "主题关键词｜SGWB · BBH 种群 · CNN · Normalizing Flow", font=meta_font, fill="#DCE8F6")

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((1160, 170, 1740, 860), radius=44, outline=(255, 255, 255, 70), width=2)
    od.rounded_rectangle((1220, 230, 1680, 800), radius=36, fill=(255, 255, 255, 30), outline=(255, 255, 255, 60), width=2)
    for i, t in enumerate(["01", "研究进度", "Data · Model · Progress", "LIGO / SGWB / Inference"]):
        od.text((1260, 280 + i * 120), t, font=font(54 if i == 0 else 26, bold=(i < 2)), fill=(255, 255, 255, 220))
    img = img.convert("RGBA")
    img.alpha_composite(overlay)
    img.convert("RGB").save(path)


def slide_outline(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "目录")
    page_footer(draw, 2, "Outline")

    draw.text((170, 150), "汇报提纲", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "本次汇报聚焦：研究动机、方法链路、阶段成果与后续计划", font=font(26), fill=COLORS["muted"])

    items = [
        ("01", "研究背景与科学问题", "为什么要做 SGWB 群体超参数后验估计？"),
        ("02", "数据生成与处理流程", "如何从物理先验构建标注样本与时间序列输入？"),
        ("03", "CNN + 归一化流模型设计", "如何完成条件后验学习与参数采样？"),
        ("04", "当前进度、风险与计划", "目前完成到哪里，下一阶段重点做什么？"),
    ]
    y = 330
    for index, title, desc in items:
        card(draw, (170, y, 1730, y + 140), COLORS["white"], COLORS["line"])
        draw.text((220, y + 34), index, font=font(42, bold=True), fill=COLORS["blue_dark"])
        draw.text((360, y + 28), title, font=font(34, bold=True), fill=COLORS["ink"])
        draw.text((360, y + 80), desc, font=font(24), fill=COLORS["muted"])
        y += 170
    img.save(path)


def slide_text(path: Path, page: int, label: str, title: str, kicker: str, bullets: list[str], right_note: str | None = None) -> None:
    img, draw = canvas()
    left_label(draw, label)
    page_footer(draw, page, label)

    draw.text((170, 150), title, font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), kicker, font=font(26), fill=COLORS["muted"])
    draw.text((170, 300), f"{page:02d}", font=font(150, bold=True), fill=COLORS["paper_2"])

    draw_bullets(draw, 290, 360, bullets, font(30), COLORS["ink"], 900)

    if right_note:
        card(draw, (1220, 280, 1730, 820), COLORS["white"], COLORS["line"])
        draw.text((1265, 330), "核心结论", font=font(32, bold=True), fill=COLORS["blue_dark"])
        draw.line((1265, 380, 1660, 380), fill=COLORS["gold"], width=3)
        draw_text_block(draw, 1265, 430, right_note, font(28), COLORS["ink"], 400, line_gap=14)
    img.save(path)


def slide_workflow(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "方法")
    page_footer(draw, 5, "Workflow")
    draw.text((170, 150), "总体技术路线", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "从物理先验、种群模拟到条件后验估计的端到端方案", font=font(26), fill=COLORS["muted"])

    pipe_img = fit_image(DOCS_DIR / "ppt_assets" / "research_pipeline.png", (170, 320, 1500, 910))
    img.paste(pipe_img, (170, 320))

    card(draw, (1540, 320, 1770, 910), COLORS["white"], COLORS["line"])
    draw.text((1585, 370), "Pipeline", font=font(30, bold=True), fill=COLORS["blue_dark"])
    draw.line((1585, 418, 1715, 418), fill=COLORS["gold"], width=3)
    draw_bullets(
        draw,
        1585,
        470,
        [
            "文献先验约束 10 维群体超参数。",
            "借助 gwfast 生成 12 小时 BBH 种群。",
            "用 gwpy 合成 H1/L1 双探测器时序。",
            "切片形成 segment，并训练条件后验模型。",
        ],
        font(24),
        COLORS["ink"],
        150,
        bullet_color=COLORS["green"],
    )
    img.save(path)


def slide_data(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "数据")
    page_footer(draw, 6, "Dataset")
    draw.text((170, 150), "数据生成与样本构建", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "重点是物理一致的标签设计、时序构造与样本切片", font=font(26), fill=COLORS["muted"])

    specs = [
        ("标签规模", "10000 组", COLORS["soft_blue"], COLORS["blue_dark"]),
        ("标签维度", "10 维", COLORS["soft_orange"], COLORS["orange"]),
        ("观测时长", "12 小时", COLORS["soft_green"], COLORS["green"]),
    ]
    x = 170
    for title, value, fill, accent in specs:
        card(draw, (x, 310, x + 290, 500), fill, accent)
        draw.text((x + 32, 345), title, font=font(28, bold=True), fill=accent)
        draw.text((x + 32, 410), value, font=font(52, bold=True), fill=COLORS["ink"])
        x += 330

    card(draw, (170, 560, 980, 900), COLORS["white"], COLORS["line"])
    draw.text((220, 610), "样本构建流程", font=font(34, bold=True), fill=COLORS["blue_dark"])
    draw_bullets(
        draw,
        220,
        680,
        [
            "采样率设为 2048 Hz，每个 frame 长度为 2048 s。",
            "每个 frame 切分为 8 个 segment，每个 segment 长度 256 s。",
            "每个 segment 继承对应的 10 维超参数标签。",
            "最终样本规模约为 23 × 8 × 10000，单样本维度为 256 × 2048。",
        ],
        font(28),
        COLORS["ink"],
        690,
    )

    card(draw, (1180, 310, 1730, 900), COLORS["navy"], COLORS["navy"])
    draw.text((1230, 370), "设计判断", font=font(32, bold=True), fill=COLORS["white"])
    draw.line((1230, 418, 1500, 418), fill=COLORS["gold"], width=3)
    draw_text_block(
        draw,
        1230,
        470,
        "当前最关键的问题不是模型是否能跑通，而是数据规模是否足以支撑高维后验的稳定学习。因此本阶段优先保证：\n\n1. 标签来源具备物理可解释性；\n2. 时序切片与监督目标严格对齐；\n3. 数据集可持续扩容与复现。",
        font(28),
        "#E9EEF6",
        440,
        line_gap=14,
    )
    img.save(path)


def slide_model(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "模型")
    page_footer(draw, 7, "Model")
    draw.text((170, 150), "CNN + 条件归一化流模型", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "卷积网络负责提取上下文特征，Flow 负责表达复杂后验", font=font(26), fill=COLORS["muted"])

    model_img = fit_image(DOCS_DIR / "ppt_assets" / "model_architecture.png", (170, 310, 1420, 920))
    img.paste(model_img, (170, 310))

    card(draw, (1470, 310, 1760, 920), COLORS["white"], COLORS["line"])
    draw.text((1510, 360), "要点", font=font(30, bold=True), fill=COLORS["blue_dark"])
    draw.line((1510, 408, 1660, 408), fill=COLORS["gold"], width=3)
    draw_bullets(
        draw,
        1510,
        460,
        [
            "输入：单通道长时序波形。",
            "CNN 输出 512 维 context。",
            "Flow 堆叠 8 层自回归变换。",
            "训练目标：最小化负对数似然。",
            "推断阶段可直接采样后验。",
        ],
        font(24),
        COLORS["ink"],
        200,
        bullet_color=COLORS["green"],
    )
    img.save(path)


def slide_plot(path: Path, page: int, label: str, title: str, kicker: str, image_path: Path, notes: list[str], caption: str) -> None:
    img, draw = canvas()
    left_label(draw, label)
    page_footer(draw, page, label)
    draw.text((170, 150), title, font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), kicker, font=font(26), fill=COLORS["muted"])

    card(draw, (170, 300, 1250, 910), COLORS["white"], COLORS["line"])
    plot = fit_image(image_path, (210, 340, 1210, 860))
    img.paste(plot, (210, 340))
    draw.text((210, 880), caption, font=font(22), fill=COLORS["muted"])

    card(draw, (1310, 300, 1760, 910), COLORS["white"], COLORS["line"])
    draw.text((1360, 350), "观测与判断", font=font(32, bold=True), fill=COLORS["blue_dark"])
    draw.line((1360, 398, 1605, 398), fill=COLORS["gold"], width=3)
    draw_bullets(draw, 1360, 460, notes, font(27), COLORS["ink"], 340)
    img.save(path)


def slide_progress(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "进度")
    page_footer(draw, 10, "Progress")
    draw.text((170, 150), "当前进度与阶段成果", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "流程已跑通，当前瓶颈集中在全量训练数据的生成速度", font=font(26), fill=COLORS["muted"])

    metrics = [
        ("流程状态", "已跑通", COLORS["soft_green"], COLORS["green"]),
        ("已处理超参", "2000 组", COLORS["soft_blue"], COLORS["blue_dark"]),
        ("完成比例", "20%", COLORS["soft_orange"], COLORS["orange"]),
        ("模型状态", "已完成初训", COLORS["soft_blue"], COLORS["blue_dark"]),
    ]
    x_positions = [170, 565, 960, 1355]
    for (title, value, fill, accent), x in zip(metrics, x_positions):
        card(draw, (x, 320, x + 250, 520), fill, accent)
        draw.text((x + 24, 354), title, font=font(24, bold=True), fill=accent)
        draw.text((x + 24, 420), value, font=font(42, bold=True), fill=COLORS["ink"])

    card(draw, (170, 600, 1760, 980), COLORS["white"], COLORS["line"])
    draw.text((220, 650), "阶段性结论", font=font(34, bold=True), fill=COLORS["blue_dark"])
    draw.line((220, 700, 430, 700), fill=COLORS["gold"], width=3)
    bullets = [
        "数据生成、标签对齐、样本切分与模型训练已经形成完整闭环。",
        "相关概念与工程流程已经理清，说明技术路线具有可执行性。",
        "模型虽已训练过，但想要得到稳定可信的后验结果，仍依赖更大规模的数据集。",
    ]
    draw_bullets(draw, 220, 745, bullets, font(28), COLORS["ink"], 1450)

    progress_x1, progress_x2, y = 220, 1560, 925
    draw.rounded_rectangle((progress_x1, y, progress_x2, y + 28), radius=14, fill="#ECE6DB")
    fill_x = progress_x1 + int((progress_x2 - progress_x1) * 0.2)
    draw.rounded_rectangle((progress_x1, y, fill_x, y + 28), radius=14, fill=COLORS["blue_dark"])
    draw.text((1590, 908), "20%", font=font(28, bold=True), fill=COLORS["blue_dark"])
    img.save(path)


def slide_plan(path: Path) -> None:
    img, draw = canvas()
    left_label(draw, "计划")
    page_footer(draw, 11, "Plan")
    draw.text((170, 150), "未完成工作与后续计划", font=font(54, bold=True), fill=COLORS["ink"])
    draw.text((170, 225), "先补足数据规模，再进入正式训练、评估与结构调优", font=font(26), fill=COLORS["muted"])

    stages = [
        ("A", "继续生成数据", "完成剩余 8000 组超参数样本，预计仍需约 1 个月。"),
        ("B", "开展正式训练", "在全量数据集上训练模型，观察负对数似然与后验质量。"),
        ("C", "评估并调结构", "根据结果决定是否调整 CNN 主干或 Flow 深度与宽度。"),
    ]
    x = 210
    for idx, (tag, title, desc) in enumerate(stages):
        card(draw, (x, 390, x + 430, 790), COLORS["white"], COLORS["line"])
        draw.ellipse((x + 30, 430, x + 110, 510), fill=COLORS["navy"])
        draw.text((x + 57, 448), tag, font=font(32, bold=True), fill=COLORS["white"])
        draw.text((x + 140, 438), title, font=font(34, bold=True), fill=COLORS["ink"])
        draw_text_block(draw, x + 40, 560, desc, font(28), COLORS["muted"], 350, line_gap=12)
        if idx < 2:
            draw.line((x + 430, 590, x + 500, 590), fill=COLORS["gold"], width=6)
            draw.polygon([(x + 500, 590), (x + 472, 574), (x + 472, 606)], fill=COLORS["gold"])
        x += 530

    card(draw, (170, 840, 1760, 950), COLORS["navy"], COLORS["navy"])
    draw.text((220, 875), "风险提示：数据生成耗时、高维后验训练稳定性、样本分布偏差，三者都需要持续监控。", font=font(28), fill=COLORS["white"])
    img.save(path)


def slide_summary(path: Path) -> None:
    img, draw = canvas(dark=True)
    page_footer(draw, 12, "Summary")

    draw.text((160, 160), "总结", font=font(62, bold=True), fill=COLORS["white"])
    draw.text((160, 250), "当前阶段已经完成“能跑通”到“可训练”的关键跨越。", font=font(32), fill="#DCE8F6")

    bullets = [
        "已完成：物理先验设计、BBH 种群模拟、双探测器时序构建、样本切片与模型实现。",
        "当前重点：继续扩充全量数据，保证后验学习具备足够样本支撑。",
        "下一目标：在完整数据集上得到稳定结果，并据此评估是否调整模型结构。",
    ]
    draw_bullets(draw, 180, 360, bullets, font(30), COLORS["white"], 1000, bullet_color=COLORS["gold"])

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((1190, 250, 1700, 800), radius=42, fill=(255, 255, 255, 28), outline=(255, 255, 255, 80), width=2)
    od.text((1260, 360), "THANK", font=font(68, bold=True), fill=(255, 255, 255, 235))
    od.text((1260, 470), "YOU", font=font(68, bold=True), fill=(255, 255, 255, 235))
    od.text((1260, 620), "敬请批评指正", font=font(28), fill=(220, 232, 246, 240))
    img = img.convert("RGBA")
    img.alpha_composite(overlay)
    img.convert("RGB").save(path)


def build_images() -> list[Path]:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    slides = [
        ASSETS_DIR / "slide01_cover.png",
        ASSETS_DIR / "slide02_outline.png",
        ASSETS_DIR / "slide03_background.png",
        ASSETS_DIR / "slide04_problem.png",
        ASSETS_DIR / "slide05_workflow.png",
        ASSETS_DIR / "slide06_dataset.png",
        ASSETS_DIR / "slide07_model.png",
        ASSETS_DIR / "slide08_hyperparams.png",
        ASSETS_DIR / "slide09_population.png",
        ASSETS_DIR / "slide10_progress.png",
        ASSETS_DIR / "slide11_plan.png",
        ASSETS_DIR / "slide12_summary.png",
    ]

    slide_cover(slides[0])
    slide_outline(slides[1])
    slide_text(
        slides[2],
        3,
        "背景",
        "研究背景与意义",
        "SGWB 为理解双黑洞种群性质与宇宙演化提供了一个统计窗口",
        [
            "随机引力波背景可视为大量不可分辨引力波源的叠加，承载双黑洞种群与宇宙演化信息。",
            "对 SGWB 的群体超参数进行反演，有助于约束质量谱形状、并合率演化及恒星形成历史。",
            "传统贝叶斯参数估计需要高成本波形计算和采样，在长时间序列任务上计算开销较大。",
            "深度学习结合归一化流可以实现 amortized inference，显著提升后验估计效率。",
        ],
        right_note="本课题的核心价值，在于把“物理建模”与“神经后验估计”连接起来：既保留物理先验，又提升推断速度。",
    )
    slide_text(
        slides[3],
        4,
        "问题",
        "关键科学问题",
        "难点不只在模型，更在于高维后验、低信噪比和大规模标注数据的协同处理",
        [
            "输入是 LIGO 双探测器长时间序列，噪声复杂，信号低信噪比。",
            "输出为 10 维群体超参数，参数间存在耦合，后验分布通常非高斯。",
            "训练需要物理一致的大规模标注数据，数据生成成本高、周期长。",
            "目标是学习从时序观测到超参数后验的稳定映射，并保证泛化能力。",
        ],
        right_note="因此，本工作不是单纯做分类或回归，而是要学习条件概率分布 p(θ|x)，这要求模型既有表达能力，也要有稳定训练数据。",
    )
    slide_workflow(slides[4])
    slide_data(slides[5])
    slide_model(slides[6])
    slide_plot(
        slides[7],
        8,
        "分布",
        "超参数标签分布示例",
        "联合分布图用于检查先验采样是否覆盖充分、相关结构是否合理",
        ROOT / "scripts" / "hyperparam_distributions" / "joint_hyperparam_distributions.png",
        [
            "标签覆盖多个超参数维度，便于观察边缘分布与联合相关性。",
            "如果联合分布覆盖不足，模型学习到的后验将受到采样偏差限制。",
            "当前分布图可作为后续数据集扩容的基准参考。",
        ],
        "图：超参数联合分布结果（已有数据样本）。",
    )
    slide_plot(
        slides[8],
        9,
        "结果",
        "双黑洞种群分布示例",
        "种群分布是否合理，直接决定后续时序数据与标签监督是否可信",
        ROOT / "scripts" / "cbc_distributions" / "cbc_distributions.png",
        [
            "图中展示了文献先验约束下生成的 BBH 种群分布。",
            "若种群分布符合预期，说明“先验 → 模拟 → 合成”链条基本可靠。",
            "这为后续训练提供了物理一致的数据基础。",
        ],
        "图：双黑洞种群分布结果（已有模拟样本）。",
    )
    slide_progress(slides[9])
    slide_plan(slides[10])
    slide_summary(slides[11])
    return slides


def write_markdown(slides: list[Path]) -> None:
    lines = []
    for index, slide in enumerate(slides):
        rel = slide.relative_to(ROOT).as_posix()
        lines.append(f"![]({rel}){{ width=13.333in }}")
        if index < len(slides) - 1:
            lines.append("")
            lines.append("---")
            lines.append("")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_ppt() -> None:
    slides = build_images()
    write_markdown(slides)
    subprocess.run(["pandoc", str(OUTPUT_MD), "-o", str(OUTPUT_PPTX)], cwd=ROOT, check=True)
    print(f"Generated {OUTPUT_PPTX}")
    print(f"Slides {len(slides)}")


if __name__ == "__main__":
    build_ppt()
