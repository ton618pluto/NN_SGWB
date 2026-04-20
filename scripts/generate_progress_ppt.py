from __future__ import annotations

import os
import subprocess
import textwrap
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "ppt_assets"
OUTPUT_MD = DOCS_DIR / "组会汇报研究进度_学术版.md"
OUTPUT_PPTX = DOCS_DIR / "组会汇报研究进度_学术版.pptx"


PALETTE = {
    "navy": "#17375E",
    "blue": "#2F5597",
    "sky": "#5B9BD5",
    "teal": "#3E6C8F",
    "green": "#70AD47",
    "orange": "#ED7D31",
    "light_blue": "#EAF1FB",
    "light_gray": "#F5F7FA",
    "mid_gray": "#D9E2F2",
    "dark": "#1F1F1F",
    "muted": "#5B6573",
    "white": "#FFFFFF",
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\SourceHanSansSC-Bold.otf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    )
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            trial = current + char
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_spacing: int = 8,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, font, max_width)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def draw_box(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    body: str | None = None,
    fill: str = PALETTE["light_blue"],
    outline: str = PALETTE["blue"],
) -> None:
    x1, y1, x2, y2 = rect
    draw.rounded_rectangle(rect, radius=26, fill=fill, outline=outline, width=4)
    title_font = get_font(28, bold=True)
    body_font = get_font(22)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = x1 + (x2 - x1 - (title_bbox[2] - title_bbox[0])) / 2
    draw.text((title_x, y1 + 18), title, font=title_font, fill=PALETTE["navy"])
    if body:
        draw_multiline_text(
            draw,
            (x1 + 22, y1 + 70),
            body,
            body_font,
            PALETTE["dark"],
            max_width=(x2 - x1 - 44),
            line_spacing=6,
        )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = PALETTE["blue"],
    width: int = 8,
) -> None:
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - 22 * direction, ey - 12), (ex - 22 * direction, ey + 12)]
    else:
        direction = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 12, ey - 22 * direction), (ex + 12, ey - 22 * direction)]
    draw.polygon(points, fill=color)


def add_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    title_font = get_font(42, bold=True)
    subtitle_font = get_font(24)
    draw.text((70, 40), title, font=title_font, fill=PALETTE["navy"])
    draw.text((70, 100), subtitle, font=subtitle_font, fill=PALETTE["muted"])
    draw.line((70, 145, width - 70, 145), fill=PALETTE["sky"], width=5)


def generate_pipeline_diagram(path: Path) -> None:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), PALETTE["white"])
    draw = ImageDraw.Draw(img)
    add_header(draw, "总体技术路线", "从物理先验、波形模拟到条件后验估计的端到端流程", width)

    top_boxes = [
        ("文献先验", "依据质量谱参数化与恒星形成历史\n设置物理可行先验"),
        ("超参数采样", "生成 10000 组 10维标签\n覆盖质量谱与并合率演化"),
        ("gwfast 模拟", "按超参数生成 BBH 种群\n观测时长 12 小时"),
        ("gwpy 合成", "拟合波形并注入 H1/L1\n得到双探测器时序"),
    ]
    bottom_boxes = [
        ("Frame 划分", "采样率 2048 Hz\n每个 frame 为 2048 s"),
        ("Segment 切片", "每个 frame 切为 8 段\n每段长度 256 s"),
        ("标签对齐", "为每个 segment 绑定\n对应的 10 维超参数"),
        ("CNN + Flow", "提取上下文特征并学习\n超参数条件后验"),
    ]

    box_w, box_h = 300, 170
    x_positions = [90, 460, 830, 1200]
    y_top, y_bottom = 210, 520

    for idx, (title, body) in enumerate(top_boxes):
        draw_box(draw, (x_positions[idx], y_top, x_positions[idx] + box_w, y_top + box_h), title, body)
    for idx, (title, body) in enumerate(bottom_boxes):
        draw_box(draw, (x_positions[idx], y_bottom, x_positions[idx] + box_w, y_bottom + box_h), title, body)

    for idx in range(3):
        draw_arrow(draw, (x_positions[idx] + box_w, y_top + box_h // 2), (x_positions[idx + 1], y_top + box_h // 2))
        draw_arrow(
            draw,
            (x_positions[idx] + box_w, y_bottom + box_h // 2),
            (x_positions[idx + 1], y_bottom + box_h // 2),
        )

    for idx in range(4):
        draw_arrow(draw, (x_positions[idx] + box_w // 2, y_top + box_h), (x_positions[idx] + box_w // 2, y_bottom))

    footer_font = get_font(24)
    draw.rounded_rectangle((90, 760, 1510, 840), radius=20, fill=PALETTE["light_gray"], outline=PALETTE["mid_gray"], width=2)
    footer_text = "输入样本形状：1 × 524288（256 s × 2048 Hz）；输出目标：10 维双黑洞种群超参数后验"
    draw.text((120, 788), footer_text, font=footer_font, fill=PALETTE["navy"])
    img.save(path)


def generate_model_diagram(path: Path) -> None:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), PALETTE["white"])
    draw = ImageDraw.Draw(img)
    add_header(draw, "模型结构图", "一维卷积特征提取器与条件归一化流的串联架构", width)

    left_x1, left_x2 = 80, 760
    right_x1, right_x2 = 920, 1510
    stage_w = left_x2 - left_x1

    draw_box(draw, (left_x1, 190, left_x2, 285), "输入波形", "单通道时序输入\n长度 524288", fill="#F5F9FF")
    conv_blocks = [
        ("Conv Block 1", "Conv1d 1→16, k=64, s=4\nBN + ReLU + MaxPool(4)\n长度快速压缩"),
        ("Conv Block 2", "Conv1d 16→32, k=16, s=2\nBN + ReLU + MaxPool(4)\n提取多尺度局部模式"),
        ("Conv Block 3", "Conv1d 32→64, k=8, s=2\nBN + ReLU + MaxPool(4)\n输出 64 × 512 特征图"),
        ("全连接层", "Flatten = 32768\nFC 32768→256 + Dropout\n输出上下文向量 512"),
    ]
    current_y = 320
    block_h = 115
    for title, body in conv_blocks:
        draw_box(draw, (left_x1, current_y, left_x2, current_y + block_h), title, body)
        current_y += 140

    for y in [285, 435, 575, 715]:
        draw_arrow(draw, ((left_x1 + left_x2) // 2, y), ((left_x1 + left_x2) // 2, y + 35))

    draw_box(draw, (right_x1, 250, right_x2, 365), "条件输入", "θ：10 维超参数\ncontext：CNN 输出 512 维", fill="#F7FBF4", outline=PALETTE["green"])
    draw_box(
        draw,
        (right_x1, 420, right_x2, 610),
        "归一化流主干",
        "8 层重复堆叠：\nRandomPermutation\n+ MaskedPiecewiseRationalQuadratic\nAutoregressiveTransform\nhidden_features=256, num_bins=8,\ntail_bound=3.0",
        fill="#F7FBF4",
        outline=PALETTE["green"],
    )
    draw_box(draw, (right_x1, 665, right_x2, 780), "输出", "后验对数似然 / 参数采样\n用于训练与推断", fill="#F7FBF4", outline=PALETTE["green"])
    draw_arrow(draw, ((right_x1 + right_x2) // 2, 365), ((right_x1 + right_x2) // 2, 420), color=PALETTE["green"])
    draw_arrow(draw, ((right_x1 + right_x2) // 2, 610), ((right_x1 + right_x2) // 2, 665), color=PALETTE["green"])

    draw_arrow(draw, (left_x2, 675), (right_x1, 310))
    note_font = get_font(24)
    draw.rounded_rectangle((840, 120, 1520, 190), radius=18, fill=PALETTE["light_gray"], outline=PALETTE["mid_gray"], width=2)
    draw.text((870, 145), "训练目标：最小化负对数似然，学习 p(θ | x)", font=note_font, fill=PALETTE["navy"])
    img.save(path)


def generate_progress_dashboard(path: Path) -> None:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), PALETTE["white"])
    draw = ImageDraw.Draw(img)
    add_header(draw, "时间计划与当前进度", "当前以数据生成扩容为主，随后进入正式训练与结构调优", width)

    title_font = get_font(28, bold=True)
    body_font = get_font(24)
    huge_font = get_font(62, bold=True)

    cards = [
        ((90, 190, 470, 420), "已处理超参", "2000 组", "约占总任务的 20%"),
        ((520, 190, 900, 420), "剩余任务量", "8000 组", "仍需持续批量生成"),
        ((950, 190, 1510, 420), "预计耗时", "约 1 个月", "完成数据集主体构建"),
    ]
    for rect, title, main_value, sub in cards:
        draw.rounded_rectangle(rect, radius=28, fill=PALETTE["light_blue"], outline=PALETTE["blue"], width=3)
        x1, y1, x2, y2 = rect
        draw.text((x1 + 28, y1 + 28), title, font=title_font, fill=PALETTE["navy"])
        draw.text((x1 + 28, y1 + 102), main_value, font=huge_font, fill=PALETTE["blue"])
        draw.text((x1 + 28, y1 + 180), sub, font=body_font, fill=PALETTE["muted"])

    draw.rounded_rectangle((110, 520, 1490, 585), radius=22, fill="#E6ECF5", outline=PALETTE["mid_gray"], width=2)
    draw.rounded_rectangle((110, 520, 386, 585), radius=22, fill=PALETTE["blue"], outline=PALETTE["blue"], width=2)
    draw.text((720, 528), "数据生成总体进度：20%", font=get_font(26, bold=True), fill=PALETTE["navy"])

    milestones = [
        (180, "当前阶段", "继续生成剩余 8000 组超参样本"),
        (690, "下一阶段", "完成全量数据集并启动正式训练"),
        (1180, "最终阶段", "依据后验质量决定是否调整模型结构"),
    ]
    for x, title, body in milestones:
        draw.ellipse((x, 660, x + 32, 692), fill=PALETTE["orange"], outline=PALETTE["orange"])
        draw.line((x + 16, 692, x + 16, 740), fill=PALETTE["orange"], width=4)
        draw_box(draw, (x - 115, 742, x + 155, 840), title, body, fill="#FFF8F2", outline=PALETTE["orange"])

    draw.line((196, 676, 1200, 676), fill=PALETTE["orange"], width=5)
    img.save(path)


def write_markdown(path: Path) -> None:
    content = textwrap.dedent(
        """\
        ---
        title: 基于CNN和归一化流的随机引力波背景参数估计方法
        subtitle: 组会汇报研究进度
        author: 导师：孙翠敏
        date: 2026年4月16日
        lang: zh-CN
        ---

        # 汇报提纲

        - 研究背景与核心科学问题
        - 数据生成、处理流程与模型设计
        - 当前已完成进度与阶段性结果
        - 后续计划、风险点与下一步安排

        # 研究背景与意义

        - 随机引力波背景（SGWB）可视为大量不可分辨引力波源的叠加，蕴含双黑洞种群与宇宙演化信息。
        - 对 SGWB 相关超参数进行反演，有助于约束质量谱形状、并合率演化与恒星形成历史。
        - 传统贝叶斯参数估计依赖高成本波形计算与采样，在长时间序列上计算开销较大。
        - 结合深度学习与归一化流，有望实现 amortized inference，提升后验估计效率。

        # 关键科学问题

        - 输入是 LIGO 双探测器长时间序列，低信噪比与噪声复杂性并存。
        - 输出为 10 维群体超参数，参数之间存在显著耦合，后验分布往往非高斯。
        - 训练需要物理一致的大规模标注数据，数据生成成本高、周期长。
        - 目标是学习“时序观测 → 超参数后验”的稳定映射，并兼顾可解释性与泛化性。

        # 总体技术路线

        - 以文献约束的物理先验构建标签，再生成双黑洞种群和双探测器时序。
        - 最终使用 CNN 提取上下文表示，再通过条件归一化流完成后验建模。

        ![](docs/ppt_assets/research_pipeline.png){ width=9.0in }

        # 数据生成与标签设计

        - 参考《Measuring the Binary Black Hole Mass Spectrum with an Astrophysically Motivated Parameterization》和《Cosmic Star-Formation History》构造物理合理先验。
        - 共生成 10000 组超参数标签，每组标签维度为 10。
        - 以超参数为输入，借助 `gwfast` 生成符合物理规律的双黑洞种群，模拟时长为 12 小时。
        - 这一阶段的重点是确保标签分布与种群模拟过程具备物理一致性。

        # 探测器时序构建与样本切分

        - 使用 `gwpy` 为每个双黑洞系统拟合波形，并生成 LIGO 双探测器时间序列。
        - 采样率设置为 2048 Hz，每个 frame 的长度为 2048 s。
        - 每个 frame 再切分为 8 个 segment，每个 segment 的长度为 256 s。
        - 每个 segment 继承对应的 10 维超参数标签，最终形成监督训练样本。
        - 数据集规模约为 `23 × 8 × 10000`，单个样本的特征维度为 `256 × 2048`。

        # CNN 特征提取器

        - 输入为单通道长时序，采用三级一维卷积块逐步压缩时间维度并提取局部模式。
        - 卷积主干后接全连接层，将高维特征压缩为 512 维上下文向量。
        - Dropout 用于抑制过拟合，增强后续后验建模的稳定性。

        ![](docs/ppt_assets/model_architecture.png){ width=9.0in }

        # 条件归一化流后验建模

        - 将 CNN 输出的 512 维上下文向量作为条件输入，与 10 维超参数共同送入流模型。
        - 流模型由 8 层堆叠组成，每层包含 `RandomPermutation` 与 `MaskedPiecewiseRationalQuadraticAutoregressiveTransform`。
        - 关键超参数包括：`hidden_features=256`、`num_bins=8`、`tail_bound=3.0`。
        - 训练阶段通过最小化负对数似然实现后验学习；推断阶段可直接生成参数样本。

        # 数据分布示例：超参数标签

        - 已生成的标签覆盖多个超参数维度，可用于初步检查先验采样的完整性与相关性结构。
        - 联合分布图有助于验证样本空间覆盖是否合理，并为后续训练稳定性提供基础。

        ![](scripts/hyperparam_distributions/joint_hyperparam_distributions.png){ width=8.4in }

        # 数据分布示例：双黑洞种群

        - 种群分布图展示了在文献约束下生成的双黑洞样本是否符合预期的质量分布形态。
        - 该结果为“物理可行标签 → 种群模拟 → 时序合成”的链条提供了直接证据。

        ![](scripts/cbc_distributions/cbc_distributions.png){ width=8.4in }

        # 当前进度与阶段性结果

        - 数据生成与处理流程已经跑通，能够从超参数生成标注样本。
        - 相关物理背景、建模思路与实现细节已经完成梳理。
        - CNN + Flow 模型已经完成搭建，并进行了初步训练验证。
        - 当前尚未得到最终稳定结果，主要原因是全量训练数据仍在持续生成中。

        # 未完成工作与时间计划

        - 目前已处理完成 2000 组超参数，对应总任务的 20%。
        - 剩余 8000 组样本预计还需约 1 个月生成完毕。
        - 数据集完成后，将进入正式训练、后验质量评估与结构调优阶段。

        ![](docs/ppt_assets/progress_dashboard.png){ width=9.0in }

        # 风险点与拟解决方案

        - 数据生成耗时较长：继续采用批量生成与过程监控，优先保证样本质量。
        - 高维后验训练可能不稳定：通过学习率、批大小、正则化和验证集策略进行调节。
        - 数据分布偏差可能影响泛化：持续检查边缘分布与联合分布，避免训练集失衡。
        - 若现有结构容量不足：基于训练结果决定是否调整 CNN 主干或流模型深度。

        # 总结

        - 本阶段已经搭建起从物理先验、双黑洞种群模拟、探测器时序构建到后验学习的完整流程。
        - 已完成的重点工作包括：标签设计、数据合成、样本切分与 CNN + Flow 模型实现。
        - 下一阶段的核心任务是扩充全量数据、完成正式训练，并评估超参数后验估计效果。
        - 汇报完毕，敬请批评指正。
        """
    )
    path.write_text(content, encoding="utf-8")


def patch_theme(pptx_path: Path) -> None:
    replacements = {
        'name="Office Theme"': 'name="Academic Theme"',
        'name="Office"': 'name="Academic"',
        '<a:srgbClr val="1F497D"/>': '<a:srgbClr val="17375E"/>',
        '<a:srgbClr val="EEECE1"/>': '<a:srgbClr val="F5F7FA"/>',
        '<a:srgbClr val="4F81BD"/>': '<a:srgbClr val="2F5597"/>',
        '<a:srgbClr val="C0504D"/>': '<a:srgbClr val="5B9BD5"/>',
        '<a:srgbClr val="9BBB59"/>': '<a:srgbClr val="70AD47"/>',
        '<a:srgbClr val="8064A2"/>': '<a:srgbClr val="44546A"/>',
        '<a:srgbClr val="4BACC6"/>': '<a:srgbClr val="3E6C8F"/>',
        '<a:srgbClr val="F79646"/>': '<a:srgbClr val="ED7D31"/>',
        '<a:latin typeface="Calibri"/>': '<a:latin typeface="Aptos"/>',
        '<a:ea typeface=""/>': '<a:ea typeface="Microsoft YaHei"/>',
        '<a:font script="Hans" typeface="宋体"/>': '<a:font script="Hans" typeface="Microsoft YaHei"/>',
        '<a:font script="Hant" typeface="新細明體"/>': '<a:font script="Hant" typeface="Microsoft JhengHei"/>',
        '<a:font script="Jpan" typeface="ＭＳ Ｐゴシック"/>': '<a:font script="Jpan" typeface="Yu Gothic"/>',
    }

    with zipfile.ZipFile(pptx_path, "r") as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}

    theme_xml = payload["ppt/theme/theme1.xml"].decode("utf-8")
    for old, new in replacements.items():
        theme_xml = theme_xml.replace(old, new)
    payload["ppt/theme/theme1.xml"] = theme_xml.encode("utf-8")

    temp_path = pptx_path.with_suffix(".tmp")
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    temp_path.replace(pptx_path)


def build_ppt() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    generate_pipeline_diagram(ASSETS_DIR / "research_pipeline.png")
    generate_model_diagram(ASSETS_DIR / "model_architecture.png")
    generate_progress_dashboard(ASSETS_DIR / "progress_dashboard.png")
    write_markdown(OUTPUT_MD)

    resource_path = os.pathsep.join([str(ROOT), str(DOCS_DIR), str(ROOT / "scripts")])
    command = [
        "pandoc",
        str(OUTPUT_MD),
        "-o",
        str(OUTPUT_PPTX),
        "--slide-level=1",
        f"--resource-path={resource_path}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    patch_theme(OUTPUT_PPTX)

    with zipfile.ZipFile(OUTPUT_PPTX, "r") as archive:
        slide_count = sum(1 for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
    print(f"Generated: {OUTPUT_PPTX}")
    print(f"Slides: {slide_count}")


if __name__ == "__main__":
    build_ppt()
