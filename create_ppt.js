const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = '沈益民';
pres.title = '基于CNN和归一化流的随机引力波背景参数估计方法';

// === Color Palette: Modern Slate + Warm Coral ===
const C = {
  dark:     "0F172A",   // near black (dark sections)
  primary:  "1E3A5F",   // deep navy
  accent:   "F97068",   // warm coral
  teal:     "0D9488",   // teal
  lightBg:  "F8FAFC",   // off-white
  white:    "FFFFFF",
  text:     "1E293B",
  muted:    "94A3B8",
  card:     "FFFFFF",
  border:   "E2E8F0",
  tagGreen: "10B981",
};

// Shadow factory (fresh object each call)
const mkShadow = () => ({ type: "outer", blur: 10, offset: 3, angle: 135, color: "000000", opacity: 0.08 });

// ================================================================
// SLIDE 1: COVER
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.dark };

  // Top coral accent line
  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  // Large decorative circle (top right, very subtle)
  sl.addShape(pres.shapes.OVAL, {
    x: 7.0, y: -1.5, w: 5, h: 5,
    fill: { color: C.primary, transparency: 60 }
  });

  // Left accent bar
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.4, w: 0.07, h: 2.6, fill: { color: C.accent }
  });

  // Main title
  sl.addText("基于CNN和归一化流的", {
    x: 0.9, y: 1.5, w: 8.5, h: 0.8,
    fontSize: 30, fontFace: "Microsoft YaHei", bold: false, color: "CBD5E1", margin: 0
  });
  sl.addText("随机引力波背景参数估计方法", {
    x: 0.9, y: 2.2, w: 8.5, h: 1.0,
    fontSize: 38, fontFace: "Microsoft YaHei", bold: true, color: C.white, margin: 0
  });

  // Divider
  sl.addShape(pres.shapes.RECTANGLE, { x: 0.9, y: 3.35, w: 2.0, h: 0.04, fill: { color: C.accent } });

  // Author info
  sl.addText("沈益民", {
    x: 0.9, y: 3.55, w: 4, h: 0.5,
    fontSize: 18, fontFace: "Microsoft YaHei", bold: true, color: C.white, margin: 0
  });
  sl.addText("导师：孙翠敏", {
    x: 0.9, y: 4.05, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Microsoft YaHei", color: "7B92B0", margin: 0
  });
  sl.addText("2026 / 04 / 16", {
    x: 0.9, y: 4.45, w: 4, h: 0.4,
    fontSize: 14, fontFace: "Arial", color: "7B92B0", margin: 0
  });

  // Bottom bar
  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.52, w: 10, h: 0.105, fill: { color: C.accent } });
}

// ================================================================
// SLIDE 2: CONTENTS
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  // Top accent line
  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  // Header area
  sl.addText("目录", {
    x: 0.6, y: 0.35, w: 3, h: 0.7,
    fontSize: 30, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });
  sl.addText("CONTENTS", {
    x: 0.6, y: 0.95, w: 3, h: 0.35,
    fontSize: 11, fontFace: "Arial", color: C.muted, charSpacing: 4, margin: 0
  });

  const sections = [
    { num: "01", title: "研究背景", sub: "随机引力波背景与参数估计方法" },
    { num: "02", title: "已完成任务", sub: "数据生成、模型搭建与训练进展" },
    { num: "03", title: "未完成任务", sub: "数据集生成与模型训练计划" },
  ];

  sections.forEach((s, i) => {
    let y = 1.6 + i * 1.2;
    let x = 0.6;

    // Number
    sl.addText(s.num, {
      x: x, y: y, w: 0.8, h: 0.9,
      fontSize: 36, fontFace: "Arial", bold: true, color: C.accent, margin: 0
    });

    // Vertical divider
    sl.addShape(pres.shapes.RECTANGLE, {
      x: x + 1.0, y: y + 0.1, w: 0.03, h: 0.7, fill: { color: C.border }
    });

    // Title & sub
    sl.addText(s.title, {
      x: x + 1.25, y: y + 0.05, w: 7, h: 0.5,
      fontSize: 20, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
    });
    sl.addText(s.sub, {
      x: x + 1.25, y: y + 0.5, w: 7, h: 0.4,
      fontSize: 12, fontFace: "Microsoft YaHei", color: C.muted, margin: 0
    });
  });
}

// ================================================================
// SLIDE 3: SECTION — 研究背景
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.dark };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  // Big number
  sl.addText("01", {
    x: 0.6, y: 1.5, w: 3, h: 1.2,
    fontSize: 80, fontFace: "Arial", bold: true, color: C.accent, margin: 0
  });

  sl.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.85, w: 2.5, h: 0.05, fill: { color: C.accent } });

  sl.addText("研究背景", {
    x: 0.6, y: 3.1, w: 8, h: 1.0,
    fontSize: 44, fontFace: "Microsoft YaHei", bold: true, color: C.white, margin: 0
  });
  sl.addText("STUDY BACKGROUND", {
    x: 0.6, y: 4.0, w: 8, h: 0.5,
    fontSize: 13, fontFace: "Arial", color: "4A6580", charSpacing: 4, margin: 0
  });
}

// ================================================================
// SLIDE 4: 研究背景内容
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("研究背景", {
    x: 0.6, y: 0.3, w: 5, h: 0.65,
    fontSize: 24, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  // --- Left Card: SGWB ---
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.1, w: 4.3, h: 4.2,
    fill: { color: C.card }, shadow: mkShadow()
  });
  // Top colored band
  sl.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 4.3, h: 0.08, fill: { color: C.teal } });

  // Icon circle
  sl.addShape(pres.shapes.OVAL, {
    x: 0.8, y: 1.4, w: 0.6, h: 0.6, fill: { color: "D1FAF5" }
  });
  sl.addText("🌌", {
    x: 0.8, y: 1.4, w: 0.6, h: 0.6,
    fontSize: 18, align: "center", valign: "middle"
  });

  sl.addText("随机引力波背景（SGWB）", {
    x: 1.55, y: 1.45, w: 3.0, h: 0.5,
    fontSize: 14, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0, valign: "middle"
  });

  sl.addText([
    { text: "宇宙中大量不可分辨的双黑洞系统产生的叠加引力波信号", options: { bullet: true, breakLine: true } },
    { text: "LIGO / Virgo 等探测器可对其进行观测", options: { bullet: true, breakLine: true } },
    { text: "是理解星系演化与双黑洞 population 的重要探针", options: { bullet: true } }
  ], {
    x: 0.75, y: 2.2, w: 3.8, h: 2.8,
    fontSize: 12.5, fontFace: "Microsoft YaHei", color: C.text, valign: "top", paraSpaceAfter: 12
  });

  // --- Right Card: 方法 ---
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.1, w: 4.3, h: 4.2,
    fill: { color: C.card }, shadow: mkShadow()
  });
  sl.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.3, h: 0.08, fill: { color: C.accent } });

  sl.addShape(pres.shapes.OVAL, {
    x: 5.5, y: 1.4, w: 0.6, h: 0.6, fill: { color: "FEE2E2" }
  });
  sl.addText("📊", {
    x: 5.5, y: 1.4, w: 0.6, h: 0.6,
    fontSize: 18, align: "center", valign: "middle"
  });

  sl.addText("参数估计方法", {
    x: 6.25, y: 1.45, w: 3.0, h: 0.5,
    fontSize: 14, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0, valign: "middle"
  });

  sl.addText([
    { text: "传统方法：贝叶斯推断，计算代价高", options: { bullet: true, breakLine: true } },
    { text: "CNN：提取引力波数据深层特征", options: { bullet: true, breakLine: true } },
    { text: "归一化流：学习超参数后验分布", options: { bullet: true, breakLine: true } },
    { text: "结合两者优势，实现高效参数估计", options: { bullet: true } }
  ], {
    x: 5.45, y: 2.2, w: 3.8, h: 2.8,
    fontSize: 12.5, fontFace: "Microsoft YaHei", color: C.text, valign: "top", paraSpaceAfter: 12
  });
}

// ================================================================
// SLIDE 5: SECTION — 已完成任务
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.dark };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("02", {
    x: 0.6, y: 1.5, w: 3, h: 1.2,
    fontSize: 80, fontFace: "Arial", bold: true, color: C.accent, margin: 0
  });

  sl.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.85, w: 2.5, h: 0.05, fill: { color: C.accent } });

  sl.addText("已完成任务", {
    x: 0.6, y: 3.1, w: 8, h: 1.0,
    fontSize: 44, fontFace: "Microsoft YaHei", bold: true, color: C.white, margin: 0
  });
  sl.addText("COMPLETED TASKS", {
    x: 0.6, y: 4.0, w: 8, h: 0.5,
    fontSize: 13, fontFace: "Arial", color: "4A6580", charSpacing: 4, margin: 0
  });
}

// ================================================================
// SLIDE 6: 数据生成
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("已完成任务 — 数据生成和处理", {
    x: 0.6, y: 0.3, w: 9, h: 0.65,
    fontSize: 24, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  // 4 step cards in a horizontal flow
  const steps = [
    { num: "1", title: "超参数生成", body: "基于物理规律生成\n10,000组超参数\n（10维标签）", color: C.teal },
    { num: "2", title: "双黑洞种群", body: "gwfast生成双黑洞\n质量群体\n耗时12小时", color: C.teal },
    { num: "3", title: "时间序列", body: "gwpy拟合波形\nLIGO双探测器\n2048Hz采样", color: C.accent },
    { num: "4", title: "数据分割", body: "每帧2048s\n分割为8个segments\n256×2048特征", color: C.accent },
  ];

  const cardW = 2.05, cardH = 2.1, startX = 0.5, cardY = 1.15, gap = 0.28;

  steps.forEach((s, i) => {
    let x = startX + i * (cardW + gap);

    // Card
    sl.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: cardW, h: cardH,
      fill: { color: C.card }, shadow: mkShadow()
    });
    // Top accent line
    sl.addShape(pres.shapes.RECTANGLE, { x, y: cardY, w: cardW, h: 0.07, fill: { color: s.color } });

    // Number badge
    sl.addShape(pres.shapes.OVAL, {
      x: x + cardW / 2 - 0.25, y: cardY + 0.2, w: 0.5, h: 0.5,
      fill: { color: s.color }
    });
    sl.addText(s.num, {
      x: x + cardW / 2 - 0.25, y: cardY + 0.2, w: 0.5, h: 0.5,
      fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle"
    });

    sl.addText(s.title, {
      x: x + 0.1, y: cardY + 0.82, w: cardW - 0.2, h: 0.4,
      fontSize: 12, fontFace: "Microsoft YaHei", bold: true, color: C.dark, align: "center", margin: 0
    });
    sl.addText(s.body, {
      x: x + 0.1, y: cardY + 1.25, w: cardW - 0.2, h: 0.8,
      fontSize: 10.5, fontFace: "Microsoft YaHei", color: C.text, align: "center", valign: "top", lineSpacingMultiple: 1.3
    });

    // Arrow
    if (i < steps.length - 1) {
      sl.addText("→", {
        x: x + cardW + 0.02, y: cardY + cardH / 2 - 0.2, w: 0.24, h: 0.4,
        fontSize: 16, color: C.muted, align: "center", valign: "middle"
      });
    }
  });

  // Bottom stats bar
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.5, w: 9, h: 1.85,
    fill: { color: C.dark }, shadow: mkShadow()
  });

  const stats = [
    { val: "23×8×10,000", label: "样本总数" },
    { val: "256×2048", label: "特征维度 / 样本" },
    { val: "2048 s", label: "每帧时长" },
    { val: "8", label: "segments / 帧" },
  ];

  stats.forEach((st, i) => {
    let sx = 0.7 + i * 2.25;
    sl.addText(st.val, {
      x: sx, y: 3.65, w: 2.0, h: 0.75,
      fontSize: 22, fontFace: "Arial", bold: true, color: C.white, margin: 0
    });
    sl.addText(st.label, {
      x: sx, y: 4.35, w: 2.0, h: 0.35,
      fontSize: 11, color: "64748B", margin: 0
    });
    if (i < stats.length - 1) {
      sl.addShape(pres.shapes.RECTANGLE, {
        x: sx + 1.8, y: 3.75, w: 0.02, h: 0.7, fill: { color: "2D3F58" }
      });
    }
  });
}

// ================================================================
// SLIDE 7: 模型搭建
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("已完成任务 — 模型搭建", {
    x: 0.6, y: 0.3, w: 9, h: 0.65,
    fontSize: 24, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  const models = [
    {
      layer: "第一层",
      layerColor: C.teal,
      bgColor: "D1FAF5",
      title: "CNNModel.py",
      sub: "卷积神经网络",
      points: [
        "自动提取引力波信号深层特征",
        "多尺度卷积核捕获时频域信息",
        "输出特征向量供后续网络使用",
      ]
    },
    {
      layer: "第二层",
      layerColor: C.accent,
      bgColor: "FEE2E2",
      title: "GWFlowModel.py",
      sub: "归一化流模型",
      points: [
        "学习10维超参数后验分布",
        "可逆变换实现精确概率推断",
        "输出超参数估计结果及不确定性",
      ]
    }
  ];

  models.forEach((m, i) => {
    let x = 0.5 + i * 4.7;

    // Card
    sl.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.1, w: 4.5, h: 3.6,
      fill: { color: C.card }, shadow: mkShadow()
    });
    sl.addShape(pres.shapes.RECTANGLE, { x, y: 1.1, w: 4.5, h: 0.08, fill: { color: m.layerColor } });

    // Layer badge
    sl.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.25, y: 1.35, w: 0.9, h: 0.35,
      fill: { color: m.layerColor }
    });
    sl.addText(m.layer, {
      x: x + 0.25, y: 1.35, w: 0.9, h: 0.35,
      fontSize: 10, bold: true, color: C.white, align: "center", valign: "middle"
    });

    sl.addText(m.title, {
      x: x + 0.25, y: 1.88, w: 4.0, h: 0.5,
      fontSize: 17, fontFace: "Consolas", bold: true, color: C.dark, margin: 0
    });
    sl.addText(m.sub, {
      x: x + 0.25, y: 2.32, w: 4.0, h: 0.35,
      fontSize: 12, color: C.muted, margin: 0
    });

    sl.addText(m.points.map((p, j) => ({
      text: p,
      options: { bullet: true, breakLine: j < m.points.length - 1 }
    })), {
      x: x + 0.25, y: 2.8, w: 4.0, h: 1.8,
      fontSize: 12.5, fontFace: "Microsoft YaHei", color: C.text, valign: "top", paraSpaceAfter: 10
    });
  });

  // Architecture placeholder
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.55,
    fill: { color: C.border }
  });
  sl.addText("[ 模型结构图 — CNN层 → 特征提取 → 归一化流层 → 后验分布 ]", {
    x: 0.5, y: 4.85, w: 9, h: 0.55,
    fontSize: 12, fontFace: "Consolas", color: C.muted, align: "center", valign: "middle", italic: true
  });
}

// ================================================================
// ================================================================
// SLIDE 8: 模型结构图（PPT内直接绘制，SCI纯色风格）
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.white };

  const sh = (o) => ({ type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: o || 0.10 });

  const T = (sl, text, x, y, w, h, size, color, bold) => {
    sl.addText(text, {
      x, y, w, h,
      fontSize: size, fontFace: "Arial", color, bold: !!bold,
      align: "center", valign: "middle", margin: 0
    });
  };

  // Fully-filled rounded rect with white text
  const B = (sl, x, y, w, h, bg, text, bold) => {
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w, h, rectRadius: 0.07,
      fill: { color: bg },
      line: { color: bg, width: 0 },
      shadow: sh()
    });
    sl.addText(text, {
      x, y, w, h,
      fontSize: 10.5, fontFace: "Arial", color: "FFFFFF", bold: !!bold,
      align: "center", valign: "middle", margin: 0
    });
  };

  // Title
  T(sl, "Model Architecture", 0.4, 0.22, 9.2, 0.44, 19, C.dark, true);
  T(sl, "SimpleCNN1D  →  Embedding  →  GWFlowModel  →  z₀ ~ 𝒩(0, I)", 0.4, 0.64, 9.2, 0.26, 9, C.muted);

  // ================================================================
  // LEFT COLUMN: CNN  (x = 0.4 – 2.55)
  // ================================================================
  const lx = 0.4, lw = 2.05;

  B(sl, lx, 1.08, lw, 0.42, "6AA0D0", "Input", true);
  B(sl, lx, 1.60, lw, 0.42, "6AA0D0", "Conv1d  1→16", true);
  B(sl, lx, 2.12, lw, 0.42, "6AA0D0", "BN · ReLU · Pool", false);
  B(sl, lx, 2.64, lw, 0.42, "6AA0D0", "Conv1d  16→32", true);
  B(sl, lx, 3.16, lw, 0.42, "6AA0D0", "BN · ReLU · Pool", false);
  B(sl, lx, 3.68, lw, 0.42, "6AA0D0", "Conv1d  32→64", true);
  B(sl, lx, 4.20, lw, 0.42, "6AA0D0", "MaxPool  64×512", true);

  // Tag
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: lx, y: 4.75, w: lw, h: 0.28, rectRadius: 0.14,
    fill: { color: "D6E8FF" }, line: { color: "6AA0D0", width: 0.8 }
  });
  T(sl, "SimpleCNN1D", lx, 4.75, lw, 0.28, 9, "2A5080", true);

  // ================================================================
  // MIDDLE COLUMN: Embedding  (x = 2.65 – 4.85)
  // ================================================================
  const mx = 2.65, mw = 2.1;

  B(sl, mx, 2.9, mw, 0.50, "7EC8A0", "Flatten · FC(256)→10", true);
  B(sl, mx, 3.50, mw, 0.50, "7EC8A0", "z  (context ∈ ℝ⁵¹²)", true);

  // Tag
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: mx, y: 4.75, w: mw, h: 0.28, rectRadius: 0.14,
    fill: { color: "D0F0E4" }, line: { color: "7EC8A0", width: 0.8 }
  });
  T(sl, "Embedding", mx, 4.75, mw, 0.28, 9, "2A6A40", true);

  // ================================================================
  // RIGHT COLUMN: Flow  (x = 5.05 – 9.55)
  // ================================================================
  const rx = 5.05, rw = 4.3;

  // Tag header
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rx, y: 1.08, w: rw, h: 0.28, rectRadius: 0.14,
    fill: { color: "FFE0E8" }, line: { color: "E090A0", width: 0.8 }
  });
  T(sl, "GWFlowModel  ·  Normalizing Flow × 8", rx, 1.08, rw, 0.28, 9, "8B3A50", true);

  B(sl, rx, 1.50, rw, 0.44, "E890A0", "Flow Layer 1  ·  Permutation + MAF", true);
  B(sl, rx, 2.04, rw, 0.38, "E890A0", "Flow Layer 2 – 8  (× 7)", true);
  B(sl, rx, 2.52, rw, 0.44, "7AACD0", "z₀ ~ 𝒩(0, I)  ·  Standard Normal Prior", true);
  B(sl, rx, 3.06, rw, 0.44, "90C8E0", "Loss:  −log p(θ | x)  (NLL)", true);

  // Tag
  sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rx, y: 4.75, w: rw, h: 0.28, rectRadius: 0.14,
    fill: { color: "FFE0E8" }, line: { color: "E090A0", width: 0.8 }
  });
  T(sl, "GWFlowModel", rx, 4.75, rw, 0.28, 9, "8B3A50", true);

  // ================================================================
  // CONNECTIONS
  // ================================================================
  const ln = { color: "B0B0C0", width: 1.2 };
  const aw = { color: "B0B0C0", width: 1.2, arrowEnd: { type: "triangle" } };

  // Internal CNN arrows
  [1.50, 2.02, 2.54, 3.06, 3.58].forEach(y => {
    sl.addShape(pres.shapes.LINE, { x: lx + lw / 2, y, w: 0, h: 0.1, line: ln, arrowEnd: { type: "triangle" } });
  });

  // Pool → FC (L-shape: right, then up)
  sl.addShape(pres.shapes.LINE, { x: lx + lw, y: 4.41, w: mx - (lx + lw), h: 0, line: ln });
  sl.addShape(pres.shapes.LINE, { x: mx, y: 3.15, w: 0, h: 4.41 - 3.15, line: ln });
  sl.addShape(pres.shapes.LINE, { x: mx, y: 3.15, w: mw / 2 - 0.05, h: 0, line: ln, arrowEnd: { type: "triangle" } });

  // FC → Context
  sl.addShape(pres.shapes.LINE, { x: mx + mw / 2, y: 3.40, w: 0, h: 0.1, line: aw });

  // Context → Flow (right, then up)
  sl.addShape(pres.shapes.LINE, { x: mx + mw, y: 3.75, w: rx - (mx + mw), h: 0, line: ln });
  sl.addShape(pres.shapes.LINE, { x: rx, y: 1.36, w: 0, h: 3.75 - 1.36, line: ln });
  sl.addShape(pres.shapes.LINE, { x: rx, y: 1.36, w: rw / 2 - 0.05, h: 0, line: ln, arrowEnd: { type: "triangle" } });

  // Internal Flow arrows
  [1.94, 2.42, 2.96].forEach(y => {
    sl.addShape(pres.shapes.LINE, { x: rx + rw / 2, y, w: 0, h: 0.1, line: aw });
  });

  // θ dashed divider
  sl.addShape(pres.shapes.LINE, {
    x: rx, y: 1.36, w: 0, h: 2.1,
    line: { color: "E0D0D8", width: 0.8, dashType: "dash" }
  });
  T(sl, "θ ∈ ℝ¹⁰", rx + 0.1, 1.36, 0.8, 0.22, 7.5, "BBBBBB");

  // Legend
  const lg = [
    { bg: "6AA0D0", label: "Conv1d" },
    { bg: "7EC8A0", label: "Flatten+FC" },
    { bg: "7EC8A0", label: "Context z" },
    { bg: "E890A0", label: "Flow Layer" },
    { bg: "7AACD0", label: "Prior z₀" },
    { bg: "90C8E0", label: "Loss" },
  ];
  lg.forEach((l, i) => {
    const lx2 = 0.4 + i * 1.55;
    sl.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: lx2, y: 5.1, w: 1.45, h: 0.28, rectRadius: 0.07,
      fill: { color: l.bg }, line: { color: l.bg }
    });
    T(sl, l.label, lx2, 5.1, 1.45, 0.28, 8, "FFFFFF");
  });
}

// ================================================================
// SLIDE 9: 小结
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("已完成任务 — 小结", {
    x: 0.6, y: 0.3, w: 9, h: 0.65,
    fontSize: 24, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  const items = [
    { ok: true, text: "整个流程已经跑通" },
    { ok: true, text: "相关物理概念已经理清" },
    { ok: true, text: "CNN + 归一化流模型已完成搭建并初步训练" },
    { ok: false, text: "需要大量数据来获得模型最终训练结果" },
  ];

  items.forEach((item, i) => {
    let y = 1.15 + i * 0.92;

    sl.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9, h: 0.77,
      fill: { color: C.card }, shadow: mkShadow()
    });
    sl.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.07, h: 0.77, fill: { color: item.ok ? C.tagGreen : C.accent }
    });

    // Check or circle icon
    sl.addShape(pres.shapes.OVAL, {
      x: 0.75, y: y + 0.17, w: 0.43, h: 0.43,
      fill: { color: item.ok ? "D1FAE5" : "FEE2E2" }
    });
    sl.addText(item.ok ? "✓" : "○", {
      x: 0.75, y: y + 0.17, w: 0.43, h: 0.43,
      fontSize: 14, color: item.ok ? C.tagGreen : C.accent, align: "center", valign: "middle"
    });

    sl.addText(item.text, {
      x: 1.35, y, w: 7.8, h: 0.77,
      fontSize: 15, fontFace: "Microsoft YaHei", color: C.text, valign: "middle", margin: 0
    });
  });

  sl.addText("下一步：利用更大规模数据集进行充分训练", {
    x: 0.5, y: 5.0, w: 9, h: 0.4,
    fontSize: 12.5, fontFace: "Microsoft YaHei", color: C.muted, align: "center"
  });
}

// ================================================================
// SLIDE 9: SECTION — 未完成任务
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.dark };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("03", {
    x: 0.6, y: 1.5, w: 3, h: 1.2,
    fontSize: 80, fontFace: "Arial", bold: true, color: C.accent, margin: 0
  });

  sl.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.85, w: 2.5, h: 0.05, fill: { color: C.accent } });

  sl.addText("未完成任务", {
    x: 0.6, y: 3.1, w: 8, h: 1.0,
    fontSize: 44, fontFace: "Microsoft YaHei", bold: true, color: C.white, margin: 0
  });
  sl.addText("UPCOMING TASKS", {
    x: 0.6, y: 4.0, w: 8, h: 0.5,
    fontSize: 13, fontFace: "Arial", color: "4A6580", charSpacing: 4, margin: 0
  });
}

// ================================================================
// SLIDE 10: 未完成任务内容
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.lightBg };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  sl.addText("未完成任务", {
    x: 0.6, y: 0.3, w: 9, h: 0.65,
    fontSize: 24, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  // --- Task 1: 数据集生成 ---
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.1, w: 9, h: 1.95,
    fill: { color: C.card }, shadow: mkShadow()
  });
  sl.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 9, h: 0.08, fill: { color: C.teal } });

  sl.addText("数据集生成", {
    x: 0.75, y: 1.3, w: 3, h: 0.45,
    fontSize: 16, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  // Progress bar
  sl.addText("当前进度：2000 / 10000 组超参数已完成", {
    x: 0.75, y: 1.82, w: 5, h: 0.35,
    fontSize: 12, color: C.muted, margin: 0
  });
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.75, y: 2.2, w: 5.5, h: 0.18, fill: { color: "E2E8F0" }
  });
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.75, y: 2.2, w: 1.1, h: 0.18, fill: { color: C.teal }
  });
  sl.addText("20%", {
    x: 6.4, y: 2.1, w: 0.6, h: 0.35,
    fontSize: 11, bold: true, color: C.teal, margin: 0
  });

  // ETA card
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 6.6, y: 1.35, w: 2.6, h: 1.5,
    fill: { color: "F0FDF4" }
  });
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 6.6, y: 1.35, w: 2.6, h: 0.06, fill: { color: C.tagGreen }
  });
  sl.addText("预计完成", {
    x: 6.75, y: 1.5, w: 2.3, h: 0.35,
    fontSize: 11, color: C.muted, margin: 0
  });
  sl.addText("~30天", {
    x: 6.6, y: 1.85, w: 2.6, h: 0.7,
    fontSize: 32, fontFace: "Arial", bold: true, color: C.tagGreen, align: "center"
  });
  sl.addText("剩余 8000 组", {
    x: 6.6, y: 2.5, w: 2.6, h: 0.3,
    fontSize: 10, color: C.muted, align: "center"
  });

  // --- Task 2: 模型训练 ---
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.25, w: 9, h: 2.1,
    fill: { color: C.card }, shadow: mkShadow()
  });
  sl.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.25, w: 9, h: 0.08, fill: { color: C.accent } });

  sl.addText("模型训练与调优", {
    x: 0.75, y: 3.45, w: 5, h: 0.45,
    fontSize: 16, fontFace: "Microsoft YaHei", bold: true, color: C.dark, margin: 0
  });

  sl.addText([
    { text: "数据集生成完毕后进行完整模型训练", options: { bullet: true, breakLine: true } },
    { text: "根据训练结果评估是否需要调整模型结构或超参数", options: { bullet: true, breakLine: true } },
    { text: "目标：获得可靠的随机引力波背景参数估计结果", options: { bullet: true } }
  ], {
    x: 0.75, y: 4.0, w: 8.5, h: 1.25,
    fontSize: 13, fontFace: "Microsoft YaHei", color: C.text, valign: "top", paraSpaceAfter: 10
  });
}

// ================================================================
// SLIDE 11: THANK YOU
// ================================================================
{
  let sl = pres.addSlide();
  sl.background = { color: C.dark };

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  // Decorative circle
  sl.addShape(pres.shapes.OVAL, {
    x: -1.5, y: 3.5, w: 4, h: 4,
    fill: { color: C.primary, transparency: 70 }
  });

  sl.addText("谢谢聆听", {
    x: 0.5, y: 1.3, w: 9, h: 1.1,
    fontSize: 52, fontFace: "Microsoft YaHei", bold: true, color: C.white, align: "center"
  });
  sl.addText("THANK YOU", {
    x: 0.5, y: 2.35, w: 9, h: 0.5,
    fontSize: 13, fontFace: "Arial", color: "4A6580", align: "center", charSpacing: 6
  });

  sl.addShape(pres.shapes.RECTANGLE, { x: 4.2, y: 2.95, w: 1.6, h: 0.05, fill: { color: C.accent } });

  // Summary card
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 1.5, y: 3.25, w: 7, h: 2.0,
    fill: { color: C.primary, transparency: 30 }
  });
  sl.addText("研究进展总结", {
    x: 1.75, y: 3.4, w: 6.5, h: 0.4,
    fontSize: 13, fontFace: "Microsoft YaHei", bold: true, color: C.accent, margin: 0
  });
  sl.addText([
    { text: "已完成：数据生成 pipeline + CNN + 归一化流模型搭建，流程已跑通", options: { breakLine: true } },
    { text: "待完成：大规模数据生成（预计 30 天）+ 充分模型训练", options: {} }
  ], {
    x: 1.75, y: 3.85, w: 6.5, h: 1.2,
    fontSize: 12.5, fontFace: "Microsoft YaHei", color: "B0C4DC", valign: "top", lineSpacingMultiple: 1.7
  });

  sl.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.52, w: 10, h: 0.105, fill: { color: C.accent } });
}

pres.writeFile({ fileName: "D:/pyProjects/LIGO_SGWB/组会汇报_沈益民_v4.pptx" })
  .then(() => console.log("PPT created successfully!"))
  .catch(err => console.error(err));
