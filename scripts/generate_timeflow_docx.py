from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


def p(text: str) -> str:
    text = escape(text)
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def heading(text: str, level: int = 1) -> str:
    text = escape(text)
    style = f"Heading{level}"
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'
    )


def build_docx(output_path: Path) -> None:
    body = []
    body.append(heading("TimeFlow实验思路与超参数范围整理", 1))

    body.append(heading("第一节 采样超参数范围", 2))
    body.append(
        p(
            "本节根据 scripts/draw_hyperparameters_LHS.py 当前版本整理。该脚本采用 Latin Hypercube Sampling（LHS）"
            "生成群体超参数样本，但当前真正参与变化的参数只有部分维度，其余参数被固定。"
        )
    )
    body.append(p("1. 固定参数："))
    body.append(p("   zp = 1.9"))
    body.append(p("   alpha_z = 2.7"))
    body.append(p("   beta_z = 2.9"))
    body.append(p("   m_max = 100.0"))
    body.append(p("   m_min = 2.0"))
    body.append(p("   lambda_peak = 0.04"))
    body.append(p("   beta_q = 1.08"))
    body.append(p("2. 当前采用LHS采样的参数范围："))
    body.append(p("   alpha_m ∈ [-3.0, 7.0]"))
    body.append(p("   delta_m ∈ [0.0, 10.0]"))
    body.append(p("   mu_m ∈ [25.0, 100.0]"))
    body.append(p("   sigma_m ∈ [0.1, 5.0]"))
    body.append(p("3. 采样方式说明："))
    body.append(
        p(
            "脚本先在单位超立方体内做LHS采样，再把各维样本线性映射到目标物理范围。与普通独立均匀采样相比，"
            "LHS在每一维上的覆盖更均匀，更适合在有限样本数下探索参数空间。"
        )
    )

    body.append(heading("第二节 实验思路", 2))
    body.append(p("当前TimeFlow实验主线可以概括为以下步骤。"))
    body.append(
        p("1. 先采样群体超参数。每一组超参数定义一个双黑洞群体模型，决定红移分布、主质量分布、峰值成分以及质量比等总体性质。")
    )
    body.append(
        p("2. 由群体超参数继续采样具体CBC事件参数，例如并合时间、主次质量、红移、距离和天空位置等，从而得到该组超参数下的一批离散事件。")
    )
    body.append(
        p("3. 生成双探测器信号并与噪声叠加。首先按固定采样率和固定frame长度构造长时序数据，再把落入当前时间窗的CBC事件逐个注入到H1和L1探测器响应中，必要时对过强事件进行筛除。")
    )
    body.append(
        p("4. 将长时序frame切分为训练样本。每个切片保存为一个npy文件，内部包含双探测器输入数据以及对应的群体超参数标签。")
    )
    body.append(
        p("5. 构造模型输入表示。当前主要有两类输入：一类是原始时域双通道序列，另一类是从双探测器时域数据中提取的频域统计特征，例如PSD、CPSD和coherence。")
    )
    body.append(
        p("6. 使用条件归一化流学习后验分布。编码器先从输入x中提取条件特征，再由normalizing flow学习 p(theta | x)，其中theta表示待恢复的群体超参数。")
    )
    body.append(
        p("7. 在测试集上评估模型后验质量。常用指标包括NLL、cov68/cov95以及ci68/r、ci95/r，并结合一维边缘后验、二维联合后验和多维整体NLL来综合判断模型表现。")
    )
    body.append(
        p("整体上，这条实验线关注的问题是：在给定双探测器输入的情况下，模型能否稳定恢复出对应的群体超参数分布，并且在不同输入表示、不同参数维数和不同数据构造方式下保持合理的校准和收窄能力。")
    )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" '
        'w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
</w:styles>"""

    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/styles.xml", styles)


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "docs" / "TimeFlow_Experiment_and_Hyperparameter_Ranges.docx"
    build_docx(output_path)
    print(output_path)
