from __future__ import annotations

import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


DOCX_PATH = Path("docs/TimeFlow_Experiment_and_Hyperparameter_Ranges.docx")
NS_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": NS_URI}

ET.register_namespace("w", NS_URI)


SECTION3_TEXTS = [
    "第三节 后验评估指标计算方式",
    "本节说明测试脚本中常用的三个指标：cov68、ci68 和 ci68/r。它们都按参数维度分别统计，并在整个测试集上取平均。",
    "1. cov68 的定义",
    "对测试集中的每个样本 i 和某个参数维度 j，先从模型给出的后验样本中计算第 16 百分位 q16(i,j) 与第 84 百分位 q84(i,j)。于是该参数的一维 68% 后验区间定义为：",
    "[q16(i,j), q84(i,j)]",
    "再判断真实标签 theta_true(i,j) 是否落在该区间内。若落在区间内记为 1，否则记为 0。于是 cov68 在参数维度 j 上的定义为：",
    "cov68_j = mean_i 1[q16(i,j) <= theta_true(i,j) <= q84(i,j)]",
    "因此，cov68 表示：模型给出的 68% 后验区间，在整个测试集中有多少比例真的包含了真值。理想情况下 cov68 应接近 0.68。",
    "2. ci68 的定义",
    "对每个样本 i、每个参数维度 j，68% 后验区间宽度定义为：",
    "ci68(i,j) = q84(i,j) - q16(i,j)",
    "然后在整个测试集上取平均，得到该参数维度上的平均区间宽度：",
    "ci68_j = mean_i [q84(i,j) - q16(i,j)]",
    "因此，ci68 反映的是：模型对某个参数给出的 68% 后验区间平均有多宽。",
    "3. ci68/r 的定义",
    "由于不同参数的物理量纲和取值范围不同，直接比较 ci68 不够公平，因此进一步用该参数的原始取值范围宽度 r_j 做归一化，其中：",
    "r_j = theta_max(j) - theta_min(j)",
    "于是：",
    "ci68/r_j = mean_i [(q84(i,j) - q16(i,j)) / r_j]",
    "因为 r_j 对于固定参数维度是常数，上式在数值上等价于：",
    "ci68/r_j = ci68_j / r_j",
    "因此，ci68/r 表示：模型给出的 68% 后验区间，平均占整个参数原始范围的多少比例。若该值较大，说明后验较宽；若该值较小，说明模型对该参数约束更强。",
]


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text for node in paragraph.findall(".//w:t", NS) if node.text).strip()


def make_paragraph(text: str) -> ET.Element:
    p = ET.Element(f"{{{NS_URI}}}p")
    r = ET.SubElement(p, f"{{{NS_URI}}}r")
    t = ET.SubElement(r, f"{{{NS_URI}}}t")
    t.text = text
    return p


def main() -> None:
    with zipfile.ZipFile(DOCX_PATH, "r") as zf:
        xml_bytes = zf.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("document body not found")

    paragraphs = body.findall("w:p", NS)
    start_idx = None
    end_idx = None
    for idx, paragraph in enumerate(paragraphs):
        text = paragraph_text(paragraph)
        if text in {"第三节 后验评估指标计算方式", "??? ??????????"}:
            start_idx = idx
        if start_idx is not None and text.endswith("说明模型对该参数约束更强。"):
            end_idx = idx
            break
        if start_idx is not None and text.endswith("??????????????????????????????????????"):
            end_idx = idx
            break

    if start_idx is None:
        sect_pr = body.find("w:sectPr", NS)
        if sect_pr is None:
            raise RuntimeError("sectPr not found")
        insert_idx = list(body).index(sect_pr)
    else:
        if end_idx is None:
            raise RuntimeError("Found section 3 start but not end")
        for idx in range(end_idx, start_idx - 1, -1):
            body.remove(paragraphs[idx])
        sect_pr = body.find("w:sectPr", NS)
        if sect_pr is None:
            raise RuntimeError("sectPr not found")
        insert_idx = list(body).index(sect_pr)

    for offset, text in enumerate(SECTION3_TEXTS):
        body.insert(insert_idx + offset, make_paragraph(text))

    new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        with zipfile.ZipFile(DOCX_PATH, "r") as zf:
            zf.extractall(tmp_dir)
        (tmp_dir / "word" / "document.xml").write_bytes(new_xml)
        tmp_docx = DOCX_PATH.with_suffix(".tmp.docx")
        with zipfile.ZipFile(tmp_docx, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in tmp_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(tmp_dir))
        shutil.move(tmp_docx, DOCX_PATH)

    print(f"Updated {DOCX_PATH}")


if __name__ == "__main__":
    main()
