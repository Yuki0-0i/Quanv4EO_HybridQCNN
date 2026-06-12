"""
写 docx 头部信息:
- 学院/学系/专业 (跟课程相关)
- 学生姓名 (你的 GitHub 账号)
- 日期
- GitHub 链接: Yuki0-0i/Quanv4EO_HybridQCNN
- 链接内容: 代码 + 数据集 + 用法 README
"""
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOCX_PATH = "/hdd/Dengxuanyu/dxy1/26项目报告1.docx"


def set_text(p, text):
    """清空段落后填新内容."""
    for run in p.runs:
        run.text = ""
    if not p.runs:
        run = p.add_run(text)
    else:
        p.runs[0].text = text


def set_para_after_text(doc, anchor_text, new_text):
    """在含 anchor_text 段后插入新段."""
    for i, p in enumerate(doc.paragraphs):
        if anchor_text in p.text:
            new_p = p.insert_paragraph_before(new_text)
            return new_p
    return None


def fill_header(doc):
    print("--- 填表头信息 ---")
    for i, p in enumerate(doc.paragraphs):
        txt = p.text
        if "学    院：" in txt and not txt.replace("学    院：", "").strip():
            set_text(p, "学    院：网络空间安全学院")
        elif "学    系：" in txt and not txt.replace("学    系：", "").strip():
            set_text(p, "学    系：网络空间安全")
        elif "专    业：" in txt and not txt.replace("专    业：", "").strip():
            set_text(p, "专    业：网络空间安全")
        elif "学生姓名（学号）：" in txt and not txt.replace("学生姓名（学号）：", "").strip():
            set_text(p, "学生姓名（学号）：Yuki0-0i (3312793379@qq.com)")
        elif "年  月  日" in txt:
            set_text(p, "2026 年 06 月 09 日")


def replace_github_section(doc):
    print("--- 替换 GitHub 链接段 ---")
    # 找到 项目的GitHub链接地址 这一行, 替换后续 3 行
    for i, p in enumerate(doc.paragraphs):
        if "项目的GitHub链接地址" in p.text:
            # 重写当前段
            set_text(p, "项目的GitHub链接地址：https://github.com/Yuki0-0i/Quanv4EO_HybridQCNN")
            # 删除后续两行 (链接内容包括 + 1.代码... )
            for k in range(2):
                if i + 1 < len(doc.paragraphs):
                    p_next = doc.paragraphs[i + 1]
                    p_next._element.getparent().remove(p_next._element)
            # 插入新段
            p2 = p.insert_paragraph_before("链接内容包括：")
            p3 = p2.insert_paragraph_before("1. 代码  2. 数据集  3. 用法 README 文件")
            return True
    return False


def add_simple_intro(doc):
    """在 链接内容包括 后, 项目功能介绍 前, 插简单项目说明"""
    print("--- 加简单项目说明 ---")
    for i, p in enumerate(doc.paragraphs):
        if "1. 代码  2. 数据集" in p.text:
            new_lines = [
                "项目简介: 基于 Quanv4EO (TGRS 2025) 的混合量子-经典遥感图像分类模型, 现代化重构 + 4 实验消融 + 5-seed 严格评估",
                "量子后端: PennyLane 0.42.3 + lightning.qubit (CPU C++ 后端, 实测比 GPU 快 40-70×)",
                "CNN 后端: PyTorch 2.11+cu130 (RTX 5090), 3 conv + 2 FC 主架构",
                "核心结果: Exp3 融合方案 (QConv+RGB+ResNet-18) 5000 张达 0.807, 距经典 ResNet (0.827) 仅 -0.020",
                "项目结构: quanv4eo_modern/ (核心) + experiments/ (55 py) + reports/ (21 md) + source/ (原版参考)",
                "复现命令: conda create -n plenv_gpu python=3.10 && pip install -r requirements.txt && bash run_all.sh",
                "",
            ]
            for line in reversed(new_lines):
                p.insert_paragraph_before(line)
            return True
    return False


def main():
    doc = Document(DOCX_PATH)
    print(f"现有段落数: {len(doc.paragraphs)}")
    fill_header(doc)
    replace_github_section(doc)
    add_simple_intro(doc)
    doc.save(DOCX_PATH)
    print(f"保存: {DOCX_PATH}")
    print(f"现在段落数: {len(doc.paragraphs)}")


if __name__ == "__main__":
    main()
