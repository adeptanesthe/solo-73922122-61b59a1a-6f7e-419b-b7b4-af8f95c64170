# -*- coding: utf-8 -*-
"""简称核查算法测试：校验规则、TF-IDF 约束、一对一匹配、待核对与嵌套计数。"""
from __future__ import annotations

import pytest

from app.abbr_checker import (
    AbbrValidationError,
    check_abbreviations,
    glyph_similarity,
    result_to_dict,
    validate_inputs,
)

# 一篇 300+ 字、含 4 对机构名称的标准稿件
MANUSCRIPT = (
    "近日，国家发展和改革委员会召开年度工作会议，部署下一阶段重点任务。"
    "国家发展改革委在会上强调，要继续推进重大基础设施建设，并加强区域协调发展。"
    "与此同时，国家卫生健康委员会发布了基层医疗服务能力提升方案。"
    "国家卫健委要求各地在年内完成社区医院的标准化改造，重点补齐儿科与老年医学短板。"
    "记者还从科学技术部获悉，新一轮重点研发计划已经启动。"
    "科技部表示，本年度计划将向人工智能、量子信息等前沿方向倾斜。"
    "工业和信息化部随后公布了制造业数字化转型的配套政策。"
    "工信部提出，三年内要培育一批具备全球竞争力的智能制造示范工厂。"
    "以上部门将在国务院的统一协调下建立月度会商机制，确保各项政策落地见效。"
    "业内专家指出，国家发展改革委与工信部在产业项目上的分工需要进一步明确，"
    "避免地方在申报时把两个部门的支持方向混为一谈。"
)
FULL = [
    "国家发展和改革委员会",
    "国家卫生健康委员会",
    "科学技术部",
    "工业和信息化部",
]
ABBR = ["国家发展改革委", "国家卫健委", "科技部", "工信部"]


@pytest.fixture()
def result():
    return check_abbreviations(MANUSCRIPT, FULL, ABBR)


# ---------------- 输入校验 ----------------


def test_manuscript_length_bounds():
    short = "短稿" * 10  # 20 字
    with pytest.raises(AbbrValidationError, match="300~1200"):
        validate_inputs(short, FULL[:2], ABBR[:2])
    long_text = "很" * 1201
    with pytest.raises(AbbrValidationError, match="300~1200"):
        validate_inputs(long_text, FULL[:2], ABBR[:2])
    # 边界值 300 与 1200 可通过长度校验（名称存在性由其它用例覆盖）
    with pytest.raises(AbbrValidationError) as exc:
        validate_inputs("字" * 300, FULL[:2], ABBR[:2])
    assert "未在稿件中出现" in str(exc.value)


def test_name_count_bounds():
    with pytest.raises(AbbrValidationError, match="2~5 个"):
        validate_inputs(MANUSCRIPT, FULL[:1], ABBR[:1])
    with pytest.raises(AbbrValidationError, match="2~5 个"):
        validate_inputs(MANUSCRIPT, FULL + ["x"] * 2, ABBR + ["y"] * 2)


def test_name_counts_must_match():
    with pytest.raises(AbbrValidationError, match="数量必须相同"):
        validate_inputs(MANUSCRIPT, FULL[:3], ABBR[:2])


def test_every_name_must_appear():
    with pytest.raises(AbbrValidationError, match="未在稿件中出现"):
        validate_inputs(MANUSCRIPT, FULL, ABBR[:-1] + ["不存在的简称"])
    with pytest.raises(AbbrValidationError, match="未在稿件中出现"):
        validate_inputs(MANUSCRIPT, FULL[:-1] + ["不存在的全称"], ABBR)


def test_duplicate_and_overlap_names_rejected():
    with pytest.raises(AbbrValidationError, match="重复"):
        validate_inputs(MANUSCRIPT, [FULL[0], FULL[0]], [ABBR[0], ABBR[1]])
    with pytest.raises(AbbrValidationError, match="不能相同"):
        validate_inputs(MANUSCRIPT, [FULL[0], "科技部"], [ABBR[0], "科技部"])


def test_empty_names_rejected():
    with pytest.raises(AbbrValidationError, match="不能为空"):
        validate_inputs(MANUSCRIPT, ["  ", FULL[1]], [ABBR[0], ABBR[1]])


# ---------------- TF-IDF 约束 ----------------


def test_tfidf_fitted_only_on_manuscript_contexts(result):
    # 特征全部来自本稿语境文档，且严格为字符 2~3 元（无单字特征）
    assert result.feature_count > 100
    # 所有名称（全称+简称）都各有一份语境文档
    assert set(result.context_docs) == set(FULL + ABBR)


def test_context_docs_mask_name_strings(result):
    # 语境窗口里的名称串被统一占位符替换，避免名称字形直接灌高语境相似度
    for name, doc in result.context_docs.items():
        assert name not in doc


# ---------------- 匹配结论 ----------------


def test_one_to_one_mapping_is_correct(result):
    mapping = {p.abbreviation: p.full_name for p in result.pairs}
    assert mapping == dict(zip(ABBR, FULL))
    fulls_used = [p.full_name for p in result.pairs]
    assert len(set(fulls_used)) == len(fulls_used)  # 一对一：全称不重复
    assert all(p.status == "confirmed" for p in result.pairs)


def test_each_pair_carries_judgement_basis(result):
    for p in result.pairs:
        assert 0.0 <= p.context_similarity <= 1.0
        assert 0.0 <= p.glyph_similarity <= 1.0
        assert p.runner_up_full is not None and p.runner_up_full != p.full_name
        assert p.runner_up_score is not None
        # 得分差为四舍五入后的值，允许一位舍入误差
        assert abs(p.score_gap - round(p.score - p.runner_up_score, 4)) < 2e-4
        assert p.abbr_context_chars > 0 and p.full_context_chars > 0


def test_glyph_similarity_basic():
    assert glyph_similarity("清华", "清华大学") > 0.7
    assert glyph_similarity("科技部", "科学技术部") > glyph_similarity("科技部", "教育部直属")
    assert glyph_similarity("abc", "xyz") == pytest.approx(0.0, abs=1e-9)


def test_ambiguous_abbreviation_marked_review():
    # “华兴”同时是华兴银行/华兴证券的高频共用简称，替代对应接近 → 待核对
    text = (
        "华兴银行与华兴证券于同一天发布了年度报告。"
        "华兴银行（业内亦称兴行）的净利润同比增长百分之八，资产质量保持稳定。"
        "华兴证券则因为自营投资亏损，营收出现明显下滑。"
        "分析师认为，华兴的表现分化反映了银行业与证券业的周期差异。"
        "此外，两家机构都宣布将在上海设立新的办公区域。"
        "监管部门要求华兴加强关联交易披露，投资者可在官网查阅完整文件。"
        "与此同时，华晟资产管理公司也披露了季报，华晟资产规模稳步增长。"
        "华晟表示将继续加大固定收益类产品的布局，与华兴形成差异化竞争。"
        "本稿由财经新闻组综合整理，数据均来自各机构公开披露的定期报告。"
        "记者将持续跟踪后续进展，并在获得更多信息后及时更新报道内容。"
    ) * 2
    res = check_abbreviations(
        text,
        ["华兴银行", "华兴证券", "华晟资产管理公司"],
        ["兴行", "华兴", "华晟"],
    )
    by_abbr = {p.abbreviation: p for p in res.pairs}
    huaxing = by_abbr["华兴"]
    assert huaxing.status == "review"
    assert huaxing.reasons  # 至少给出一条明确依据
    # 仍给出一对一最优假设，但不强行确认
    assert huaxing.full_name in ("华兴银行", "华兴证券")
    # 依据中包含次优对应名称
    assert any(huaxing.runner_up_full in r for r in huaxing.reasons) or \
           any("低于确认阈值" in r or "差距" in r for r in huaxing.reasons)


def test_insufficient_context_marked_review():
    # 大段与名称无关的填充语之后，名称挤在稿件最末尾的短句里，
    # 单侧有效语境字符数低于阈值 → 即使相似度高也必须标记待核对
    pad = "背景介绍文字与机构名称无关，仅用于把稿件填充到最低字数要求。" * 10
    text = pad + "工业和信息化部工信部。科学技术部科技部。"
    assert 300 <= len(text) <= 1200, len(text)
    res = check_abbreviations(text, ["工业和信息化部", "科学技术部"], ["工信部", "科技部"])
    assert all(p.status == "review" for p in res.pairs)
    for p in res.pairs:
        assert any("语境不足" in r for r in p.reasons)


# ---------------- 出现位置与嵌套计数 ----------------


def test_spans_cover_every_occurrence(result):
    text = MANUSCRIPT
    for name in FULL + ABBR:
        expected = text.count(name)
        got = sum(1 for s in result.spans if s.name == name)
        assert got == expected, name


def test_embedded_abbreviation_not_double_counted():
    text = (
        "北京大学今天公布了新一年的本科招生方案，北大将继续扩大强基计划规模。"
        "北京大学校长在发布会上介绍了学科建设思路。"
        "清华大学随后也发布了人才培养改革举措，清华学堂班将新增两个交叉方向。"
        "两校负责人均表示，将加强本科阶段的通识教育与科研训练并重的培养模式。"
        "相关招生细则将于下周在两校官网公布，考生可及时关注报名时间与材料要求。"
        "专家建议考生结合自身兴趣理性选择志愿，避免盲目追逐热门专业与分数线。"
        "教育主管部门将全程监督招生流程，确保信息公开透明与录取公平公正。"
        "高校招生关系千家万户，各校还将举办多场线上线下相结合的校园开放日，"
        "并组织名师讲座、专业体验营与校园参观，帮助中学生充分了解各校的学科特色、"
        "师资力量与毕业生去向，从而做出更适合自身发展的升学选择。"
    )
    assert len(text) >= 300, len(text)
    res = check_abbreviations(text, ["北京大学", "清华大学"], ["北大", "清华"])
    # “清华”出现在“清华大学”内部一次（embedded），另有“清华学堂班”一次独立出现
    qinghua = next(p for p in res.pairs if p.abbreviation == "清华")
    assert len(qinghua.abbr_occurrences) == 2
    assert qinghua.standalone_count == 1
    embed_spans = [s for s in res.spans if s.embedded_in == "清华大学"]
    assert embed_spans and all(s.name == "清华" for s in embed_spans)
    # 嵌套出现仍在 spans 中保留位置信息
    assert all(0 <= s.start < s.end <= len(text) for s in res.spans)


def test_result_to_dict_is_json_friendly(result):
    import json

    d = result_to_dict(result)
    payload = json.dumps(d, ensure_ascii=False)
    again = json.loads(payload)
    assert again["full_names"] == FULL
    assert len(again["pairs"]) == len(FULL)
    assert "thresholds" in again and "spans" in again


def test_zero_standalone_abbreviation_warns():
    # 简称“清华”只作为“清华大学”的一部分出现，没有任何独立用法
    text = (
        "北京大学今天公布了新一年的本科招生方案，北大将继续扩大强基计划规模。"
        "北京大学校长在发布会上介绍了学科建设思路与人才培养改革方向。"
        "清华大学随后也发布了招生资讯，今年将新增两个交叉学科方向。"
        "两校负责人均表示，将加强本科阶段的通识教育与科研训练并重的培养模式。"
        "相关招生细则将于下周在两校官网公布，考生可及时关注报名时间与材料要求。"
        "专家建议考生结合自身兴趣理性选择志愿，避免盲目追逐热门专业与分数线。"
        "教育主管部门将全程监督招生流程，确保信息公开透明与录取公平公正。"
        "高校招生关系千家万户，各校还将举办多场线上线下相结合的校园开放日，"
        "并组织名师讲座、专业体验营与校园参观，帮助中学生充分了解各校的学科特色、"
        "师资力量与毕业生去向，从而做出更适合自身发展的升学选择。"
    )
    res = check_abbreviations(text, ["北京大学", "清华大学"], ["北大", "清华"])
    qinghua = next(p for p in res.pairs if p.abbreviation == "清华")
    assert qinghua.standalone_count == 0
    assert any("清华" in w and "没有独立出现" in w for w in res.warnings)


def test_deterministic_results():
    r1 = result_to_dict(check_abbreviations(MANUSCRIPT, FULL, ABBR))
    r2 = result_to_dict(check_abbreviations(MANUSCRIPT, FULL, ABBR))
    assert r1 == r2
