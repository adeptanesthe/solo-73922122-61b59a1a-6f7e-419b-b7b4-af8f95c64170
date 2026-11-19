# -*- coding: utf-8 -*-
"""机构简称核查核心算法（纯函数、无状态、不落盘）。

设计约束（对应需求）
====================
1. **只拟合本稿语境**：TF-IDF 的语料仅由“本稿件中各名称出现位置周围的上下文
   窗口”构成，不使用任何外部语料；特征为字符 2~3 元组，词表中不存在单字特征。
2. **语境 + 字形双证据**：语境余弦相似度（TF-IDF）与字形相似度（最长公共子序列
   类指标 + 子序列覆盖）加权合成综合得分。
3. **一对一最优对应**：简称与全称数量相等（2~5 个），用全排列枚举求总得分最高的
   一对一匹配（n<=5，至多 120 种排列，确定性平局打破），不做贪心近似。
4. **不强行确认**：替代对应得分接近、语境不足或综合得分偏低时一律标记“待核对”，
   并逐条给出判断依据。
5. **嵌在全称内的简称不重复计数**：简称出现位置若落在任一全称出现区间内，标记为
   embedded，不计入独立出现次数，但仍在原文位置中标出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import permutations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- 可调阈值（集中管理，便于核对） ----------------
MIN_MANUSCRIPT = 300          # 稿件字数下限
MAX_MANUSCRIPT = 1200         # 稿件字数上限
MIN_NAMES = 2                 # 全称/简称最少数量
MAX_NAMES = 5                 # 全称/简称最多数量
CONTEXT_WINDOW = 24           # 名称每侧取多少字符作为语境窗口
MIN_CONTEXT_CHARS = 20        # 单侧有效语境字符数低于此值视为语境不足
CONTEXT_WEIGHT = 0.6          # 语境相似度在综合得分中的权重
GLYPH_WEIGHT = 0.4            # 字形相似度在综合得分中的权重
CONFIRM_MIN_SCORE = 0.45      # 综合得分确认阈值
CONTEXT_MIN_SIM = 0.15        # 语境相似度确认下限
AMBIGUITY_MARGIN = 0.08       # 与次优对应得分差小于此值视为“替代对应接近”
# 语境窗口中替换名称串的占位符：所有名称统一替换成同一 token，
# 其 ngram 在各文档中同频、IDF 很低，避免名称自身字形把语境相似度“灌高”。
NAME_MASK = "机构名占位符"


class AbbrValidationError(ValueError):
    """输入校验失败，message 可直接展示给用户。"""


@dataclass
class Span:
    start: int
    end: int
    name: str
    kind: str  # "full" 全称 | "abbr" 简称
    embedded_in: str | None = None  # 被哪个名称的出现区间包含


@dataclass
class PairResult:
    abbreviation: str
    full_name: str
    status: str                      # "confirmed" | "review"
    score: float
    context_similarity: float
    glyph_similarity: float
    runner_up_full: str | None
    runner_up_score: float | None
    score_gap: float | None
    reasons: list[str] = field(default_factory=list)
    abbr_occurrences: list[dict] = field(default_factory=list)
    standalone_count: int = 0
    abbr_context_chars: int = 0
    full_context_chars: int = 0


@dataclass
class CheckResult:
    full_names: list[str]
    abbreviations: list[str]
    pairs: list[PairResult]
    spans: list[Span]
    warnings: list[str]
    feature_count: int
    context_docs: dict[str, str]


# ---------------- 基础工具 ----------------


def _find_occurrences(text: str, name: str) -> list[tuple[int, int]]:
    """返回 name 在 text 中全部不重叠出现区间 (start, end)。"""
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(name, start)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(name)))
        start = idx + len(name)


def _lcs_length(a: str, b: str) -> int:
    """最长公共子序列长度（名称很短，O(len(a)*len(b)) 足够）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def glyph_similarity(abbr: str, full: str) -> float:
    """字形相似度，落在 [0,1]。

    = 0.7 * SequenceMatcher 比率（基于匹配块，近似 LCS 比率）
      + 0.3 * 简称字符在全称中的子序列覆盖率
    （简称常由全称抽字构成，如“北京大学”→“北大”，子序列覆盖能捕捉这一关系。）
    """
    ratio = SequenceMatcher(None, abbr, full).ratio()
    coverage = _lcs_length(abbr, full) / len(abbr) if abbr else 0.0
    return 0.7 * ratio + 0.3 * coverage


def _mask_names_in_window(window: str, win_start: int,
                          all_spans: list[tuple[int, int, str]]) -> str:
    """把窗口内出现的任何已知名称统一替换为占位符。"""
    pieces: list[str] = []
    cursor = 0
    for s, e, _ in sorted(all_spans):
        if e <= win_start or s >= win_start + len(window):
            continue
        local_s = max(0, s - win_start)
        local_e = min(len(window), e - win_start)
        if local_s < cursor:
            continue  # 已被更长的名称区间覆盖
        pieces.append(window[cursor:local_s])
        pieces.append(NAME_MASK)
        cursor = local_e
    pieces.append(window[cursor:])
    return "".join(pieces)


def _build_context_doc(
    text: str,
    own_spans: list[tuple[int, int]],
    all_name_spans: list[tuple[int, int, str]],
) -> tuple[str, int]:
    """合并某名称所有出现位置周围的语境窗口，并返回有效语境字符数。

    优先使用未被其他名称包含的独立出现；若全部出现都嵌在更长名称内，
    则退而使用全部出现（并由调用方据此给出 warning）。
    """
    standalone = [
        (s, e) for (s, e) in own_spans
        if not any(os <= s and e <= oe and (os, oe) != (s, e)
                   for os, oe, _ in all_name_spans)
    ]
    chosen = standalone or own_spans
    windows: list[str] = []
    effective_chars = 0
    for s, e in chosen:
        ws = max(0, s - CONTEXT_WINDOW)
        we = min(len(text), e + CONTEXT_WINDOW)
        raw = text[ws:we]
        masked = _mask_names_in_window(raw, ws, all_name_spans)
        windows.append(masked)
        effective_chars += len(masked.replace(NAME_MASK, ""))
    return "。".join(windows), effective_chars


# ---------------- 输入校验 ----------------


def validate_inputs(manuscript: str, full_names: list[str],
                    abbreviations: list[str]) -> tuple[str, list[str], list[str]]:
    """校验并规范化输入；不合法时抛 AbbrValidationError（信息可直接展示）。"""
    if not isinstance(manuscript, str):
        raise AbbrValidationError("稿件必须是文本")
    manuscript = manuscript.strip()
    length = len(manuscript)
    if length < MIN_MANUSCRIPT or length > MAX_MANUSCRIPT:
        raise AbbrValidationError(
            f"稿件字数需在 {MIN_MANUSCRIPT}~{MAX_MANUSCRIPT} 字之间，当前为 {length} 字"
        )
    if not (isinstance(full_names, list) and isinstance(abbreviations, list)):
        raise AbbrValidationError("全称列表与简称列表格式不正确")
    if not (MIN_NAMES <= len(full_names) <= MAX_NAMES):
        raise AbbrValidationError(f"机构全称需提供 {MIN_NAMES}~{MAX_NAMES} 个")
    if len(full_names) != len(abbreviations):
        raise AbbrValidationError(
            f"全称与简称数量必须相同（当前全称 {len(full_names)} 个、简称 {len(abbreviations)} 个）"
        )
    full_names = [n.strip() for n in full_names]
    abbreviations = [n.strip() for n in abbreviations]
    if any(not n for n in full_names + abbreviations):
        raise AbbrValidationError("名称不能为空")
    if len(set(full_names)) != len(full_names):
        raise AbbrValidationError("机构全称存在重复项")
    if len(set(abbreviations)) != len(abbreviations):
        raise AbbrValidationError("机构简称存在重复项")
    overlap = set(full_names) & set(abbreviations)
    if overlap:
        raise AbbrValidationError(f"简称与全称不能相同：{sorted(overlap)}")
    missing = [n for n in full_names + abbreviations if n not in manuscript]
    if missing:
        raise AbbrValidationError(f"以下名称未在稿件中出现：{missing}")
    for n in full_names + abbreviations:
        if len(n) < 2:
            raise AbbrValidationError(f"名称过短，无法构造字符 2~3 元特征：{n}")
    return manuscript, full_names, abbreviations


# ---------------- 主流程 ----------------


def check_abbreviations(manuscript: str, full_names: list[str],
                        abbreviations: list[str]) -> CheckResult:
    """对单篇稿件执行简称→全称一对一核查。

    仅使用本稿件内名称周围的语境拟合字符 2~3 元 TF-IDF；
    全部中间量随返回值给出，便于在网页上展示判断依据。
    """
    manuscript, full_names, abbreviations = validate_inputs(
        manuscript, full_names, abbreviations
    )
    n = len(full_names)

    # 1) 收集所有名称的出现区间
    raw_spans: list[Span] = []
    span_tuples: list[tuple[int, int, str]] = []
    for name in full_names:
        for s, e in _find_occurrences(manuscript, name):
            raw_spans.append(Span(s, e, name, "full"))
            span_tuples.append((s, e, name))
    for name in abbreviations:
        for s, e in _find_occurrences(manuscript, name):
            raw_spans.append(Span(s, e, name, "abbr"))
            span_tuples.append((s, e, name))

    # 2) 标注包含关系：区间被另一个不同名称区间完全覆盖即为 embedded
    def covering_name(s: int, e: int, self_name: str) -> str | None:
        for os_, oe, oname in span_tuples:
            if oname == self_name:
                continue
            if os_ <= s and e <= oe:
                return oname
        return None

    for sp in raw_spans:
        sp.embedded_in = covering_name(sp.start, sp.end, sp.name)

    full_span_map = {
        name: [(sp.start, sp.end) for sp in raw_spans
               if sp.kind == "full" and sp.name == name]
        for name in full_names
    }
    abbr_span_map = {
        name: [(sp.start, sp.end) for sp in raw_spans
               if sp.kind == "abbr" and sp.name == name]
        for name in abbreviations
    }

    # 3) 仅用本稿名称周围语境构造文档并拟合字符 2~3 元 TF-IDF
    context_docs: dict[str, str] = {}
    context_chars: dict[str, int] = {}
    for name in full_names + abbreviations:
        own = full_span_map.get(name) or abbr_span_map.get(name)
        doc, chars = _build_context_doc(manuscript, own, span_tuples)
        context_docs[name] = doc
        context_chars[name] = chars

    ordered_names = full_names + abbreviations
    docs = [context_docs[name] for name in ordered_names]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(docs)
    feature_count = len(vectorizer.vocabulary_)
    # 硬性断言：纯字符 ngram(2,3) 词表不得出现单字特征
    assert all(len(feat) in (2, 3) for feat in vectorizer.vocabulary_)

    full_idx = {name: i for i, name in enumerate(full_names)}
    abbr_idx = {name: i for i, name in enumerate(abbreviations)}
    full_matrix = matrix[[full_idx[name] for name in full_names]]
    abbr_matrix = matrix[[n + len(full_names) for n in range(len(abbreviations))]]
    if feature_count:
        cos = cosine_similarity(abbr_matrix, full_matrix)
    else:  # 极端情况下语境无可用特征，全部按 0 处理并走“语境不足”分支
        cos = [[0.0] * n for _ in range(n)]

    # 4) 语境相似度 + 字形相似度 → 综合得分矩阵
    score = [[0.0] * n for _ in range(n)]
    glyph = [[0.0] * n for _ in range(n)]
    for ai, abbr in enumerate(abbreviations):
        for fi, full in enumerate(full_names):
            g = glyph_similarity(abbr, full)
            glyph[ai][fi] = g
            score[ai][fi] = CONTEXT_WEIGHT * float(cos[ai][fi]) + GLYPH_WEIGHT * g

    # 5) 全排列枚举一对一最优匹配（n<=5，至多 120 种；平局按排列字典序）
    best_perm: tuple[int, ...] | None = None
    best_total = -1.0
    for perm in permutations(range(n)):
        total = sum(score[ai][perm[ai]] for ai in range(n))
        if total > best_total + 1e-12 or (
            abs(total - best_total) <= 1e-12
            and (best_perm is None or perm < best_perm)
        ):
            best_total = total
            best_perm = perm
    assert best_perm is not None

    # 6) 逐对给出结论与依据
    pairs: list[PairResult] = []
    warnings: list[str] = []
    for name in full_names:
        occ = full_span_map[name]
        if occ and all(
            sp.embedded_in is not None
            for sp in raw_spans if sp.kind == "full" and sp.name == name
        ):
            warnings.append(
                f"全称“{name}”在稿中仅作为更长名称的一部分出现，独立语境缺失，结论需谨慎"
            )
    for name in abbreviations:
        standalone = sum(
            1 for sp in raw_spans
            if sp.kind == "abbr" and sp.name == name and sp.embedded_in is None
        )
        if standalone == 0:
            warnings.append(
                f"简称“{name}”在稿中没有独立出现（全部嵌在全称内，不重复计数），"
                "无法据独立语境核实其用法，请人工确认"
            )

    for ai, abbr in enumerate(abbreviations):
        fi = best_perm[ai]
        full = full_names[fi]
        s = score[ai][fi]
        ctx_sim = float(cos[ai][fi])
        g_sim = glyph[ai][fi]

        runner_j = max(
            (j for j in range(n) if j != fi),
            key=lambda j: score[ai][j],
        )
        runner_score = score[ai][runner_j]
        gap = s - runner_score

        reasons: list[str] = []
        if gap < AMBIGUITY_MARGIN:
            reasons.append(
                f"替代对应“{full_names[runner_j]}”得分 {runner_score:.3f}，"
                f"与最高分 {s:.3f} 差距仅 {gap:.3f}（< {AMBIGUITY_MARGIN:.2f}），存在指错风险"
            )
        if context_chars[abbr] < MIN_CONTEXT_CHARS:
            reasons.append(
                f"简称“{abbr}”有效语境仅 {context_chars[abbr]} 字"
                f"（阈值 {MIN_CONTEXT_CHARS} 字），语境不足"
            )
        if context_chars[full] < MIN_CONTEXT_CHARS:
            reasons.append(
                f"全称“{full}”有效语境仅 {context_chars[full]} 字"
                f"（阈值 {MIN_CONTEXT_CHARS} 字），语境不足"
            )
        if ctx_sim < CONTEXT_MIN_SIM:
            reasons.append(
                f"语境 TF-IDF 余弦相似度 {ctx_sim:.3f} 低于确认下限 "
                f"{CONTEXT_MIN_SIM:.2f}，语境证据偏弱"
            )
        if s < CONFIRM_MIN_SCORE:
            reasons.append(
                f"综合得分 {s:.3f} 低于确认阈值 {CONFIRM_MIN_SCORE:.2f}"
            )

        occs = [
            {
                "start": sp.start,
                "end": sp.end,
                "embedded_in": sp.embedded_in,
            }
            for sp in raw_spans
            if sp.kind == "abbr" and sp.name == abbr
        ]
        standalone = sum(1 for o in occs if o["embedded_in"] is None)

        pairs.append(
            PairResult(
                abbreviation=abbr,
                full_name=full,
                status="review" if reasons else "confirmed",
                score=round(s, 4),
                context_similarity=round(ctx_sim, 4),
                glyph_similarity=round(g_sim, 4),
                runner_up_full=full_names[runner_j],
                runner_up_score=round(runner_score, 4),
                score_gap=round(gap, 4),
                reasons=reasons,
                abbr_occurrences=occs,
                standalone_count=standalone,
                abbr_context_chars=context_chars[abbr],
                full_context_chars=context_chars[full],
            )
        )

    spans = sorted(raw_spans, key=lambda sp: (sp.start, sp.end))
    return CheckResult(
        full_names=full_names,
        abbreviations=abbreviations,
        pairs=pairs,
        spans=spans,
        warnings=warnings,
        feature_count=feature_count,
        context_docs=context_docs,
    )


def result_to_dict(result: CheckResult) -> dict:
    """把核查结果序列化为 API/前端可用的 JSON 结构。"""
    return {
        "full_names": result.full_names,
        "abbreviations": result.abbreviations,
        "feature_count": result.feature_count,
        "warnings": result.warnings,
        "thresholds": {
            "context_weight": CONTEXT_WEIGHT,
            "glyph_weight": GLYPH_WEIGHT,
            "confirm_min_score": CONFIRM_MIN_SCORE,
            "context_min_sim": CONTEXT_MIN_SIM,
            "ambiguity_margin": AMBIGUITY_MARGIN,
            "min_context_chars": MIN_CONTEXT_CHARS,
            "context_window": CONTEXT_WINDOW,
        },
        "pairs": [
            {
                "abbreviation": p.abbreviation,
                "full_name": p.full_name,
                "status": p.status,
                "score": p.score,
                "context_similarity": p.context_similarity,
                "glyph_similarity": p.glyph_similarity,
                "runner_up_full": p.runner_up_full,
                "runner_up_score": p.runner_up_score,
                "score_gap": p.score_gap,
                "reasons": p.reasons,
                "abbr_occurrences": p.abbr_occurrences,
                "standalone_count": p.standalone_count,
                "abbr_context_chars": p.abbr_context_chars,
                "full_context_chars": p.full_context_chars,
            }
            for p in result.pairs
        ],
        "spans": [
            {
                "start": sp.start,
                "end": sp.end,
                "name": sp.name,
                "kind": sp.kind,
                "embedded_in": sp.embedded_in,
            }
            for sp in result.spans
        ],
        "context_docs": result.context_docs,
    }
