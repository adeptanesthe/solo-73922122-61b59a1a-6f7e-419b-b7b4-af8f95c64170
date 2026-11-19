# -*- coding: utf-8 -*-
"""简称核查 API 与任务注册表测试：世代过期、迟到结果丢弃、输入校验。"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.abbr_jobs import get_registry
from app.main import app

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
PAYLOAD = {"manuscript": MANUSCRIPT, "full_names": FULL, "abbreviations": ABBR}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # lifespan 会初始化推荐系统的 SQLite，默认路径 /data 在本机不可写，
    # 这里仿照 test_api.py 把数据库指到临时目录。
    from app import database

    db_path = tmp_path / "abbr.db"
    monkeypatch.setenv("RECOMMENDER_DB", str(db_path))
    database._DB_PATH = str(db_path)

    registry = get_registry()
    registry.stop()
    registry.start()
    with TestClient(app) as c:
        yield c, registry
    registry.stop()


def _wait_job(c, request_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = c.get(f"/api/abbr/jobs/{request_id}").json()["job"]
        if data["status"] in ("done", "failed", "stale"):
            return data
        time.sleep(0.03)
    raise AssertionError("任务未按时结束")


def test_check_happy_path_returns_one_to_one_pairs(client):
    c, _ = client
    r = c.post("/api/abbr/check", json=PAYLOAD)
    assert r.status_code == 200
    job = r.json()["job"]
    assert job["status"] in ("pending", "running", "done")
    done = _wait_job(c, job["request_id"])
    assert done["status"] == "done"
    result = done["result"]
    mapping = {p["abbreviation"]: p["full_name"] for p in result["pairs"]}
    assert mapping == dict(zip(ABBR, FULL))
    assert result["spans"], "应返回原文位置"
    assert result["thresholds"]["context_weight"] + result["thresholds"]["glyph_weight"] == 1.0


def test_validation_errors_are_422(client):
    c, _ = client
    r = c.post("/api/abbr/check", json={**PAYLOAD, "manuscript": "太短"})
    assert r.status_code == 422
    assert "300~1200" in r.json()["detail"]

    r = c.post("/api/abbr/check", json={**PAYLOAD, "abbreviations": ABBR[:-1] + ["不存在机构"]})
    assert r.status_code == 422
    assert "未在稿件中出现" in r.json()["detail"]

    r = c.post("/api/abbr/check", json={
        **PAYLOAD, "full_names": FULL[:3], "abbreviations": ABBR[:2]
    })
    assert r.status_code == 422 and "数量必须相同" in r.json()["detail"]


def test_new_submission_marks_old_job_stale(client):
    c, registry = client
    first = c.post("/api/abbr/check", json={
        **PAYLOAD, "compute_delay_ms": 1500
    }).json()["job"]
    # 立刻提交第二次：旧任务应立即 stale，latest 指向新世代
    second = c.post("/api/abbr/check", json=PAYLOAD).json()["job"]
    assert second["request_id"] == first["request_id"] + 1
    old = registry.snapshot(first["request_id"])
    assert old["status"] == "stale"
    assert registry.latest()["request_id"] == second["request_id"]
    # 旧任务即使轮询也永远是 stale，不会带出结果
    waited = _wait_job(c, first["request_id"])
    assert waited["status"] == "stale"
    assert waited["result"] is None
    # 新任务正常完成
    done = _wait_job(c, second["request_id"])
    assert done["status"] == "done" and done["result"]


def test_late_result_never_overwrites_new_input(client):
    c, registry = client
    # 慢任务 A
    job_a = c.post("/api/abbr/check", json={
        **PAYLOAD, "compute_delay_ms": 1200
    }).json()["job"]
    time.sleep(0.1)
    # 快速任务 B 取代 A
    job_b = c.post("/api/abbr/check", json=PAYLOAD).json()["job"]
    done_b = _wait_job(c, job_b["request_id"])
    assert done_b["status"] == "done"

    # 等 A 的延迟窗口走完，worker 写回时必须发现自己已过期
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        a = registry.snapshot(job_a["request_id"])
        if a["status"] == "stale":
            break
        time.sleep(0.05)
    assert registry.snapshot(job_a["request_id"])["status"] == "stale"
    # 当前最新结果仍只属于 B
    latest = registry.latest()
    assert latest["request_id"] == job_b["request_id"]
    assert latest["result"] is not None


def test_latest_endpoint_and_unknown_job(client):
    c, _ = client
    assert c.get("/api/abbr/latest").json()["job"] is None
    r = c.post("/api/abbr/check", json=PAYLOAD).json()["job"]
    latest = c.get("/api/abbr/latest").json()["job"]
    assert latest["request_id"] == r["request_id"]
    assert c.get("/api/abbr/jobs/99999").status_code == 404


def test_delayed_job_runs_and_poll_succeeds(client):
    c, _ = client
    job = c.post("/api/abbr/check", json={
        **PAYLOAD, "compute_delay_ms": 300
    }).json()["job"]
    done = _wait_job(c, job["request_id"])
    assert done["status"] == "done"
    # 每个简称都带独立出现计数与判断依据
    for p in done["result"]["pairs"]:
        assert "reasons" in p and "runner_up_full" in p
        assert p["standalone_count"] >= 1
