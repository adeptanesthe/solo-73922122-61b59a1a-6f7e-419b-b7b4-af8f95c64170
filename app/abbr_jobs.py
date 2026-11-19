# -*- coding: utf-8 -*-
"""简称核查的异步任务注册表与世代保护（纯内存）。

并发语义（对应需求：改稿或改名称后立即撤下旧判断，迟到结果不能覆盖新输入）
==========================================================================
* 每次提交核查生成单调递增的 request_id（世代），旧任务立刻标记 stale，
  其结果即使稍后算完也不会成为“当前结果”；
* worker 线程池执行计算，写回前再次核对 request_id，迟到结果直接丢弃；
* 可选 compute_delay_ms 人为制造慢计算，方便在演示页观察过期保护；
* 与推荐系统的数据库世代不同：核查结果是随稿件即算即弃的，不需要落盘，
  因此用进程内注册表即可，语义与持久化世代完全一致。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import abbr_checker


class AbbrJobRegistry:
    def __init__(self, max_workers: int = 2) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._jobs: dict[int, dict[str, Any]] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers = max_workers

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="abbr-check",
            )

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        with self._lock:
            self._jobs.clear()

    # ---------------- 提交 / 查询 ----------------

    def submit(self, manuscript: str, full_names: list[str],
               abbreviations: list[str], compute_delay_ms: int = 0,
               client_token: int | None = None) -> dict[str, Any]:
        """登记新任务并使所有未完成任务立即过期。返回任务快照。

        client_token 由前端生成并原样回显，前端据此丢弃迟到响应；
        服务端自身的 request_id 世代独立维护，二者不必相等。
        """
        assert self._executor is not None, "registry 尚未 start"
        with self._lock:
            self._seq += 1
            request_id = self._seq
            for job in self._jobs.values():
                if job["status"] in ("pending", "running"):
                    job["status"] = "stale"
            # 只保留最近 20 条历史，避免演示中注册表无限增长
            if len(self._jobs) >= 20:
                for old_id in sorted(self._jobs)[:-19]:
                    self._jobs.pop(old_id, None)
            job = {
                "request_id": request_id,
                "client_token": client_token,
                "status": "pending",
                "submitted_at": time.time(),
                "compute_delay_ms": compute_delay_ms,
                "input_digest": {
                    "manuscript_len": len(manuscript),
                    "full_names": list(full_names),
                    "abbreviations": list(abbreviations),
                },
                "result": None,
                "error": None,
            }
            self._jobs[request_id] = job
        self._executor.submit(self._run, request_id, manuscript,
                              list(full_names), list(abbreviations),
                              compute_delay_ms)
        return self.snapshot(request_id)

    def snapshot(self, request_id: int) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(request_id)
            return dict(job) if job else None

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._jobs:
                return None
            return dict(self._jobs[max(self._jobs)])

    # ---------------- worker ----------------

    def _run(self, request_id: int, manuscript: str, full_names: list[str],
             abbreviations: list[str], compute_delay_ms: int) -> None:
        delay = max(0, compute_delay_ms) / 1000.0
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            time.sleep(min(0.1, deadline - time.monotonic()))
            with self._lock:
                if self._jobs.get(request_id, {}).get("status") == "stale":
                    return  # 已被新输入取代，不必再算
        try:
            result = abbr_checker.result_to_dict(
                abbr_checker.check_abbreviations(manuscript, full_names, abbreviations)
            )
        except abbr_checker.AbbrValidationError as exc:
            # 提交时已做校验，这里属于兜底：记录为 failed，不污染当前结果
            with self._lock:
                job = self._jobs.get(request_id)
                if job and job["status"] in ("pending", "running"):
                    job["status"] = "failed"
                    job["error"] = str(exc)
            return
        except Exception as exc:  # pragma: no cover - 防御性
            with self._lock:
                job = self._jobs.get(request_id)
                if job and job["status"] in ("pending", "running"):
                    job["status"] = "failed"
                    job["error"] = f"{type(exc).__name__}: {exc}"
            return
        # 双重检查：写回前确认自己仍是最新世代，迟到结果一律丢弃
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job["status"] not in ("pending", "running"):
                # 已被新输入取代（stale）或注册表已清空：结果不落库
                return
            job["status"] = "done"
            job["result"] = result


_registry = AbbrJobRegistry()


def get_registry() -> AbbrJobRegistry:
    return _registry
