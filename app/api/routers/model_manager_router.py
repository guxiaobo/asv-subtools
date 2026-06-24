"""
模型管理员路由：录音断句 / 说话人打标 / 增量训练 / 模型详情 / 发布。

专门从 auth_router.py 拆出，保持代码简洁。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

import html
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from services.auth import get_current_user, is_logged_in, require_role

ROLE_MODEL_MANAGER = "model_manager"

router = APIRouter()
logger = logging.getLogger("model_manager_router")

# Path resolution
# __file__ = app/api/routers/model_manager_router.py
#   .parent            = app/api/routers
#   .parent.parent     = app/api
#   .parent.parent.parent = app/
_APP_DIR = Path(__file__).resolve().parent.parent.parent  # app/
APP_DIR = _APP_DIR
PROJECT_ROOT = _APP_DIR.parent                              # asv-subtools/
TEMPLATES_DIR = APP_DIR / "api" / "templates"
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "training.db"
VAD_CONFIG_PATH = DATA_DIR / "vad_config.json"

# Default VAD config
DEFAULT_VAD_CONFIG = {
    "vad_threshold": 0.5,
    "min_segment_sec": 1.5,
    "max_segment_sec": 15.0,
    "snr_threshold": 15.0,
    "filter_leading_sec": 2.0,
    "target_sample_rate": 16000,
    "window_ms": 30,
    "channel_separated": False,
    "apply_noise_reduction": True,
    "diarizer_model": "CAM++",
    "diarizer_cluster_threshold": 0.55,
    "diarizer_agent_threshold": None,
    "min_segment_sec_ignore": 0.0,  # <= 此值时自动忽略片段（0=不启用）
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vad_config() -> dict:
    """加载 VAD 配置（从 JSON 文件或默认值）。"""
    if VAD_CONFIG_PATH.exists():
        try:
            return json.loads(VAD_CONFIG_PATH.read_text())
        except Exception:
            pass
    return dict(DEFAULT_VAD_CONFIG)


def _save_vad_config(cfg: dict) -> None:
    """保存 VAD 配置到 JSON 文件。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VAD_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def _recordings_db():
    """动态导入 recording_db 模块（避免循环依赖）。"""
    import sys
    if "services.recording_db" not in sys.modules:
        from services import recording_db as db
    else:
        db = sys.modules["services.recording_db"]
    return db


# ── 分段结果版本删除 ──


@router.delete("/model-manager/segments/{recording_id}/batch/{batch_id}")
async def delete_segmentation_batch(
    recording_id: int,
    batch_id: str,
    request: Request,
):
    """删除录音某版本的分段结果（需安全确认）。"""
    await require_role(ROLE_MODEL_MANAGER)(request)
    db = _recordings_db()
    try:
        check = await db.is_batch_deletable(recording_id, batch_id)
        if not check.get("safe"):
            refs = check.get("referenced_by", [])
            msg = f"该版本 {check['seg_count']} 段中有片段已被以下模型使用，无法删除："
            for r in refs:
                msg += f"\n  · {r['model_name']} ({r['version_tag']})"
            return {"success": False, "error": msg, "details": check}
        count = await db.delete_batch_segments(recording_id, batch_id)
        return {"success": True, "deleted": count}
    except Exception as e:
        logger.exception("delete batch failed")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# VAD 配置
# ---------------------------------------------------------------------------

@router.get("/model-manager/vad-config", response_class=HTMLResponse)
async def vad_config_page(request: Request):
    """VAD 配置页面。"""
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user if hasattr(request.state, 'current_user') else None
    user = user or {}
    cfg = _load_vad_config()
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VAD 参数配置 - 声纹管理系统</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50; color:#fff; padding:12px 24px; display:flex; align-items:center; gap:20px; font-size:14px }}
.nav a {{ color:#bdc3c7; text-decoration:none }}
.nav a:hover {{ color:#fff }}
.nav .user {{ margin-left:auto; font-size:13px }}
.container {{ max-width:800px; margin:24px auto; padding:0 16px }}
.card {{ background:#fff; border-radius:8px; padding:24px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.1) }}
.card h2 {{ font-size:18px; margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid #eee }}
.form-row {{ display:flex; align-items:center; margin-bottom:14px; gap:12px }}
.form-row label {{ width:200px; font-size:14px; color:#555; flex-shrink:0 }}
.form-row input[type=number], .form-row select {{ flex:1; max-width:300px; padding:8px 12px; border:1px solid #ddd; border-radius:6px; font-size:14px }}
.form-row .hint {{ font-size:12px; color:#999; margin-left:8px }}
.form-row input[type=checkbox] {{ width:20px; height:20px }}
.btn-primary {{ background:#3498db; color:#fff; border:none; padding:10px 28px; border-radius:6px; cursor:pointer; font-size:15px }}
.btn-primary:hover {{ background:#2980b9 }}
.btn-secondary {{ background:#95a5a6; color:#fff; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:14px; text-decoration:none; display:inline-block }}
.msg {{ padding:10px 16px; border-radius:6px; margin-bottom:16px; font-size:14px }}
.msg.success {{ background:#d4edda; color:#155724; border:1px solid #c3e6cb }}
.msg.error {{ background:#f8d7da; color:#721c24; border:1px solid #f5c6cb }}
.status-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px }}
.badge-ok {{ background:#d4edda; color:#155724 }}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">VAD参数</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="card">
        <h2>VAD 切割参数配置</h2>
        <form id="vadForm" onsubmit="saveConfig(event)">
            <div class="form-row">
                <label>VAD 阈值 (0~1)</label>
                <input type="number" name="vad_threshold" value="{cfg['vad_threshold']}" min="0.01" max="0.99" step="0.01">
                <span class="hint">值越高越敏感</span>
            </div>
            <div class="form-row">
                <label>最小时长 (秒)</label>
                <input type="number" name="min_segment_sec" value="{cfg['min_segment_sec']}" min="0.5" max="60" step="0.1">
            </div>
            <div class="form-row">
                <label>自动忽略短于 (秒)</label>
                <input type="number" name="min_segment_sec_ignore" value="{cfg.get('min_segment_sec_ignore', 0)}" min="0" max="60" step="0.1">
                <span class="hint">0=不启用，>0 时长低于此值的片段自动忽略</span>
            </div>
            <div class="form-row">
                <label>最大片段时长 (秒)</label>
                <input type="number" name="max_segment_sec" value="{cfg['max_segment_sec']}" min="1" max="300" step="0.1">
            </div>
            <div class="form-row">
                <label>SNR 阈值 (dB)</label>
                <input type="number" name="snr_threshold" value="{cfg['snr_threshold']}" min="0" max="50" step="0.5">
                <span class="hint">低于此值标记为噪声</span>
            </div>
            <div class="form-row">
                <label>去除首尾静音 (秒)</label>
                <input type="number" name="filter_leading_sec" value="{cfg['filter_leading_sec']}" min="0" max="30" step="0.1">
            </div>
            <div class="form-row">
                <label>目标采样率</label>
                <select name="target_sample_rate">
                    <option value="8000" {"selected" if cfg['target_sample_rate']==8000 else ""}>8000 Hz</option>
                    <option value="16000" {"selected" if cfg['target_sample_rate']==16000 else ""}>16000 Hz</option>
                    <option value="44100" {"selected" if cfg['target_sample_rate']==44100 else ""}>44100 Hz</option>
                </select>
            </div>
            <div class="form-row">
                <label>说话人分离模型</label>
                <select name="diarizer_model">
                    <option value="CAM++" {"selected" if cfg['diarizer_model']=='CAM++' else ""}>CAM++ (192维)</option>
                    <option value="ResNet34" {"selected" if cfg['diarizer_model']=='ResNet34' else ""}>ResNet34 (256维)</option>
                    <option value="ECAPA" {"selected" if cfg['diarizer_model']=='ECAPA' else ""}>ECAPA (192维)</option>
                </select>
            </div>
            <div class="form-row">
                <label>客户聚类阈值</label>
                <input type="number" name="diarizer_cluster_threshold" value="{cfg['diarizer_cluster_threshold']}" min="0.1" max="1" step="0.01">
            </div>
            <div class="form-row">
                <label>降噪</label>
                <input type="checkbox" name="apply_noise_reduction" value="1" {"checked" if cfg.get('apply_noise_reduction') else ""}>
            </div>
            <div class="form-row" style="margin-top:20px">
                <button type="submit" class="btn-primary">保存配置</button>
            </div>
        </form>
    </div>
</div>
<script>
async function saveConfig(e) {{
    e.preventDefault();
    const form = document.getElementById('vadForm');
    const data = {{}};
    new FormData(form).forEach((v, k) => {{ 
        if (k === 'apply_noise_reduction') data[k] = true;
        else if (k === 'diarizer_agent_threshold' && !v) data[k] = null;
        else if (['vad_threshold','min_segment_sec','max_segment_sec','snr_threshold','filter_leading_sec','target_sample_rate','window_ms','diarizer_cluster_threshold','min_segment_sec_ignore'].includes(k))
            data[k] = parseFloat(v);
        else data[k] = v;
    }});
    if (!data.apply_noise_reduction) data.apply_noise_reduction = false;
    try {{
        const resp = await fetch('/model-manager/vad-config/api', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(data) }});
        const result = await resp.json();
        if (result.success) alert('配置已保存');
        else alert('保存失败: ' + JSON.stringify(result.error));
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
}}
</script>
</body>
</html>"""
    return HTMLResponse(page)


@router.post("/model-manager/vad-config/api")
async def save_vad_config(request: Request):
    """保存 VAD 配置到 JSON 文件（API）。"""
    try:
        cfg = await request.json()
        _save_vad_config(cfg)
        return {"success": True}
    except Exception as e:
        logger.exception("VAD config save failed")
        return {"success": False, "error": str(e)}


@router.get("/model-manager/vad-config/api")
async def get_vad_config(request: Request):
    """获取当前 VAD 配置（API）。"""
    return {"success": True, "config": _load_vad_config()}


# ---------------------------------------------------------------------------
# 录音断句页面
# ---------------------------------------------------------------------------

@router.get("/model-manager/segments", response_class=HTMLResponse)
async def segment_manager_page(
    request: Request,
    agent_id: str = "",
    customer_id: str = "",
    date_from: str = "",
    date_to: str = "",
    status: str = "",
    page: int = 1,
    per_page: int = 50,
):
    """录音断句管理页面。"""
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user if hasattr(request.state, 'current_user') else None

    # Load recording data
    db = _recordings_db()
    vad_cfg = _load_vad_config()

    recordings = []
    total_count = 0
    try:
        # 仅当设置了筛选条件时加载录音列表
        has_filter = bool(agent_id or customer_id or date_from or date_to or status)
        if has_filter:
            offset_val = (page - 1) * per_page
            recordings, total_count = await db.list_recordings_with_segments_paginated(
                agent_id=agent_id,
                customer_id=customer_id,
                date_from=date_from,
                date_to=date_to,
                status_filter=status,
                limit=per_page,
                offset=offset_val,
            )
        seg_stats = await db.get_segment_stats()
    except Exception as e:
        logger.error("Failed to load recordings: %s", e)
        seg_stats = {"total_recordings": 0, "segmented": 0,
                     "labeled": 0, "pending": 0, "unsegmentable": 0, "unlabeled": 0}

    # Get agent list for filter dropdown
    agents = []
    try:
        agents = await db.get_users_by_role("agent")
    except Exception:
        pass

    total_pages = max(1, (total_count + per_page - 1) // per_page) if total_count else 1

    page = _render_segment_page(user, recordings, seg_stats, agents, vad_cfg,
                                agent_id, customer_id, date_from, date_to, status,
                                page, total_pages, total_count, bool(agent_id or customer_id or date_from or date_to or status))
    return HTMLResponse(page)


import html
from datetime import datetime


def _pending_hint(pre_status: str, pre_queued_at: Optional[str]) -> str:
    """根据 pre_status + pre_queued_at 时长生成进度提示（方案A）。

    返回内联 HTML 片段，追加在 VAD 状态徽章后。规则：
      - done/failed/其他非活跃状态：无提示
      - pending（首次断句派发）：
          · pre_queued_at 为空（老数据）：待处理
          · <2 分钟：⏳ 等待处理中…
          · 2~5 分钟：⏳ 处理较慢，请稍候（橙）
          · ≥5 分钟：⚠ 可能超时，建议重试（红）
      - reprocessing（重新断句已派发，等待脚本启动）：
          同 pending 时长规则，但文案前缀"重新断句"
      - processing（脚本正在执行 VAD/diarize）：
          · 永远显示 🔄 处理中…（蓝），不计算超时
            （processing 由脚本写入，一旦进入说明 claim 成功，时长不反映卡死；
             卡死时 recover 会把它转回 reprocessing 并记时间戳）
    """
    # processing：脚本正在跑，无法按 queued_at 判断超时（claim 时已清空）
    if pre_status == "processing":
        return '<span style="font-size:11px;color:#3498db;margin-left:4px">🔄 处理中…</span>'

    # pending / reprocessing：基于 pre_queued_at 时长判断
    if pre_status not in ("pending", "reprocessing"):
        return ""

    prefix = "重新断句：" if pre_status == "reprocessing" else ""

    if not pre_queued_at:
        return f'<span style="font-size:11px;color:#999;margin-left:4px">{prefix}待处理</span>'
    try:
        queued = datetime.fromisoformat(pre_queued_at)
        elapsed = (datetime.now() - queued).total_seconds()
    except (ValueError, TypeError):
        return f'<span style="font-size:11px;color:#999;margin-left:4px">{prefix}待处理</span>'
    if elapsed < 120:
        return f'<span style="font-size:11px;color:#3498db;margin-left:4px">⏳ {prefix}等待处理中…</span>'
    elif elapsed < 300:
        return f'<span style="font-size:11px;color:#e67e22;margin-left:4px">⏳ {prefix}处理较慢，请稍候</span>'
    else:
        return f'<span style="font-size:11px;color:#e74c3c;margin-left:4px">⚠ {prefix}可能超时，建议重试</span>'


def _seg_view_link(rec: dict, batch: str, can_segment: bool, pre_status: str = "") -> str:
    """生成录音片段详情链接。

    - done: 蓝色"查看片段"可点击
    - unsegmentable: 灰色"无法断句"不可点击（VAD 已分析但无语音段）
    - 其他（pending/processing 等）: 灰色"未断句"不可点击
    """
    if can_segment:
        url = f'/model-manager/segments/{rec["id"]}?batch={batch}'
        return f'<a href="{html.escape(url)}" class="btn-sm" style="background:#3498db">查看片段</a>'
    if pre_status == "unsegmentable":
        return '<span class="btn-sm" style="background:#7f8c8d;cursor:default">无法断句</span>'
    return '<span class="btn-sm" style="background:#95a5a6;cursor:default">未断句</span>'


def _render_segment_page(user, recordings, seg_stats, agents, vad_cfg,
                         agent_id, customer_id, date_from, date_to, status,
                         current_page, total_pages, total_count, has_filter):
    """渲染录音断句页面 HTML（含分页、筛选、多选断句）。"""
    user = user or {}
    rows_html = ""
    for rec in recordings:
        seg_count = rec.get("seg_count", 0)
        seg_ignored = rec.get("seg_ignored", 0)
        pre_status = rec.get("pre_status", "pending")
        batch = rec.get("latest_batch", "v1")

        status_color = {"pending": "#f39c12", "processing": "#3498db",
                        "reprocessing": "#9b59b6", "unsegmentable": "#7f8c8d",
                        "done": "#27ae60", "failed": "#e74c3c"}.get(pre_status, "#95a5a6")

        seg_info = f"{seg_count} 段"
        if seg_ignored > 0:
            seg_info += f' <span style="color:#e74c3c">({seg_ignored} 忽略)</span>'

        can_segment = pre_status == "done"
        # checkbox 可选/灰掉规则：
        #   pending 无 pre_queued_at → 可勾选（从未提交，新鲜录音）
        #   pending 有 pre_queued_at → 灰掉（已提交，等待 VAD 处理中）
        #   reprocessing/processing → 灰掉（正在处理中）
        #   unsegmentable → 灰掉（无法断句）
        #   done/failed → 可勾选（可重新提交）
        queued = bool(rec.get("pre_queued_at"))
        checkbox_disabled = (
            pre_status in ("reprocessing","processing","unsegmentable")
            or (pre_status == "pending" and queued)
        )
        chk = f'<input type="checkbox" class="rec-check" value="{rec["id"]}" data-status="{pre_status}"'
        if checkbox_disabled:
            chk += ' disabled'
        chk += '>'
        vad_hint = _pending_hint(pre_status, rec.get("pre_queued_at"))
        rows_html += f"""<tr>
            <td>{chk}</td>
            <td>{rec.get('id','')}</td>
            <td>{rec.get('agent_id','')}</td>
            <td>{rec.get('customer_id','')}</td>
            <td>{rec.get('call_timestamp','')[:16]}</td>
            <td>{rec.get('call_id','')}</td>
            <td><span style="color:{status_color};font-weight:bold">{pre_status}</span>{vad_hint}</td>
            <td>{seg_info}</td>
            <td>
                {_seg_view_link(rec, batch, can_segment, pre_status)}
                {"" if can_segment else ""}
            </td>
        </tr>"""

    if not recordings and has_filter:
        rows_html = "<tr><td colspan='9' style='text-align:center;color:#999;padding:30px'>无匹配录音</td></tr>"

    # Agent filter options
    agent_opts = '<option value="">全部坐席</option>'
    for a in agents:
        sel = 'selected' if a.get('agent_id','') == agent_id else ''
        name = html.escape(str(a.get("display_name", "") or a.get("agent_id", "")))
        agent_opts += f'<option value="{html.escape(str(a["agent_id"]))}" {sel}>{name}</option>'

    # Pagination
    pag_html = ""
    if has_filter and total_pages > 1:
        qs = f"agent_id={agent_id}&customer_id={customer_id}&date_from={date_from}&date_to={date_to}&status={status}"
        prev_dis = "disabled" if current_page <= 1 else ""
        next_dis = "disabled" if current_page >= total_pages else ""
        pag_html = f"""<div class="pagination">
            <a href="/model-manager/segments?{qs}&page={current_page-1}&per_page=50" class="btn {"btn-disabled" if prev_dis else "btn-primary"}" {prev_dis}>← 上一页</a>
            <span class="page-info">第 {current_page}/{total_pages} 页（共 {total_count} 条）</span>
            <a href="/model-manager/segments?{qs}&page={current_page+1}&per_page=50" class="btn {"btn-disabled" if next_dis else "btn-primary"}" {next_dis}>下一页 →</a>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>录音断句 - 声纹管理系统</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50; color:#fff; padding:12px 24px; display:flex; align-items:center; gap:20px; font-size:14px }}
.nav a {{ color:#bdc3c7; text-decoration:none }}
.nav a:hover {{ color:#fff }}
.nav .user {{ margin-left:auto; font-size:13px }}
.nav .nav-active {{ color:#fff; font-weight:bold }}
.container {{ max-width:1200px; margin:24px auto; padding:0 16px }}
.card {{ background:#fff; border-radius:8px; padding:20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.1) }}
.card h2 {{ font-size:18px; margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid #eee }}
.filter-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; align-items:end }}
.filter-row select, .filter-row input {{ padding:7px 10px; border:1px solid #ddd; border-radius:5px; font-size:13px }}
.filter-row label {{ font-size:13px; color:#555 }}
.btn {{ padding:7px 16px; border:none; border-radius:5px; cursor:pointer; font-size:13px; text-decoration:none; display:inline-block }}
.btn-primary {{ background:#3498db; color:#fff }}
.btn-primary:hover {{ background:#2980b9 }}
.btn-primary:disabled {{ background:#95a5a6; cursor:default }}
.btn-danger {{ background:#e74c3c; color:#fff }}
.btn-success {{ background:#27ae60; color:#fff }}
.btn-sm {{ padding:4px 10px; border:none; border-radius:4px; cursor:pointer; font-size:12px; color:#fff; text-decoration:none; display:inline-block }}
.btn-disabled {{ background:#95a5a6 !important; cursor:default !important; padding:7px 16px; border:none; border-radius:5px; font-size:13px; color:#fff !important; display:inline-block; text-decoration:none }}
table {{ width:100%; border-collapse:collapse }}
th {{ background:#f8f9fa; padding:10px 8px; text-align:left; font-size:13px; font-weight:600; border-bottom:2px solid #dee2e6 }}
td {{ padding:8px; font-size:13px; border-bottom:1px solid #eee }}
tr:hover {{ background:#f8f9fa }}
.summary-row {{ display:flex; gap:12px; margin-bottom:16px }}
.summary-item {{ flex:1; text-align:center; padding:14px; border-radius:8px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.1) }}
.summary-num {{ font-size:26px; font-weight:bold; color:#2c3e50 }}
.summary-label {{ font-size:12px; color:#777; margin-top:4px }}
.pagination {{ display:flex; align-items:center; justify-content:center; gap:12px; margin-top:16px; padding:12px 0 }}
.page-info {{ font-size:13px; color:#666 }}
input[type=checkbox]:disabled {{ opacity:0.35; cursor:not-allowed }}
input[type=checkbox]:disabled+label {{ color:#999 }}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">录音断句</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="summary-row">
        <div class="summary-item">
            <div class="summary-num">{seg_stats.get("total_recordings",0)}</div>
            <div class="summary-label">总录音数</div>
        </div>
        <div class="summary-item">
            <div class="summary-num">{seg_stats.get("segmented",0)}</div>
            <div class="summary-label">已断句录音</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:{'#e67e22' if seg_stats.get('pending',0)>0 else '#27ae60'}">{seg_stats.get("pending",0)}</div>
            <div class="summary-label">未断句录音</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:{'#e74c3c' if seg_stats.get('unsegmentable',0)>0 else '#27ae60'}">{seg_stats.get("unsegmentable",0)}</div>
            <div class="summary-label">无法断句</div>
        </div>
        <div class="summary-item">
            <div class="summary-num">{seg_stats.get("labeled",0)}</div>
            <div class="summary-label">已打标片段</div>
        </div>
        <div class="summary-item">
            <div class="summary-num" style="color:{'#dc2626' if seg_stats.get('unlabeled',0)>0 else '#16a34a'}">{seg_stats.get("unlabeled",0)}</div>
            <div class="summary-label">未打标片段</div>
        </div>
    </div>

    <div class="card">
        <h2>录音断句</h2>
        <form method="get" action="/model-manager/segments" class="filter-row" id="filterForm">
            <div>
                <label style="display:block">坐席</label>
                <select name="agent_id">{agent_opts}</select>
            </div>
            <div>
                <label style="display:block">客户</label>
                <input type="text" name="customer_id" value="{customer_id}" placeholder="客户ID">
            </div>
            <div>
                <label style="display:block">日期从</label>
                <input type="date" name="date_from" value="{date_from}">
            </div>
            <div>
                <label style="display:block">日期到</label>
                <input type="date" name="date_to" value="{date_to}">
            </div>
            <div>
                <label style="display:block">状态</label>
                <select name="status">
                    <option value="">全部</option>
                    <option value="done" {"selected" if status=='done' else ""}>已断句</option>
                    <option value="pending" {"selected" if status=='pending' else ""}>未断句</option>
                    <option value="unsegmentable" {"selected" if status=='unsegmentable' else ""}>无法断句</option>
                    <option value="failed" {"selected" if status=='failed' else ""}>处理异常</option>
                </select>
            </div>
            <div>
                <button type="submit" class="btn btn-primary" style="margin-top:18px">🔍 筛选</button>
                <a href="/model-manager/segments" class="btn btn-primary" style="background:#95a5a6;margin-top:18px">重置</a>
            </div>
        </form>

        <div style="margin-bottom:12px;display:flex;gap:8px;align-items:center">
            <button class="btn btn-success" id="runVadBtn" onclick="runVad()" disabled>⏳ 开始断句（选中后激活）</button>
            <span style="font-size:12px;color:#999">勾选待断句录音后激活按钮</span>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width:36px"><input type="checkbox" id="checkAll" onchange="toggleAll()"></th>
                    <th>ID</th><th>坐席</th><th>客户</th><th>时间</th><th>文件名</th><th>VAD状态</th><th>断句</th><th>操作</th>
                </tr>
            </thead>
            <tbody>{rows_html if has_filter else '<tr><td colspan="9" style="text-align:center;color:#999;padding:40px">请在上方设置筛选条件后点击 <strong>🔍 筛选</strong> 查看录音清单</td></tr>'}</tbody>
        </table>

        {pag_html}
    </div>
</div>
<script>
document.querySelectorAll('.rec-check').forEach(function(cb) {{
    cb.addEventListener('change', updateVadBtn);
}});

function updateVadBtn() {{
    var checked = document.querySelectorAll('.rec-check:checked:not(:disabled)');
    var btn = document.getElementById('runVadBtn');
    if (checked.length > 0) {{
        btn.disabled = false;
        btn.textContent = '🔊 开始断句（已选 ' + checked.length + ' 条）';
    }} else {{
        btn.disabled = true;
        btn.textContent = '⏳ 开始断句（选中后激活）';
    }}
}}

function toggleAll() {{
    var all = document.getElementById('checkAll').checked;
    document.querySelectorAll('.rec-check:not(:disabled)').forEach(function(cb) {{ cb.checked = all; }});
    updateVadBtn();
}}

async function runVad() {{
    var checked = document.querySelectorAll('.rec-check:checked:not(:disabled)');
    var ids = Array.from(checked).map(function(cb) {{ return parseInt(cb.value); }});
    if (ids.length === 0) {{ alert('请先选择待断句的录音'); return; }}
    if (!confirm('确定对选中的 ' + ids.length + ' 条录音执行断句吗？')) return;
    var btn = document.getElementById('runVadBtn');
    btn.disabled = true; btn.textContent = '提交中...';
    try {{
        var resp = await fetch('/model-manager/run-preprocess', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{recording_ids: ids}})
        }});
        var result = await resp.json();
        if (result.success) {{
            alert('断句任务已提交，刷新后查看进度。');
            btn.textContent = '✅ 已提交…';
            location.reload();
        }} else {{
            alert('启动失败: ' + (result.error || JSON.stringify(result)));
            btn.disabled = false; btn.textContent = '开始断句';
        }}
    }} catch(e) {{ alert('网络错误: ' + e.message); btn.disabled = false; btn.textContent = '开始断句'; }}
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 单个录音的片段详情页
# ---------------------------------------------------------------------------

@router.get("/model-manager/segments/{recording_id}", response_class=HTMLResponse)
async def recording_segments_page(
    request: Request,
    recording_id: int,
    batch: str = "",
):
    """查看指定录音的断句片段列表。"""
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    user = await get_current_user(request)
    if user.get("role") != ROLE_MODEL_MANAGER:
        return RedirectResponse(url="/login", status_code=302)
    db = _recordings_db()

    # Get recording info
    recording = None
    segments = []
    batches = []
    try:
        conn = await db._open_conn()
        try:
            cursor = await conn.execute(
                "SELECT * FROM recordings WHERE id = ?", (recording_id,)
            )
            row = await cursor.fetchone()
            recording = dict(row) if row else None
        finally:
            await conn.close()

        if not batch:
            batch = await db.get_latest_batch_for_recording(recording_id)
        segments = await db.get_segments_by_recording(recording_id, batch_id=batch)
        batches = await db.get_batches_with_counts(recording_id)
    except Exception as e:
        logger.error("Failed to load segments for recording %d: %s", recording_id, e)

    if not recording:
        return HTMLResponse("<h1>录音不存在</h1><a href='/model-manager/segments'>返回</a>", status_code=404)

    existing_speakers = []
    try:
        existing_speakers = await db.list_speakers_from_segments()
    except Exception as e:
        logger.error("Failed to load speakers: %s", e)

    return HTMLResponse(_render_recording_detail(user, recording, segments, batches, batch, recording_id, existing_speakers))


@router.get("/model-manager/evaluate", response_class=HTMLResponse)
async def evaluate_redirect(request: Request):
    """模型评估页面 — 重定向到首页（评估为 POST 内联操作）。"""
    return RedirectResponse(url="/model-manager", status_code=302)


def _render_recording_detail(user, recording, segments, batches, current_batch, recording_id, existing_speakers=None):
    """渲染录音详情页（含片段列表）。"""
    user = user or {}
    existing_speakers = existing_speakers or []

    # 收集当前录音本身的坐席ID和客户ID（作为快速选项）
    rec_agent = (recording.get("agent_id") or "").strip()
    rec_cust = (recording.get("customer_id") or "").strip()

    # Build datalist options for searchable input
    dl_opts = ""
    seen_ids = set()
    if rec_agent:
        dl_opts += f'<option value="{rec_agent}">'
        seen_ids.add(rec_agent)
    if rec_cust:
        dl_opts += f'<option value="{rec_cust}">'
        seen_ids.add(rec_cust)

    # 已有说话人（去重，跳过已预置的）
    for sp in existing_speakers:
        label = (sp.get("speaker_label") or "").strip()
        if not label or label in seen_ids:
            continue
        seen_ids.add(label)
        dl_opts += f'<option value="{label}">'

    # 噪音/忽略作为可选值
    dl_opts += '<option value="__noise__">'

    seg_rows = ""
    for seg in segments:
        ignore_link = f"/model-manager/segments/{seg['id']}/toggle-ignore"
        seg_class = "seg-ignored" if seg.get("is_ignored") else ""
        ignore_btn = "取消忽略" if seg.get("is_ignored") else "忽略"
        ignore_btn_class = "btn-sm" if not seg.get("is_ignored") else "btn-sm btn-sm-warning"

        speaker_type = seg.get("speaker_type", "unknown")
        speaker_type_label = {"agent": "坐席", "customer": "客户", "unknown": "未知", "ignored": "已忽略"}.get(speaker_type, speaker_type)

        current_label = seg.get('speaker_label', '') or speaker_type_label

        seg_rows += f"""<tr class="{seg_class}">
            <td>{seg.get('segment_index','')}</td>
            <td>{seg.get('start_sec',''):.1f}s</td>
            <td>{seg.get('end_sec',''):.1f}s</td>
            <td>{seg.get('duration_sec',''):.1f}s</td>
            <td>
                <button class="btn-sm" style="background:#2c3e50" onclick="playAudio('{seg['id']}')">▶ 播放</button>
            </td>
            <td>
                <span id="label-{seg['id']}">{current_label}</span>
            </td>
            <td>
                <a href="javascript:void(0)" onclick="toggleIgnore({seg['id']})" id="ignore-{seg['id']}" class="{ignore_btn_class}">{ignore_btn}</a>
            </td>
            <td style="max-width:220px">
                <input type="text" id="sel-{seg['id']}" class="speaker-search" list="dl-{seg['id']}"
                       placeholder="搜索或输入说话人ID" value="{current_label}">
                <datalist id="dl-{seg['id']}">{dl_opts}</datalist>
                <button class="btn-sm" style="background:#8e44ad" onclick="setLabel({seg['id']})">设置</button>
            </td>
        </tr>"""

    if not segments:
        seg_rows = "<tr><td colspan='8' style='text-align:center;color:#999;padding:30px'>暂无断句片段，请先执行 VAD 断句</td></tr>"

    # Batch selector — batches is list of dicts from get_batches_with_counts
    batch_opts = ""
    total = len(batches)
    for idx, b in enumerate(batches):
        bid = b["batch_id"]
        seg_count = b["seg_count"]
        is_latest = b.get("is_latest", False)
        version_num = total - idx  # 最新=版本1
        # 提取时间戳批次的日期部分显示在 tooltip 中
        date_str = ""
        if bid.startswith("v") and len(bid) > 9 and bid[1:9].isdigit():
            date_str = f"（{bid[1:5]}-{bid[5:7]}-{bid[7:9]}）"
        label = f"版本 {version_num}{date_str}"
        if is_latest:
            label += " ✨最新"
        sel = "selected" if bid == current_batch else ""
        batch_opts += f'<option value="{bid}" {sel}>{label}（{seg_count}段）</option>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>录音详情 - {recording.get('call_id','')} - 声纹管理系统</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50; color:#fff; padding:12px 24px; display:flex; align-items:center; gap:20px; font-size:14px }}
.nav a {{ color:#bdc3c7; text-decoration:none }}
.nav a:hover {{ color:#fff }}
.nav .user {{ margin-left:auto; font-size:13px }}
.container {{ max-width:1100px; margin:24px auto; padding:0 16px }}
.card {{ background:#fff; border-radius:8px; padding:20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.1) }}
.card h2 {{ font-size:18px; margin-bottom:16px }}
.info-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:16px }}
.info-item {{ padding:8px 12px; background:#f8f9fa; border-radius:6px }}
.info-label {{ font-size:11px; color:#888 }}
.info-value {{ font-size:14px; font-weight:bold }}
table {{ width:100%; border-collapse:collapse }}
th {{ background:#f8f9fa; padding:10px 8px; text-align:left; font-size:13px; font-weight:600; border-bottom:2px solid #dee2e6 }}
td {{ padding:8px; font-size:13px; border-bottom:1px solid #eee }}
tr:hover {{ background:#f8f9fa }}
tr.seg-ignored td {{ color:#999; text-decoration:line-through }}
.btn {{ padding:7px 16px; border:none; border-radius:5px; cursor:pointer; font-size:13px; text-decoration:none; display:inline-block }}
.btn-primary {{ background:#3498db; color:#fff }}
.btn-primary:hover {{ background:#2980b9 }}
.btn-sm {{ padding:4px 10px; border:none; border-radius:4px; cursor:pointer; font-size:12px; color:#fff; background:#7f8c8d; display:inline-block; text-decoration:none }}
.btn-sm-warning {{ background:#e67e22 !important }}
audio {{ width:100%; margin-top:8px }}
.batch-select {{ padding:6px 10px; border:1px solid #ddd; border-radius:5px; font-size:13px; margin-right:8px }}
.speaker-select {{ padding:3px 6px; border:1px solid #ddd; border-radius:3px; font-size:12px; max-width:130px }}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">录音详情</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div style="margin-bottom:12px">
        <a href="/model-manager/segments" class="btn btn-primary" style="background:#95a5a6">← 返回列表</a>
    </div>
    
    <div class="card">
        <h2>录音信息</h2>
        <div class="info-grid">
            <div class="info-item"><div class="info-label">ID</div><div class="info-value">{recording.get('id','')}</div></div>
            <div class="info-item"><div class="info-label">坐席</div><div class="info-value">{recording.get('agent_id','')}</div></div>
            <div class="info-item"><div class="info-label">客户</div><div class="info-value">{recording.get('customer_id','')}</div></div>
            <div class="info-item"><div class="info-label">文件名</div><div class="info-value">{recording.get('call_id','')}</div></div>
            <div class="info-item"><div class="info-label">时间</div><div class="info-value">{recording.get('call_timestamp','')}</div></div>
            <div class="info-item"><div class="info-label">时长</div><div class="info-value">{(recording.get('duration_sec') or 0):.1f}s</div></div>
        </div>
        <div style="margin-top:12px">
            <label>断句版本:</label>
            <select class="batch-select" onchange="switchBatch(this.value)">
                {batch_opts}
            </select>
            <button class="btn btn-primary" onclick="reprocessRecording()">🔁 重新断句</button>
            <button class="btn btn-sm" style="background:#e74c3c;margin-left:8px" onclick="deleteBatch()">🗑 删除此版本</button>
        </div>
    </div>

    <div class="card">
        <h2>断句片段 ({len(segments)} 段)</h2>
        <div id="audioPlayer" style="display:none;margin-bottom:12px">
            <audio id="player" controls></audio>
        </div>
        <table>
            <thead>
                <tr><th>序号</th><th>起始</th><th>结束</th><th>时长</th><th>播放</th><th>说话人</th><th>忽略</th><th>设置标签</th></tr>
            </thead>
            <tbody>{seg_rows}</tbody>
        </table>
    </div>
</div>
<script>
const audioPlayer = document.getElementById('audioPlayer');
const player = document.getElementById('player');

function playAudio(segId) {{
    fetch('/model-manager/segments/' + segId + '/stream')
        .then(r => r.blob())
        .then(blob => {{
            audioPlayer.style.display = 'block';
            player.src = URL.createObjectURL(blob);
            player.play();
        }});
}}

async function setLabel(segId) {{
    const input = document.getElementById('sel-' + segId);
    let label = input.value.trim();
    if (!label) return;
    let speakerType = '';
    if (label === '__noise__') {{
        speakerType = 'ignored';
    }}
    try {{
        const resp = await fetch('/model-manager/set-label', {{
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{segment_id: segId, speaker_label: label, speaker_type: speakerType}})
        }});
        const result = await resp.json();
        if (result.success) location.reload();
        else alert('设置失败: ' + (result.error || ''));
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
}}

async function toggleIgnore(segId) {{
    try {{
        const resp = await fetch('/model-manager/segments/' + segId + '/toggle-ignore', {{ method:'POST' }});
        const result = await resp.json();
        if (result.success) location.reload();
        else alert('操作失败');
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
}}

function switchBatch(batch) {{
    location.href = '/model-manager/segments/{recording_id}?batch=' + batch;
}}

async function reprocessRecording() {{
    if (!confirm('确定要重新断句此录音吗？旧结果会被保留在新版本中。')) return;
    const btn = event.target; btn.disabled = true; btn.textContent = '处理中...';
    try {{
        const resp = await fetch('/model-manager/run-preprocess', {{ 
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{recording_id: {recording.get('id','')}}})
        }});
        const result = await resp.json();
        if (result.success) {{ alert('重新断句已启动'); location.reload(); }}
        else alert('启动失败: ' + (result.error || ''));
        btn.disabled = false; btn.textContent = '🔁 重新断句';
    }} catch(e) {{ alert('网络错误: ' + e.message); btn.disabled = false; btn.textContent = '🔁 重新断句'; }}
}}

async function deleteBatch() {{
    const sel = document.querySelector('.batch-select');
    const batch = sel ? sel.value : '';
    if (!batch || !confirm('确定要删除断句版本「' + sel.options[sel.selectedIndex].text + '」吗？此操作不可恢复。')) return;
    const btn = event && event.target; if (btn) {{ btn.disabled = true; btn.textContent = '删除中...'; }}
    try {{
        const resp = await fetch('/model-manager/segments/{recording_id}/batch/' + encodeURIComponent(batch), {{ method:'DELETE' }});
        const result = await resp.json();
        if (result.success) {{ alert('已删除 ' + result.deleted + ' 段'); location.reload(); }}
        else alert('删除失败: ' + (result.error || ''));
        if (btn) {{ btn.disabled = false; btn.textContent = '🗑 删除此版本'; }}
    }} catch(e) {{ alert('网络错误: ' + e.message); if (btn) {{ btn.disabled = false; btn.textContent = '🗑 删除此版本'; }} }}
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Audio segment streaming
# ---------------------------------------------------------------------------

@router.get("/model-manager/segments/audio/{seg_id}")
async def stream_segment_audio(seg_id: int):
    """流式返回单个断句片段的音频文件。"""
    db = _recordings_db()
    try:
        segments = await db.get_segments_by_recording(0)  # dummy, need dedicated query
        # Quick direct query
        conn = await db._open_conn()
        try:
            cursor = await conn.execute(
                "SELECT file_path FROM audio_segments WHERE id = ?", (seg_id,)
            )
            row = await cursor.fetchone()
        finally:
            await conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Segment not found")

        audio_path = row["file_path"]
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(audio_path, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stream segment %d: %s", seg_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Segment actions
# ---------------------------------------------------------------------------

@router.post("/model-manager/segments/{seg_id}/toggle-ignore")
async def toggle_segment_ignore(seg_id: int):
    """切换断句片段的忽略状态。"""
    db = _recordings_db()
    try:
        # Get current state
        conn = await db._open_conn()
        try:
            cursor = await conn.execute(
                "SELECT is_ignored FROM audio_segments WHERE id = ?", (seg_id,)
            )
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(404, "Segment not found")
            new_state = 0 if row["is_ignored"] else 1
        finally:
            await conn.close()
        await db.set_segment_ignored(seg_id, ignored=bool(new_state))
        return {"success": True, "ignored": bool(new_state)}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/model-manager/segments/{seg_id}/label")
async def label_segment(seg_id: int, request: Request):
    """手动设置片段标签，支持已有说话人/新说话人/噪音，记录打标历史。"""
    db = _recordings_db()
    try:
        data = await request.json()
        speaker_label = data.get("speaker_label", "")
        label_source = data.get("label_source", "manual")
        speaker_type = data.get("speaker_type", "")
        update_trained = data.get("update_trained_status", False)

        # Handle "__noise__" — mark as ignored
        if speaker_label == "__noise__":
            await db.update_segment_label(
                seg_id,
                speaker_label="",
                speaker_type="ignored",
                label_source=label_source,
            )
            await db.set_segment_ignored(seg_id, ignored=True)
            # Log
            await db.log_segment_label_change(
                segment_id=seg_id,
                old_label="",
                new_label="__noise__",
                old_ignored=0,
                new_ignored=1,
                operated_by="admin",
            )
            return {"success": True}

        # Get current label for history
        old_seg = await db.get_segment_by_id(seg_id)

        await db.update_segment_label(
            seg_id,
            speaker_label=speaker_label,
            speaker_type=speaker_type,
            label_source=label_source,
        )

        # If user changed the label, reset trained_status
        if update_trained:
            conn = await db._open_conn()
            try:
                await conn.execute(
                    "UPDATE audio_segments SET trained_status = 'untrained' WHERE id = ?",
                    (seg_id,),
                )
                await conn.commit()
            finally:
                await conn.close()

        # Log label change
        old_label = (old_seg or {}).get("speaker_label", "")
        await db.log_segment_label_change(
            segment_id=seg_id,
            old_label=old_label,
            new_label=speaker_label,
            operated_by="admin",
        )

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
# ---------------------------------------------------------------------------
# l14: 说话人打标页面
@router.get("/model-manager/label", response_class=HTMLResponse)
async def label_speakers_page(
    request: Request,
    speaker_type: str = "",
    label_status: str = "",
    segment_limit: int = 200,
    model_name: str = "",
    checkpoint_id: str = "",
):
    """说话人打标页面：支持手动选说话人/新说话人 + 模型自动打标预览。"""
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user
    db = _recordings_db()
    recordings = []
    try:
        recordings = await db.list_recordings_with_segments(
            status_filter="done" if not label_status else label_status,
            limit=segment_limit,
        )
    except Exception as e:
        logger.error("label page: %s", e)

    all_segments = []
    unlabeled_count = 0
    ignored_count = 0
    for rec in recordings:
        rec_id = rec.get("id")
        try:
            segs = await db.get_segments_by_recording(rec_id)
            for s in segs:
                if speaker_type and s.get("speaker_type") != speaker_type:
                    continue
                all_segments.append({**s, "recording_info": rec})
                if s.get("is_ignored"):
                    ignored_count += 1
                elif not s.get("speaker_label"):
                    unlabeled_count += 1
        except Exception:
            pass
    total_valid = len(all_segments) - ignored_count
    labeled_count = total_valid - unlabeled_count

    # Get existing speakers + available checkpoints
    existing_speakers = []
    checkpoints = []
    recommended_info = None
    try:
        existing_speakers = await db.list_speakers_from_segments()
        cps = await db.list_checkpoints(limit=50)
        for cp in cps:
            st = cp.get("status", "")
            if st in ("done", "published", "pretrained", "incremental"):
                checkpoints.append(cp)

        # 从 model_evaluations 获取最佳性能记录
        conn = await db._open_conn()
        try:
            import json as _json
            # 找 EER 最低的评估（有评估数据时用，否则 fallback 到最新 incremental）
            best = await conn.execute(
                "SELECT me.*, c.model_name, c.version_tag FROM model_evaluations me "
                "JOIN checkpoints c ON me.checkpoint_id = c.id "
                "WHERE me.eer IS NOT NULL "
                "ORDER BY me.eer ASC LIMIT 1"
            )
            best_row = await best.fetchone()
            if best_row:
                recommended_info = dict(best_row)
        except Exception:
            pass
        finally:
            await conn.close()

    except Exception as e:
        logger.error("label page data: %s", e)

    # Build model + version selectors
    model_opts = '<option value="">-- 选择模型 --</option>'
    cp_opts = '<option value="">-- 选择版本 --</option>'
    seen_models = set()
    first_model = None
    for cp in checkpoints:
        mn = cp.get("model_name", "")
        if mn and mn not in seen_models:
            seen_models.add(mn)
            if first_model is None:
                first_model = mn
            sel = 'selected' if model_name == mn else ''
            if not model_name and first_model:
                sel = 'selected' if mn == first_model else ''
            model_opts += f'<option value="{mn}" {sel}>{mn}</option>'

    # 版本列表：默认只显示第一个或选中模型的版本
    effective_model = model_name or first_model or ""

    # cp_info 用作 JS 数据源：包含全部 checkpoint（供动态切换筛选）
    cp_info = []
    for cp in checkpoints:
        cp_info.append({
            "id": cp["id"],
            "model_name": cp.get("model_name", ""),
            "version_tag": cp.get("version_tag", "") or f"v{cp.get('id','')}",
            "file_path": cp.get("file_path", ""),
            "embedding_dim": cp.get("embedding_dim", 192),
            "recommended": cp.get("status") == "incremental",
        })

    # cp_opts 仅渲染当前选中模型的版本（初始 HTML）
    best_cp_id = None
    for cp in checkpoints:
        if effective_model and cp.get("model_name") != effective_model:
            continue
        vt = cp.get("version_tag", "") or f"v{cp.get('id','')}"
        sel = 'selected' if str(cp.get("id","")) == checkpoint_id else ''
        is_recommended = cp.get("status") == "incremental"
        rec_label = " ★推荐" if is_recommended else ""
        cp_opts += f'<option value="{cp["id"]}" data-model="{cp.get("model_name","")}" {sel}>{cp["id"]} - {cp.get("model_name","")} ({vt}){rec_label}</option>'
        if is_recommended and best_cp_id is None:
            best_cp_id = cp["id"]

    # Build recommendation banner
    recommendation_html = ""
    if recommended_info:
        eer_str = f"{recommended_info['eer']:.4f}" if recommended_info.get('eer') is not None else "—"
        acc_str = f"{recommended_info['accuracy']:.1%}" if recommended_info.get('accuracy') is not None else "—"
        dcf_str = f"{recommended_info['min_dcf']:.4f}" if recommended_info.get('min_dcf') is not None else "—"
        recommendation_html = (
            '<div style="background:#ebf5fb;border:1px solid #aed6f1;border-radius:6px;padding:10px 14px;margin-bottom:14px">'
            '<strong>🏆 推荐模型：</strong>'
            f'{recommended_info["model_name"]} ({recommended_info["version_tag"]})'
            f' · EER={eer_str} · minDCF={dcf_str} · ACC={acc_str}'
            f' <span style="color:#888;font-size:11px">（{recommended_info.get("dataset_desc","")}）</span>'
            '</div>'
        )

    # Build datalist options for searchable manual marking
    dl_opts = ""
    seen_ids = set()
    for sp in existing_speakers:
        label = (sp.get("speaker_label") or "").strip()
        if not label or label in seen_ids:
            continue
        seen_ids.add(label)
        dl_opts += f'<option value="{label}">'
    dl_opts += '<option value="__noise__">'

    rows = ""
    for seg in all_segments:
        rec = seg.get("recording_info", {})
        color = {"agent": "#3498db", "customer": "#27ae60", "ignored": "#95a5a6"}.get(
            seg.get("speaker_type", ""), "#95a5a6"
        )
        lbl = seg.get("speaker_label") or "—"
        sid = seg["id"]
        is_ignored = seg.get("is_ignored", 0)
        ts_label = seg.get("trained_status", "")
        ts_badge = ""
        if ts_label == "trained":
            ts_badge = '<span style="color:#27ae60;font-size:10px">✓已训练</span>'
        elif ts_label == "training":
            ts_badge = '<span style="color:#f39c12;font-size:10px">⏳训练中</span>'
        elif seg.get("speaker_label") and ts_label == "untrained":
            ts_badge = '<span style="color:#e67e22;font-size:10px">⚠️未训练</span>'
        if is_ignored:
            ts_badge = '<span style="color:#95a5a6;font-size:10px">已忽略</span>'

        rows += (
            "<tr>"
            f'<td>{rec.get("id","")}</td>'
            f'<td>{rec.get("agent_id","")}</td>'
            f'<td>{seg.get("segment_index","")}</td>'
            f'<td>{seg.get("duration_sec",0):.1f}s</td>'
            f'<td><span class="badge" style="background:{color};color:#fff">{seg.get("speaker_type","")}</span></td>'
            f'<td><span id="lbl-{sid}">{lbl}</span>{ts_badge}</td>'
            f'<td><button class="btn-sm" style="background:#2c3e50" onclick="playAudio({sid})">▶</button></td>'
            '<td>'
            f'<input type="text" id="sel-{sid}" class="label-search" list="dl-{sid}" placeholder="搜索或输入说话人ID" value="{sid}">'
            f'<datalist id="dl-{sid}">{dl_opts}</datalist>'
            f'<button class="btn-sm" style="background:#8e44ad" onclick="setLabel({sid})">✓</button>'
            "</td>"
            # Auto-label preview column
            f'<td id="auto-{sid}">—</td>'
            "</tr>"
        )

    if not all_segments:
        rows = "<tr><td colspan='10' style='text-align:center;color:#999;padding:30px'>暂无待打标片段</td></tr>"

    # Serialize checkpoint info for JS
    import json
    cp_json = json.dumps(cp_info, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>说话人打标 - 声纹管理系统</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:20px;font-size:14px }}
.nav a {{ color:#bdc3c7;text-decoration:none }}
.nav .user{{margin-left:auto;font-size:13px}}
.container{{max-width:1200px;margin:24px auto;padding:0 16px}}
.card{{background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.card h2{{font-size:18px;margin-bottom:16px}}
.summary-bar{{display:flex;gap:16px;margin-bottom:16px}}
.summary-item{{flex:1;text-align:center;padding:16px;border-radius:8px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.summary-num{{font-size:28px;font-weight:bold}}
.summary-label{{font-size:12px;color:#777}}
table{{width:100%;border-collapse:collapse}}
th{{background:#f8f9fa;padding:8px;text-align:left;font-size:12px;border-bottom:2px solid #dee2e6}}
td{{padding:6px 8px;font-size:13px;border-bottom:1px solid #eee}}
.badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px}}
.btn{{padding:7px 16px;border:none;border-radius:5px;cursor:pointer;font-size:13px}}
.btn-primary{{background:#3498db;color:#fff}}
.btn-sm{{padding:3px 8px;border:none;border-radius:3px;cursor:pointer;font-size:11px;color:#fff}}
.label-select{{padding:3px 4px;border:1px solid #ddd;border-radius:3px;font-size:11px;max-width:100px}}
.label-search{{width:100px;padding:3px 6px;border:1px solid #ddd;border-radius:3px;font-size:11px}}
#audioPlayer{{display:none;margin-bottom:12px}}
audio{{width:100%}}
.filter-bar{{display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap}}
.filter-bar label{{font-size:12px;color:#555}}
.filter-bar select,.filter-bar input{{padding:4px 8px;border:1px solid #ddd;border-radius:4px;font-size:12px}}
.auto-preview{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:12px;margin-bottom:12px;display:none}}
.auto-preview h3{{font-size:14px;margin-bottom:8px}}
.preview-result{{font-size:12px;color:#555;margin-bottom:4px}}
.preview-accept{{margin:4px 0}}
.preview-accept label{{font-size:12px;margin-left:4px}}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">说话人打标</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="summary-bar">
        <div class="summary-item"><div class="summary-num">{total_valid}</div><div class="summary-label">总声纹片段</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#27ae60">{labeled_count}</div><div class="summary-label">已打标片段</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#e67e22">{unlabeled_count}</div><div class="summary-label">待打标片段</div></div>
    </div>

    <!-- 系统自动打标面板 -->
    <div class="card">
        <h2>系统自动打标</h2>
        {recommendation_html}
        <div class="filter-bar">
            <label>模型：</label>
            <select id="autoModel" onchange="onModelChange()">
                {model_opts}
            </select>
            <label>版本：</label>
            <select id="autoCheckpoint">{cp_opts}</select>
            <label style="margin-left:12px">阈值：</label>
            <input type="number" id="autoThreshold" value="0.35" min="0.1" max="0.9" step="0.05" style="width:60px;padding:4px 6px;border:1px solid #ddd;border-radius:4px;font-size:12px">
            <label>最少片段：</label>
            <input type="number" id="autoMinSeg" value="2" min="1" max="10" step="1" style="width:50px;padding:4px 6px;border:1px solid #ddd;border-radius:4px;font-size:12px">
            <button class="btn btn-primary" onclick="previewAutoLabel()">🔍 预览自动打标</button>
            <button class="btn" style="background:#27ae60;color:#fff" onclick="confirmAutoLabel()" id="btnConfirmAuto" disabled>💾 全部确认保存</button>
        </div>
        <div id="autoPreview" class="auto-preview">
            <h3>自动打标预览</h3>
            <div id="autoPreviewText">执行预览后显示各片段识别结果...</div>
        </div>
    </div>

    <div class="card">
        <h2>手动打标</h2>
        <div class="filter-bar">
            <label>类型：</label>
            <select onchange="location.href='/model-manager/label?speaker_type='+this.value+'&label_status={label_status}&model_name={model_name}&checkpoint_id={checkpoint_id}'">
                <option value="">全部</option>
                <option value="agent" {'selected' if speaker_type=='agent' else ''}>坐席</option>
                <option value="customer" {'selected' if speaker_type=='customer' else ''}>客户</option>
                <option value="unknown" {'selected' if speaker_type=='unknown' else ''}>未知</option>
            </select>
            <label>状态：</label>
            <select onchange="location.href='/model-manager/label?label_status='+this.value+'&speaker_type={speaker_type}&model_name={model_name}&checkpoint_id={checkpoint_id}'">
                <option value="">全部（已断句）</option>
                <option value="all" {'selected' if label_status=='all' else ''}>全部</option>
            </select>
            <span style="font-size:11px;color:#999">选择片段 → 选说话人 → 点 ✓ 确认</span>
        </div>
        <div id="audioPlayer"><audio id="player" controls></audio></div>
        <table>
            <thead><tr><th>录音ID</th><th>坐席</th><th>片段</th><th>时长</th><th>类型</th><th>标签</th><th>播放</th><th>手动标记</th><th>自动结果</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
</div>
<script>
var checkpoints = {cp_json};
var autoResults = {{}};  // segId -> {{speaker_label, score, reason}}

function playAudio(segId) {{
    document.getElementById("audioPlayer").style.display = "block";
    document.getElementById("player").src = "/model-manager/segments/audio/" + segId;
    document.getElementById("player").play();
}}

function onModelChange() {{
    const mn = document.getElementById("autoModel").value;
    const cp = document.getElementById("autoCheckpoint");
    cp.innerHTML = '<option value="">-- 选择版本 --</option>';
    var firstRecommended = null;
    checkpoints.forEach(function(c) {{
        if (!mn || c.model_name === mn) {{
            var opt = document.createElement("option");
            opt.value = c.id;
            opt.textContent = c.id + " - " + c.model_name + " (" + c.version_tag + ")" + (c.recommended ? " ★推荐" : "");
            if (c.recommended && !firstRecommended) firstRecommended = c.id;
            cp.appendChild(opt);
        }}
    }});
    // Auto-select recommended version
    if (firstRecommended) cp.value = firstRecommended;
}}

async function setLabel(segId) {{
    const input = document.getElementById("sel-" + segId);
    var label = input.value.trim();
    if (!label) return;
    var speakerType = "";
    if (label === "__noise__") {{
        speakerType = "ignored";
    }}
    try {{
        const resp = await fetch("/model-manager/segments/" + segId + "/label", {{
            method: "POST", headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{
                speaker_label: label,
                label_source: "manual",
                speaker_type: speakerType,
                update_trained_status: true
            }})
        }});
        const r = await resp.json();
        if (r.success) {{
            var displayLabel = (label === "__noise__") ? "🔇 噪音" : label;
            document.getElementById("lbl-" + segId).textContent = displayLabel;
        }} else {{
            alert("设置失败: " + (r.error || ""));
        }}
    }} catch(e) {{ alert("网络错误: " + e.message); }}
}}

function getCheckpointInfo(cpId) {{
    for (var i = 0; i < checkpoints.length; i++) {{
        if (String(checkpoints[i].id) === String(cpId)) return checkpoints[i];
    }}
    return null;
}}

async function previewAutoLabel() {{
    const cpId = document.getElementById("autoCheckpoint").value;
    if (!cpId) {{ alert("请选择模型版本"); return; }}

    const btn = event.target; btn.disabled = true;
    var origText = btn.textContent; btn.textContent = "分析中...";

    try {{
        const resp = await fetch("/model-manager/run-label", {{
            method: "POST",
            headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{
                checkpoint_id: parseInt(cpId),
                preview_only: true,
                model_name: document.getElementById("autoModel").value,
                threshold: parseFloat(document.getElementById("autoThreshold").value) || 0.35,
                min_segments: parseInt(document.getElementById("autoMinSeg").value) || 2,
            }})
        }});
        const r = await resp.json();

        if (r.error) {{
            document.getElementById("autoPreview").style.display = "block";
            document.getElementById("autoPreviewText").innerHTML = '<span style="color:#e74c3c">错误: ' + r.error + '</span>';
            btn.disabled = false; btn.textContent = origText;
            return;
        }}

        // r.results is {{segId: {{speaker_label, score, reason}}}}
        autoResults = r.results || {{}};
        var previewHtml = "";
        var matchCount = 0;
        var newSpeakerCount = 0;
        var segIds = Object.keys(autoResults);
        for (var si = 0; si < segIds.length; si++) {{
            var segId = segIds[si];
            var res = autoResults[segId];
            if (!res) continue;
            matchCount++;

            // Group: "已匹配" vs "新说话人候选"
            var isNewCandidate = res.speaker_label && res.speaker_label.startsWith("NEW_");
            if (isNewCandidate) newSpeakerCount++;

            // Detailed preview line
            var icon = isNewCandidate ? "🆕" : "✅";
            var reasonHtml = res.reason ? '<span style="color:#888;font-size:11px"> ' + res.reason + '</span>' : "";
            previewHtml += '<div class="preview-result">';
            previewHtml += '  ' + icon + ' 片段#' + segId + ': <strong>' + (res.speaker_label || "?") + '</strong>';
            previewHtml += '  相似度: ' + (res.score != null ? res.score.toFixed(3) : "—");
            previewHtml += reasonHtml;
            previewHtml += '</div>';

            // Update table cell with checkbox + label + score
            var autoCell = document.getElementById("auto-" + segId);
            if (autoCell) {{
                var labelDisplay = isNewCandidate ? res.speaker_label.replace("NEW_", "🆕新:") : res.speaker_label;
                var scoreStr = res.score != null ? res.score.toFixed(3) : "";
                var bgColor = isNewCandidate ? "#fff3cd" : (res.score >= 0.5 ? "#d1fae5" : "#fef3c7");
                autoCell.innerHTML = '<input type="checkbox" class="auto-accept" data-seg="' + segId + '" checked onchange="updateConfirmBtn()"> '
                    + '<span style="background:' + bgColor + ';padding:2px 5px;border-radius:3px;font-size:11px">'
                    + '<strong>' + labelDisplay + '</strong>'
                    + ' <span style="color:#666">' + scoreStr + '</span>'
                    + '</span>';
            }}
        }}

        if (matchCount === 0) {{
            previewHtml = '<span style="color:#888">所有片段已有标签或已被忽略，无需自动打标。</span>';
        }}

        document.getElementById("autoPreview").style.display = "block";
        var summaryHtml = '<div style="margin-bottom:6px;font-size:13px">'
            + '匹配 <strong>' + matchCount + '</strong> 段'
            + (newSpeakerCount ? ', 其中 <strong>' + newSpeakerCount + '</strong> 段被推荐为新说话人' : '')
            + '</div>';
        document.getElementById("autoPreviewText").innerHTML = summaryHtml + previewHtml;
        updateConfirmBtn();
    }} catch(e) {{
        document.getElementById("autoPreview").style.display = "block";
        document.getElementById("autoPreviewText").innerHTML = '<span style="color:#e74c3c">请求失败: ' + e.message + '</span>';
    }}
    btn.disabled = false; btn.textContent = origText;
}}

function updateConfirmBtn() {{
    var checkboxes = document.querySelectorAll(".auto-accept:checked");
    document.getElementById("btnConfirmAuto").disabled = (checkboxes.length === 0);
}}

async function confirmAutoLabel() {{
    if (!confirm("确定用自动打标结果更新所有勾选的片段？")) return;
    var checkboxes = document.querySelectorAll(".auto-accept:checked");
    var toSave = [];
    checkboxes.forEach(function(cb) {{
        var segId = cb.getAttribute("data-seg");
        var res = autoResults[segId];
        if (res) toSave.push(segId);
    }});

    if (toSave.length === 0) {{ alert("没有勾选的片段"); return; }}

    var btn = document.getElementById("btnConfirmAuto");
    var origText = btn.textContent; btn.disabled = true; btn.textContent = "保存中...";

    try {{
        const resp = await fetch("/model-manager/run-label", {{
            method: "POST",
            headers:{{"Content-Type":"application/json"}},
            body: JSON.stringify({{
                confirm_segments: toSave,
                results: autoResults,
                model_name: document.getElementById("autoModel").value,
                checkpoint_id: parseInt(document.getElementById("autoCheckpoint").value),
            }})
        }});
        const r = await resp.json();
        if (r.success) {{
            alert("已保存 " + (r.saved_count || toSave.length) + " 个片段标签");
            location.reload();
        }} else {{
            alert("保存失败: " + (r.error || ""));
        }}
    }} catch(e) {{ alert("网络错误: " + e.message); }}
    btn.disabled = false; btn.textContent = origText;
}}
</script>
</body>
</html>"""
    return HTMLResponse(html)
# ---------------------------------------------------------------------------
# l16: 增量训练页面
# ---------------------------------------------------------------------------

@router.get("/model-manager/training", response_class=HTMLResponse)
async def training_page(request: Request):
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user
    db = _recordings_db()
    try:
        versions = await db.list_model_versions(limit=50)
        pending = await db.count_pending_training()
    except Exception as e:
        logger.error("training page: %s", e)
        versions = []
        pending = 0
    return HTMLResponse(_render_training_page(user, versions, pending))


def _render_training_page(user, versions, pending):
    user = user or {}
    version_rows = ""
    for v in versions:
        score_str = f"{v.get('score',''):.4f}" if v.get("score") else "—"
        metrics_short = ""
        try:
            import json as j
            m = j.loads(v.get("metrics","{}"))
            if m:
                metrics_short = f'<span class="badge badge-blue">sep={m.get("Sep","?"):.3f}</span> ' if "Sep" in m else ""
                metrics_short += f'<span class="badge badge-green">within={m.get("Within","?"):.3f}</span>' if "Within" in m else ""
        except Exception:
            pass
        sb = {"training":"badge-orange","done":"badge-green","published":"badge-blue","failed":"badge-red","draft":"badge-gray"}
        sb_cls = sb.get(v.get("status",""), "badge-gray")
        pub_badge = '<span class="badge badge-blue">已发布</span>' if v.get("status") == "published" else ""
        version_rows += (
            "<tr>"
            f'<td>{v.get("id","")}</td>'
            f'<td>{v.get("model_name","")}</td>'
            f'<td>{v.get("version_tag","")}</td>'
            f'<td>{v.get("base_model","")}</td>'
            f'<td>{v.get("embedding_dim","")}d</td>'
            f'<td><span class="badge {sb_cls}">{v.get("status","")}</span></td>'
            f'<td>{score_str}</td><td>{metrics_short}</td>'
            f'<td>{v.get("created_at","")}</td>'
            f'<td><button class="btn-sm" style="background:#8e44ad" onclick="compareVersion({v["id"]})">对比</button>{pub_badge}</td>'
            "</tr>"
        )
    if not versions:
        version_rows = "<tr><td colspan='10' style='text-align:center;color:#999;padding:30px'>暂无训练记录</td></tr>"

    suggestions = ""
    if not versions:
        suggestions = '<div class="training-panel"><strong>📋 训练建议：</strong> 暂无训练历史。建议先完成录音断句和说话人打标，然后启动第一次增量训练。</div>'
    else:
        latest = versions[-1]
        best = max(versions, key=lambda kv: kv.get("score") or 0)
        if latest.get("status") == "failed":
            suggestions = '<div class="training-panel" style="background:#fef2f2;border-color:#fca5a5"><strong>⚠️ 训练失败：</strong> 最新训练失败，请检查训练数据或模型配置后重试。</div>'
        elif latest.get("score") and best.get("score"):
            ratio = (latest["score"] / best["score"] * 100) if best["score"] else 0
            if ratio < 95:
                suggestions = f'<div class="training-panel" style="background:#fff7ed;border-color:#fed7aa"><strong>💡 提示：</strong> 最新版本评分 ({latest["score"]:.4f}) 低于最佳版本 ({best["score"]:.4f})，建议调整训练参数或增加标注数据。</div>'
            else:
                suggestions = f'<div class="training-panel" style="background:#f0fdf4;border-color:#bbf7d0"><strong>✅ 状态良好：</strong> 训练效果良好，最新版本评分 {latest["score"]:.4f}。</div>'
        else:
            suggestions = '<div class="training-panel"><strong>📊</strong> 已有训练记录，建议定期增量训练以适应当前数据分布。</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>增量训练 - 声纹管理系统</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:20px;font-size:14px }}
.nav a {{ color:#bdc3c7;text-decoration:none }}
.nav .user{{margin-left:auto;font-size:13px}}
.nav .active{{color:#fff;font-weight:bold}}
.container{{max-width:1200px;margin:24px auto;padding:0 16px}}
.card{{background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.card h2{{font-size:18px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #eee}}
.summary-bar{{display:flex;gap:16px;margin-bottom:16px}}
.summary-item{{flex:1;text-align:center;padding:16px;border-radius:8px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.summary-num{{font-size:28px;font-weight:bold}}
.summary-label{{font-size:12px;color:#777}}
table{{width:100%;border-collapse:collapse}}
th{{background:#f8f9fa;padding:8px;text-align:left;font-size:12px;border-bottom:2px solid #dee2e6;white-space:nowrap}}
td{{padding:6px 8px;font-size:13px;border-bottom:1px solid #eee}}
.badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px}}
.badge-orange{{background:#fef3cd;color:#856404}}
.badge-green{{background:#d4edda;color:#155724}}
.badge-blue{{background:#cfe2ff;color:#0a58ca}}
.badge-red{{background:#f8d7da;color:#721c24}}
.badge-gray{{background:#f1f5f9;color:#64748b}}
.btn{{padding:7px 16px;border:none;border-radius:5px;cursor:pointer;font-size:13px;text-decoration:none;display:inline-block}}
.btn-primary{{background:#3498db;color:#fff}}
.btn-secondary{{background:#95a5a6;color:#fff}}
.btn-sm{{padding:3px 8px;border:none;border-radius:3px;cursor:pointer;font-size:11px;color:#fff}}
.training-panel{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px;margin-bottom:16px}}
.comp-row{{display:flex;gap:16px;margin-top:12px}}
.comp-card{{flex:1;background:#f8f9fa;border-radius:8px;padding:12px;text-align:center}}
.comp-card h4{{font-size:14px;margin-bottom:8px}}
.comp-score{{font-size:32px;font-weight:bold}}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">增量训练</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="summary-bar">
        <div class="summary-item"><div class="summary-num">{len(versions)}</div><div class="summary-label">训练版本</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#e67e22">{pending}</div><div class="summary-label">待训练录音</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#3498db">{sum(1 for v in versions if v.get("status")=="training")}</div><div class="summary-label">训练中</div></div>
    </div>
    {suggestions}
    <div class="card">
        <h2>训练版本列表</h2>
        <div style="margin-bottom:12px;display:flex;gap:8px">
            <button class="btn btn-primary" onclick="triggerTraining()">📈 启动增量训练</button>
            <button class="btn btn-secondary" onclick="showCompareDialog()">📊 版本对比</button>
        </div>
        <table>
            <thead><tr><th>ID</th><th>模型</th><th>版本</th><th>基座</th><th>维度</th><th>状态</th><th>Score</th><th>指标</th><th>时间</th><th>操作</th></tr></thead>
            <tbody>{version_rows}</tbody>
        </table>
    </div>
    <div id="comparePanel" class="card" style="display:none">
        <h2>版本对比</h2>
        <div style="margin-bottom:12px">
            <label>版本 A:</label> <select id="compA" class="label-select" onchange="updateCompare()"></select>
            <label style="margin-left:16px">版本 B:</label> <select id="compB" class="label-select" onchange="updateCompare()"></select>
        </div>
        <div id="compResult" class="comp-row"></div>
    </div>
</div>
<script>
var versionsData = [];
async function triggerTraining() {{
    if (!confirm("确定启动增量训练？训练周期较长，请确保已标记足够的说话人数据。")) return;
    const btn = event.target; btn.disabled = true; btn.textContent = "训练启动中...";
    try {{
        const resp = await fetch("/model-manager/run-train", {{ method:"POST" }});
        const r = await resp.json();
        alert(r.success ? "训练已启动" : "失败: " + (r.error||JSON.stringify(r)));
        location.reload();
    }} catch(e) {{ alert(e.message); }}
    finally {{ btn.disabled = false; btn.textContent = "启动增量训练"; }}
}}
function showCompareDialog() {{
    document.getElementById("comparePanel").style.display = "block";
    var selA = document.getElementById("compA");
    var selB = document.getElementById("compB");
    selA.innerHTML = ""; selB.innerHTML = "";
    versionsData.forEach(function(v) {{
        var optA = document.createElement("option"); optA.value = v.id; optA.text = v.model_name + " @" + v.version_tag;
        var optB = optA.cloneNode(true);
        selA.appendChild(optA); selB.appendChild(optB);
    }});
    if (versionsData.length >= 2) {{
        selA.value = versionsData[versionsData.length-2].id;
        selB.value = versionsData[versionsData.length-1].id;
    }}
    updateCompare();
}}
async function updateCompare() {{
    var a = document.getElementById("compA").value;
    var b = document.getElementById("compB").value;
    if (!a || !b || a === b) return;
    var resp = await fetch("/model-manager/training/compare/" + a + "/" + b);
    var r = await resp.json();
    if (!r.success) return;
    var vA = r.a, vB = r.b;
    var scoreA = vA.score || 0, scoreB = vB.score || 0;
    var diff = scoreB - scoreA;
    var diffColor = diff >= 0 ? "#27ae60" : "#e74c3c";
    var diffIcon = diff >= 0 ? "▲" : "▼";
    document.getElementById("compResult").innerHTML =
        '<div class="comp-card"><h4><span class="badge badge-blue">A</span> ' + vA.model_name + ' @' + vA.version_tag + '</h4>' +
        '<div class="comp-score">' + scoreA.toFixed(4) + '</div>' +
        '<div style="font-size:12px;color:#666">' + vA.created_at + '</div></div>' +
        '<div class="comp-card" style="display:flex;flex-direction:column;align-items:center;justify-content:center">' +
        '<div style="font-size:14px;color:#666">差异</div>' +
        '<div class="comp-score" style="color:' + diffColor + '">' + diffIcon + ' ' + diff.toFixed(4) + '</div>' +
        '<div style="font-size:12px;color:#666">' + (diff >= 0 ? "新版提升" : "新版下降") + '</div></div>' +
        '<div class="comp-card"><h4><span class="badge badge-green">B</span> ' + vB.model_name + ' @' + vB.version_tag + '</h4>' +
        '<div class="comp-score">' + scoreB.toFixed(4) + '</div>' +
        '<div style="font-size:12px;color:#666">' + vB.created_at + '</div></div>';
}}
document.addEventListener("DOMContentLoaded", function() {{
    fetch("/model-manager/training/versions-data").then(r=>r.json()).then(d=>{{ if(d.success) versionsData = d.versions; }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Training versions data API (for JS dropdown)
# ---------------------------------------------------------------------------

@router.get("/model-manager/training/versions-data")
async def training_versions_data(request: Request):
    await require_role(ROLE_MODEL_MANAGER)(request)
    db = _recordings_db()
    try:
        versions = await db.list_model_versions(limit=50)
        clean = []
        for v in versions:
            clean.append({"id": v["id"], "model_name": v.get("model_name",""),
                          "version_tag": v.get("version_tag",""), "score": v.get("score"),
                          "status": v.get("status",""), "created_at": v.get("created_at","")})
        return {"success": True, "versions": clean}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/model-manager/training/compare/{a_id}/{b_id}")
async def compare_versions(request: Request, a_id: int, b_id: int):
    await require_role(ROLE_MODEL_MANAGER)(request)
    db = _recordings_db()
    try:
        versions = await db.get_version_diff(a_id, b_id)
        if len(versions) < 2:
            return {"success": False, "error": "版本不存在"}
        v_map = {v["id"]: v for v in versions}
        for v in v_map.values():
            for field in ("metrics", "config"):
                if isinstance(v.get(field), str):
                    try:
                        import json as j
                        v[field] = j.loads(v[field])
                    except Exception:
                        v[field] = {}
        return {"success": True, "a": v_map.get(a_id), "b": v_map.get(b_id)}
    except Exception as e:
        return {"success": False, "error": str(e)}
# ---------------------------------------------------------------------------
# l18: 模型详情页面
# ---------------------------------------------------------------------------

@router.get("/model-manager/models", response_class=HTMLResponse)
async def model_detail_page(request: Request):
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user

    # ── 直接从 SQLite 读取 model_definitions + checkpoints（含 DAG lineage）──
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(DB_PATH))
    conn.row_factory = _sqlite3.Row

    # 1) 模型架构定义
    def_rows = conn.execute(
        "SELECT id, name, arch_version, code_path, code_hash, class_name, "
        "embedding_dim, description, created_at, created_by "
        "FROM model_definitions ORDER BY name"
    ).fetchall()
    model_defs = [dict(r) for r in def_rows]

    # 2) checkpoints（按 model_name 分组，带 base/audit 字段）
    ck_rows = conn.execute(
        "SELECT c.id, c.model_name, c.version_tag, c.file_path, c.file_size, "
        "c.embedding_dim, c.metrics, c.is_published, c.model_def_id, "
        "c.base_checkpoint_id, c.status, c.trained_at, c.created_at, c.created_by "
        "FROM checkpoints c ORDER BY c.model_name, c.id"
    ).fetchall()
    all_ckpts = [dict(r) for r in ck_rows]

    # 3) 每个 checkpoint 的训练数据片段数
    seg_counts = {}
    for r in conn.execute(
        "SELECT checkpoint_id, COUNT(*) AS cnt FROM checkpoint_training_segments GROUP BY checkpoint_id"
    ):
        seg_counts[r["checkpoint_id"]] = r["cnt"]

    conn.close()

    return HTMLResponse(render_model_detail_page(
        user, model_defs, all_ckpts, seg_counts,
        fs_root=APP_DIR / "model_data" / "checkpoints",
    ))


def render_model_detail_page(user, model_defs, all_ckpts, seg_counts, fs_root=None):
    """渲染模型详情页。

    Args:
        user: 当前用户 dict
        model_defs: model_definitions 行列表
        all_ckpts: checkpoints 行列表（含 model_def_id / base_checkpoint_id / status）
        seg_counts: {checkpoint_id: 训练片段数}
        fs_root: 文件系统快照根目录（用于检测 DB 外的快照）
    """
    from collections import defaultdict
    user = user or {}
    fs_root = fs_root or Path()

    # 按模型名分组 checkpoint
    ck_by_name = defaultdict(list)
    for c in all_ckpts:
        ck_by_name[c.get("model_name", "")].append(c)
    # 建立 checkpoint_id → version_tag 映射（用于显示 base lineage）
    id_to_tag = {c["id"]: c.get("version_tag", "?") for c in all_ckpts}

    # 模型架构信息（网络结构描述，补充展示用）
    ARCH_INFO = {
        "CAM++": {"layers": "前端(conv1d/BN/ReLU) → DenseRes2Net → ASP → FC256/192", "params": "~7.2M"},
        "ECAPA": {"layers": "TDNN front-end → SE-Res2Block ×3 → ASP+ChannelAttn → FC192", "params": "~6.5M"},
        "ResNet34": {"layers": "Conv1x3x3 → 4×[3x3 ResBlock]×{3,4,6,3} → GAP → FC512/256", "params": "~21.8M"},
    }

    cards = ""
    for mdef in model_defs:
        mname = mdef["name"]
        dim_val = mdef.get("embedding_dim", 0)
        arch = ARCH_INFO.get(mname, {"layers": "—", "params": "—"})
        cks = sorted(ck_by_name.get(mname, []), key=lambda x: x.get("id", 0))

        # 预训练 / 增量分离
        pretrained = [c for c in cks if c.get("status") == "pretrained"]
        incremental = [c for c in cks if c.get("status") == "incremental"]

        # checkpoint 表格（含 DAG lineage + 训练数据来源）
        ck_rows_html = ""
        for c in cks:
            vtag = c.get("version_tag", "").replace(f"{mname}@", "")
            st = c.get("status", "")
            st_badge = {"pretrained": "blue", "incremental": "green",
                        "published": "orange", "archived": "gray"}.get(st, "gray")
            st_label = {"pretrained": "预训练", "incremental": "增量",
                        "published": "已发布", "archived": "归档"}.get(st, st)
            base_id = c.get("base_checkpoint_id")
            base_label = f"← {id_to_tag.get(base_id, '?')}" if base_id else "—"
            seg_n = seg_counts.get(c["id"], 0)
            seg_label = f"{seg_n} 段" if seg_n else "—"
            ts = (c.get("trained_at") or c.get("created_at") or "")[:16]
            ck_rows_html += (
                f'<tr>'
                f'<td>{vtag}</td>'
                f'<td><span class="badge badge-{st_badge}">{st_label}</span></td>'
                f'<td style="font-size:11px;color:#64748b">{base_label}</td>'
                f'<td>{seg_label}</td>'
                f'<td style="font-size:11px">{ts}</td>'
                f'</tr>'
            )
        ck_table = (
            f'<table style="margin-top:12px"><thead><tr>'
            f'<th>版本</th><th>状态</th><th>基础来源</th><th>训练数据</th><th>时间</th>'
            f'</tr></thead><tbody>{ck_rows_html}</tbody></table>'
            if ck_rows_html else '<p style="color:#999;font-size:13px">暂无 checkpoint</p>'
        )

        # 文件系统快照检测
        fs_dir = fs_root / mname
        fs_dirs = [d.name for d in fs_dir.iterdir() if d.is_dir()] if fs_dir.exists() else []

        cards += f"""<div class="card">
            <h2>{mname}
                <span class="badge badge-blue">{dim_val}d</span>
                <span class="badge badge-gray">架构 {mdef.get('arch_version','v1')}</span>
            </h2>
            <p style="font-size:13px;color:#666;margin-bottom:12px">{mdef.get('description','')}</p>
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Embedding维度</div><div class="info-value">{dim_val}</div></div>
                <div class="info-item"><div class="info-label">参数量</div><div class="info-value">{arch["params"]}</div></div>
                <div class="info-item"><div class="info-label">预训练快照</div><div class="info-value">{len(pretrained)}</div></div>
                <div class="info-item"><div class="info-label">增量快照</div><div class="info-value">{len(incremental)}</div></div>
                <div class="info-item"><div class="info-label">PyTorch类</div><div class="info-value" style="font-size:12px">{mdef.get('class_name','')}</div></div>
                <div class="info-item"><div class="info-label">代码Hash</div><div class="info-value" style="font-size:11px;font-family:monospace">{mdef.get('code_hash','') or '—'}</div></div>
                <div class="info-item" style="grid-column:1/-1"><div class="info-label">代码位置</div><div class="info-value" style="font-size:11px;font-family:monospace">{mdef.get('code_path','')}</div></div>
                <div class="info-item" style="grid-column:1/-1"><div class="info-label">网络结构</div><div class="info-value" style="font-size:11px">{arch["layers"]}</div></div>
                <div class="info-item" style="grid-column:1/-1"><div class="info-label">快照目录</div><div class="info-value" style="font-size:11px;font-family:monospace">app/model_data/checkpoints/{mname}/</div></div>
            </div>
            {ck_table}
            {f'<p style="font-size:11px;color:#999;margin-top:8px">文件系统快照: {", ".join(fs_dirs)}</p>' if fs_dirs else ''}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型详情 - 声纹管理系统</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:20px;font-size:14px }}
.nav a {{ color:#bdc3c7;text-decoration:none }}
.nav .user{{margin-left:auto;font-size:13px}}
.nav .active{{color:#fff;font-weight:bold}}
.container{{max-width:1000px;margin:24px auto;padding:0 16px}}
.card{{background:#fff;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.card h2{{font-size:18px;margin-bottom:8px}}
.info-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:8px}}
.info-item{{padding:8px;background:#f8f9fa;border-radius:6px}}
.info-label{{font-size:11px;color:#888}}
.info-value{{font-size:14px;font-weight:bold}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f8f9fa;padding:6px;text-align:left;font-size:12px;border-bottom:2px solid #dee2e6}}
td{{padding:6px;border-bottom:1px solid #eee}}
.badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px}}
.badge-blue{{background:#cfe2ff;color:#0a58ca}}
.badge-green{{background:#d4edda;color:#155724}}
.badge-orange{{background:#fef3cd;color:#856404}}
.badge-gray{{background:#f1f5f9;color:#64748b}}
.badge-red{{background:#f8d7da;color:#721c24}}
.legend{{font-size:11px;color:#999;margin-bottom:12px;padding:8px;background:#fffbe6;border-radius:6px;border:1px solid #ffe58f}}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">模型管理</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="legend">
        ℹ️ 共 {len(model_defs)} 个模型定义。"基础来源"列显示该增量 checkpoint 由哪个 checkpoint 继续训练（DAG 演化链）。
        "训练数据"列显示该 checkpoint 用了多少语音片段做增量训练。架构变化（结构性）会创建新的 arch_version 分支。
    </div>
    {cards}
</div>
</body>
</html>"""
# ---------------------------------------------------------------------------
# l20: 模型发布页面
# ---------------------------------------------------------------------------

@router.get("/model-manager/publish", response_class=HTMLResponse)
async def publish_page(request: Request):
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user
    db = _recordings_db()
    checkpoints = []
    versions = []
    try:
        checkpoints = await db.list_checkpoints(limit=100)
        versions = await db.list_model_versions(limit=100)
    except Exception as e:
        logger.error("publish page: %s", e)
    return HTMLResponse(render_publish_page(user, checkpoints, versions))


def render_publish_page(user, checkpoints, versions):
    user = user or {}
    from collections import defaultdict
    model_cps = defaultdict(list)
    for cp in checkpoints:
        model_cps[cp.get("model_name","unknown")].append(cp)
    published_cps = {}
    for mname, cps in model_cps.items():
        for cp in cps:
            if cp.get("is_published"):
                published_cps[mname] = cp

    sections = ""
    for mname in ["CAM++", "ResNet34", "ECAPA"]:
        cps = model_cps.get(mname, [])
        pub = published_cps.get(mname)
        vfm = [v for v in versions if v.get("model_name") == mname]
        rows = ""
        for cp in cps[::-1]:
            is_pub = cp.get("is_published")
            row_cls = ' class="pub-row"' if is_pub else ""
            badge = '<span class="badge badge-green">✅ 已发布</span>' if is_pub else ""
            pub_btn = "" if is_pub else f'<button class="btn-sm" onclick="publishCheckpoint({cp["id"]})">发布</button>'
            rows += f"<tr{row_cls}>"
            rows += f'<td>{cp.get("version_tag","")}</td>'
            rows += f'<td>{cp.get("embedding_dim","")}d</td>'
            rows += f'<td style="font-size:11px;font-family:monospace">{cp.get("file_path","")[:50]}...</td>'
            rows += f'<td>{cp.get("created_at","")[:16]}</td>'
            rows += f"<td>{badge}</td>"
            rows += f"<td>{pub_btn}</td>"
            rows += "</tr>"
        if not rows:
            rows = "<tr><td colspan='6' style='text-align:center;color:#999'>暂无 checkpoint</td></tr>"
        vt = ""
        for v in vfm[::-1][:5]:
            sc = f"{v.get('score',''):.4f}" if v.get('score') else "—"
            vt += "<tr>"
            vt += f'<td>{v.get("version_tag","")}</td>'
            vt += f'<td>{v.get("status","")}</td>'
            vt += f'<td>{sc}</td>'
            vt += f'<td>{v.get("created_at","")[:16]}</td>'
            vt += '</tr>'
        sections += f"""<div class="card">
            <h2>{mname}</h2>
            <div style="display:flex;gap:24px;margin-bottom:16px">
                <div><span class="badge badge-blue">已发布:</span> <strong>{pub.get("version_tag","—") if pub else "未发布"}</strong></div>
                <div><span class="badge badge-orange">Checkpoints:</span> <strong>{len(cps)}</strong></div>
                <div><span class="badge badge-green">训练版本:</span> <strong>{len(vfm)}</strong></div>
            </div>
            <h3 style="font-size:15px;margin-bottom:8px">Checkpoints</h3>
            <table><thead><tr><th>版本</th><th>维度</th><th>路径</th><th>时间</th><th>状态</th><th>操作</th></tr></thead><tbody>{rows}</tbody></table>
            {f'<h3 style="font-size:15px;margin-top:16px;margin-bottom:8px">训练版本记录</h3><table><thead><tr><th>版本</th><th>状态</th><th>评分</th><th>时间</th></tr></thead><tbody>{vt}</tbody></table>' if vt else ""}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型发布 - 声纹管理系统</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:20px;font-size:14px }}
.nav a {{ color:#bdc3c7;text-decoration:none }}
.nav .user{{margin-left:auto;font-size:13px}}
.nav .active{{color:#fff;font-weight:bold}}
.container{{max-width:1000px;margin:24px auto;padding:0 16px}}
.card{{background:#fff;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.card h2{{font-size:18px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #eee}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#f8f9fa;padding:6px;text-align:left;font-size:12px;border-bottom:2px solid #dee2e6}}
td{{padding:6px;border-bottom:1px solid #eee}}
tr.pub-row{{background:#d4edda}}
.badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px}}
.badge-blue{{background:#cfe2ff;color:#0a58ca}}
.badge-green{{background:#d4edda;color:#155724}}
.badge-orange{{background:#fef3cd;color:#856404}}
.btn{{padding:7px 16px;border:none;border-radius:5px;cursor:pointer;font-size:13px;text-decoration:none;display:inline-block}}
.btn-primary{{background:#3498db;color:#fff}}
.btn-sm{{padding:3px 8px;border:none;border-radius:3px;cursor:pointer;font-size:11px;color:#fff;background:#3498db}}
.auto-switch{{display:flex;align-items:center;gap:12px;padding:12px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;margin-bottom:16px}}
</style></head>
<body>
<div class="nav">
    <strong>声纹管理系统</strong>
    <a href="/model-manager">← 首页</a>
    <span style="color:#fff;font-weight:bold">模型发布</span>
    <span class="user">{user.get('username','')}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    {sections}
</div>
<script>
async function publishCheckpoint(cpId) {{
    if (!confirm("确定发布此 checkpoint？现有已发布版本将被取消。")) return;
    try {{
        const resp = await fetch("/model-manager/publish/" + cpId, {{ method:"POST" }});
        const r = await resp.json();
        if (r.success) {{ alert("已发布"); location.reload(); }}
        else alert("发布失败: " + (r.error||""));
    }} catch(e) {{ alert(e.message); }}
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Publish checkpoint API
# ---------------------------------------------------------------------------

@router.post("/model-manager/publish/{checkpoint_id}")
async def publish_checkpoint(request: Request, checkpoint_id: int):
    await require_role(ROLE_MODEL_MANAGER)(request)
    db = _recordings_db()
    try:
        ok = await db.set_published_checkpoint(checkpoint_id)
        return {"success": ok}
    except Exception as e:
        return {"success": False, "error": str(e)}
