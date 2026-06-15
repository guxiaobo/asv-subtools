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

from services.auth import get_current_user, is_logged_in, require_role

ROLE_MODEL_MANAGER = "model_manager"

router = APIRouter()
logger = logging.getLogger("model_manager_router")

# Project root (two levels up from this router file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DATA_DIR = PROJECT_ROOT / "data"
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
<title>VAD 参数配置 - ASV 声纹识别系统</title>
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
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">模型管理</a>
    <a href="/model-manager/segments">录音断句</a>
    <a href="/model-manager/vad-config" style="color:#fff;font-weight:bold">VAD 参数</a>
    <span class="user">{user.get('username','')} (模型管理员)</span>
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
                <label>最小片段时长 (秒)</label>
                <input type="number" name="min_segment_sec" value="{cfg['min_segment_sec']}" min="0.5" max="60" step="0.1">
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
        else if (['vad_threshold','min_segment_sec','max_segment_sec','snr_threshold','filter_leading_sec','target_sample_rate','window_ms','diarizer_cluster_threshold'].includes(k))
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
    customer_phone: str = "",
    date_from: str = "",
    date_to: str = "",
    status: str = "",
):
    """录音断句管理页面。"""
    await require_role(ROLE_MODEL_MANAGER)(request)
    user = request.state.current_user if hasattr(request.state, 'current_user') else None

    # Load recording data
    db = _recordings_db()
    vad_cfg = _load_vad_config()

    recordings = []
    pending_count = 0
    try:
        recordings = await db.list_recordings_with_segments(
            agent_id=agent_id,
            customer_phone=customer_phone,
            date_from=date_from,
            date_to=date_to,
            status_filter=status,
            limit=200,
        )
        pending_count = await db.count_pending_segment_recordings()
    except Exception as e:
        logger.error("Failed to load recordings: %s", e)

    # Get agent list for filter dropdown
    agents = []
    try:
        agents = await db.get_users_by_role("agent")
    except Exception:
        pass

    page = _render_segment_page(user, recordings, pending_count, agents, vad_cfg,
                                agent_id, customer_phone, date_from, date_to, status)
    return HTMLResponse(page)


def _render_segment_page(user, recordings, pending_count, agents, vad_cfg,
                         agent_id, customer_phone, date_from, date_to, status):
    """渲染录音断句页面 HTML。"""
    user = user or {}
    rows_html = ""
    for rec in recordings:
        seg_count = rec.get("seg_count", 0)
        seg_ignored = rec.get("seg_ignored", 0)
        pre_status = rec.get("pre_status", "pending")
        batch = rec.get("latest_batch", "v1")

        status_color = {"pending": "#f39c12", "processing": "#3498db",
                        "done": "#27ae60", "failed": "#e74c3c"}.get(pre_status, "#95a5a6")

        seg_info = f"{seg_count} 段"
        if seg_ignored > 0:
            seg_info += f' <span style="color:#e74c3c">({seg_ignored} 忽略)</span>'

        rows_html += f"""<tr>
            <td>{rec.get('id','')}</td>
            <td>{rec.get('agent_id','')}</td>
            <td>{rec.get('customer_phone','')}</td>
            <td>{rec.get('call_timestamp','')[:16]}</td>
            <td>{rec.get('call_id','')}</td>
            <td><span style="color:{status_color};font-weight:bold">{pre_status}</span></td>
            <td>{seg_info}</td>
            <td>
                <a href="/model-manager/segments/{rec['id']}?batch={batch}" class="btn-sm" style="background:#3498db">查看片段</a>
                {"<span class='btn-sm' style='background:#95a5a6;cursor:default'>请先 VAD 预处理</span>" if pre_status != 'done' else ""}
            </td>
        </tr>"""

    if not recordings:
        rows_html = "<tr><td colspan='8' style='text-align:center;color:#999;padding:30px'>暂无录音数据</td></tr>"

    # Agent options
    agent_opts = '<option value="">全部坐席</option>'
    for a in agents:
        sel = 'selected' if a.get('agent_id','') == agent_id else ''
        agent_opts += f'<option value="{a["agent_id"]}" {sel}>{a.get("display_name","") or a["agent_id"]}</option>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>录音断句 - ASV 声纹识别系统</title>
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
.btn-danger {{ background:#e74c3c; color:#fff }}
.btn-success {{ background:#27ae60; color:#fff }}
.btn-sm {{ padding:4px 10px; border:none; border-radius:4px; cursor:pointer; font-size:12px; color:#fff; text-decoration:none; display:inline-block }}
.btn-disabled {{ background:#95a5a6; cursor:default; padding:4px 10px; border-radius:4px; font-size:12px; color:#fff; display:inline-block }}
table {{ width:100%; border-collapse:collapse }}
th {{ background:#f8f9fa; padding:10px 8px; text-align:left; font-size:13px; font-weight:600; border-bottom:2px solid #dee2e6 }}
td {{ padding:8px; font-size:13px; border-bottom:1px solid #eee }}
tr:hover {{ background:#f8f9fa }}
.msg {{ padding:10px 16px; border-radius:6px; margin-bottom:16px; font-size:14px }}
.msg.success {{ background:#d4edda; color:#155724; border:1px solid #c3e6cb }}
.msg.error {{ background:#f8d7da; color:#721c24; border:1px solid #f5c6cb }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold }}
.badge-orange {{ background:#fef3cd; color:#856404 }}
.badge-blue {{ background:#cfe2ff; color:#0a58ca }}
.badge-green {{ background:#d4edda; color:#155724 }}
.badge-red {{ background:#f8d7da; color:#721c24 }}
.summary-row {{ display:flex; gap:20px; margin-bottom:16px }}
.summary-item {{ flex:1; text-align:center; padding:16px; border-radius:8px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.1) }}
.summary-num {{ font-size:28px; font-weight:bold; color:#2c3e50 }}
.summary-label {{ font-size:12px; color:#777; margin-top:4px }}
</style></head>
<body>
<div class="nav">
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">模型管理</a>
    <a href="/model-manager/segments" class="nav-active">录音断句</a>
    <a href="/model-manager/vad-config">VAD 参数</a>
    <span class="user">{user.get('username','')} (模型管理员)</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="summary-row">
        <div class="summary-item">
            <div class="summary-num">{len(recordings)}</div>
            <div class="summary-label">总录音数</div>
        </div>
        <div class="summary-item">
            <div class="summary-num">{pending_count}</div>
            <div class="summary-label">待断句</div>
        </div>
        <div class="summary-item">
            <div class="summary-num">{sum(1 for r in recordings if r.get('seg_count',0)>0)}</div>
            <div class="summary-label">已断句</div>
        </div>
        <div class="summary-item">
            <div class="summary-num">{sum(1 for r in recordings if r.get('seg_count',0)>0 and r.get('seg_ignored',0)==0)}</div>
            <div class="summary-label">已打标</div>
        </div>
    </div>

    <div class="card">
        <h2>录音断句</h2>
        <form method="get" action="/model-manager/segments" class="filter-row">
            <div>
                <label style="display:block">坐席</label>
                <select name="agent_id">{agent_opts}</select>
            </div>
            <div>
                <label style="display:block">客户</label>
                <input type="text" name="customer_phone" value="{customer_phone}" placeholder="电话号码">
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
                    <option value="pending_segment" {"selected" if status=='pending_segment' else ""}>待断句</option>
                    <option value="segmented" {"selected" if status=='segmented' else ""}>已断句</option>
                    <option value="done" {"selected" if status=='done' else ""}>预处理完成</option>
                    <option value="pending" {"selected" if status=='pending' else ""}>待处理</option>
                    <option value="failed" {"selected" if status=='failed' else ""}>失败</option>
                </select>
            </div>
            <div>
                <button type="submit" class="btn btn-primary">筛选</button>
                <a href="/model-manager/segments" class="btn btn-primary" style="background:#95a5a6">重置</a>
            </div>
        </form>

        <div style="margin-bottom:12px;display:flex;gap:8px">
            <button class="btn btn-success" onclick="runVad()">开始断句（处理待处理录音）</button>
        </div>

        <table>
            <thead>
                <tr><th>ID</th><th>坐席</th><th>客户</th><th>时间</th><th>文件名</th><th>VAD状态</th><th>断句</th><th>操作</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
</div>
<script>
async function runVad() {{
    if (!confirm('确定要对所有待 VAD 预处理的录音执行断句吗？')) return;
    const btn = event.target; btn.disabled = true; btn.textContent = '处理中...';
    try {{
        const resp = await fetch('/model-manager/run-preprocess', {{ method:'POST' }});
        const result = await resp.json();
        if (result.success) {{
            alert('断句任务已启动: ' + result.message);
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
        # Query recording directly by ID
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
        batches = await db.get_batches_for_recording(recording_id)
    except Exception as e:
        logger.error("Failed to load segments for recording %d: %s", recording_id, e)

    if not recording:
        return HTMLResponse("<h1>录音不存在</h1><a href='/model-manager/segments'>返回</a>", status_code=404)

    return HTMLResponse(_render_recording_detail(user, recording, segments, batches, batch, recording_id))


@router.get("/model-manager/evaluate", response_class=HTMLResponse)
async def evaluate_redirect(request: Request):
    """模型评估页面 — 重定向到首页（评估为 POST 内联操作）。"""
    return RedirectResponse(url="/model-manager", status_code=302)


def _render_recording_detail(user, recording, segments, batches, current_batch, recording_id):
    """渲染录音详情页（含片段列表）。"""
    user = user or {}
    seg_rows = ""
    for seg in segments:
        ignore_link = f"/model-manager/segments/{seg['id']}/toggle-ignore"
        seg_class = "seg-ignored" if seg.get("is_ignored") else ""
        ignore_btn = "取消忽略" if seg.get("is_ignored") else "忽略"
        ignore_btn_class = "btn-sm" if not seg.get("is_ignored") else "btn-sm btn-sm-warning"

        speaker_type = seg.get("speaker_type", "unknown")
        speaker_type_label = {"agent": "坐席", "customer": "客户", "unknown": "未知", "ignored": "已忽略"}.get(speaker_type, speaker_type)

        seg_rows += f"""<tr class="{seg_class}">
            <td>{seg.get('segment_index','')}</td>
            <td>{seg.get('start_sec',''):.1f}s</td>
            <td>{seg.get('end_sec',''):.1f}s</td>
            <td>{seg.get('duration_sec',''):.1f}s</td>
            <td>
                <button class="btn-sm" style="background:#2c3e50" onclick="playAudio('{seg['id']}')">▶ 播放</button>
            </td>
            <td>
                <span id="label-{seg['id']}">{seg.get('speaker_label','') or speaker_type_label}</span>
            </td>
            <td>
                <a href="javascript:void(0)" onclick="toggleIgnore({seg['id']})" id="ignore-{seg['id']}" class="{ignore_btn_class}">{ignore_btn}</a>
            </td>
            <td>
                <input type="text" id="custom-label-{seg['id']}" placeholder="自定义标签" style="width:100px;padding:3px 6px;border:1px solid #ddd;border-radius:3px;font-size:12px">
                <button class="btn-sm" style="background:#8e44ad" onclick="setLabel({seg['id']})">设置</button>
            </td>
        </tr>"""

    if not segments:
        seg_rows = "<tr><td colspan='8' style='text-align:center;color:#999;padding:30px'>暂无断句片段，请先执行 VAD 断句</td></tr>"

    # Batch selector
    batch_opts = ""
    for b in batches:
        sel = "selected" if b == current_batch else ""
        batch_opts += f'<option value="{b}" {sel}>第 {b.upper()} 次</option>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>录音详情 - {recording.get('call_id','')} - ASV</title>
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
.btn-sm {{ padding:4px 10px; border:none; border-radius:4px; cursor:pointer; font-size:12px; color:#fff; display:inline-block }}
.btn-sm-warning {{ background:#e67e22 !important }}
audio {{ width:100%; margin-top:8px }}
.batch-select {{ padding:6px 10px; border:1px solid #ddd; border-radius:5px; font-size:13px; margin-right:8px }}
</style></head>
<body>
<div class="nav">
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">模型管理</a>
    <a href="/model-manager/segments" style="color:#fff;font-weight:bold">录音断句</a>
    <a href="/model-manager/vad-config">VAD 参数</a>
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
            <div class="info-item"><div class="info-label">客户</div><div class="info-value">{recording.get('customer_phone','')}</div></div>
            <div class="info-item"><div class="info-label">文件名</div><div class="info-value">{recording.get('call_id','')}</div></div>
            <div class="info-item"><div class="info-label">时间</div><div class="info-value">{recording.get('call_timestamp','')}</div></div>
            <div class="info-item"><div class="info-label">时长</div><div class="info-value">{(recording.get('duration_sec') or 0):.1f}s</div></div>
        </div>
        <div style="margin-top:12px">
            <label>断句版本:</label>
            <select class="batch-select" onchange="switchBatch(this.value)">
                {batch_opts}
                <option value="__new__">+ 重新断句</option>
            </select>
            <button class="btn btn-primary" onclick="reprocessRecording()">重新断句</button>
        </div>
    </div>

    <div class="card">
        <h2>断句片段 ({len(segments)} 段)</h2>
        <div id="audioPlayer" style="display:none;margin-bottom:12px">
            <audio id="player" controls></audio>
        </div>
        <table>
            <thead>
                <tr><th>序号</th><th>起始</th><th>结束</th><th>时长</th><th>播放</th><th>说话人</th><th>操作</th><th>自定义标签</th></tr>
            </thead>
            <tbody>{seg_rows}</tbody>
        </table>
    </div>
</div>
<script>
const audioPlayer = document.getElementById('audioPlayer');
const player = document.getElementById('player');

function playAudio(segId) {{
    audioPlayer.style.display = 'block';
    player.src = '/model-manager/segments/audio/' + segId;
    player.play();
}}

async function toggleIgnore(segId) {{
    try {{
        const resp = await fetch('/model-manager/segments/' + segId + '/toggle-ignore', {{ method:'POST' }});
        const result = await resp.json();
        if (result.success) location.reload();
        else alert('操作失败');
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
}}

async function setLabel(segId) {{
    const input = document.getElementById('custom-label-' + segId);
    const label = input.value.trim();
    if (!label) return;
    try {{
        const resp = await fetch('/model-manager/segments/' + segId + '/label', {{ 
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{speaker_label: label, label_source: 'manual'}})
        }});
        const result = await resp.json();
        if (result.success) {{
            document.getElementById('label-' + segId).textContent = label;
            input.value = '';
        }} else alert('设置失败');
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
}}

function switchBatch(batch) {{
    if (batch === '__new__') return;
    location.href = '/model-manager/segments/{recording_id}?batch=' + batch;
}}

async function reprocessRecording() {{
    if (!confirm('确定要重新断句此录音吗？旧结果会被保留在新版本中。')) return;
    try {{
        const resp = await fetch('/model-manager/run-preprocess', {{ 
            method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{recording_id: {recording.get('id','')}}})
        }});
        const result = await resp.json();
        if (result.success) {{ alert('重新断句已启动'); location.reload(); }}
        else alert('启动失败: ' + result.error);
    }} catch(e) {{ alert('网络错误: ' + e.message); }}
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
    """手动设置片段标签。"""
    db = _recordings_db()
    try:
        data = await request.json()
        speaker_label = data.get("speaker_label", "")
        label_source = data.get("label_source", "manual")
        speaker_type = data.get("speaker_type", "")
        await db.update_segment_label(
            seg_id,
            speaker_label=speaker_label,
            speaker_type=speaker_type,
            label_source=label_source,
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
# ---------------------------------------------------------------------------
# l14: 说话人打标页面
# ---------------------------------------------------------------------------

@router.get("/model-manager/label", response_class=HTMLResponse)
async def label_speakers_page(
    request: Request,
    speaker_type: str = "",
    label_status: str = "",
    segment_limit: int = 200,
):
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
    for rec in recordings:
        rec_id = rec.get("id")
        try:
            segs = await db.get_segments_by_recording(rec_id)
            for s in segs:
                if speaker_type and s.get("speaker_type") != speaker_type:
                    continue
                all_segments.append({**s, "recording_info": rec})
                if not s.get("speaker_label") and not s.get("is_ignored"):
                    unlabeled_count += 1
        except Exception:
            pass

    rows = ""
    for seg in all_segments:
        rec = seg.get("recording_info", {})
        color = {"agent": "#3498db", "customer": "#27ae60", "ignored": "#95a5a6"}.get(
            seg.get("speaker_type", ""), "#95a5a6"
        )
        lbl = seg.get("speaker_label") or "—"
        rows += (
            "<tr>"
            f'<td>{rec.get("id","")}</td>'
            f'<td>{rec.get("agent_id","")}</td>'
            f'<td>{seg.get("segment_index","")}</td>'
            f'<td>{seg.get("duration_sec",0):.1f}s</td>'
            f'<td><span class="badge" style="background:{color};color:#fff">{seg.get("speaker_type","")}</span></td>'
            f'<td><span id="lbl-{seg["id"]}">{lbl}</span></td>'
            f'<td><button class="btn-sm" style="background:#2c3e50" onclick="playAudio({seg["id"]})">▶</button></td>'
            '<td><select id="sel-{sid}" class="label-select" onchange="setLabel({sid})">'
            '<option value="">选择</option>'
            '<option value="agent"{sa}>坐席</option>'
            '<option value="customer"{sc}>客户</option>'
            '<option value="noise"{sn}>噪声</option>'
            "</select></td>"
            '<td>'
            '<input type="text" id="custom-{sid}" placeholder="自定义" style="width:80px;padding:2px 4px;border:1px solid #ddd;border-radius:3px;font-size:11px">'
            '<button class="btn-sm" style="background:#8e44ad" onclick="setCustomLabel({sid})">✓</button>'
            "</td></tr>"
        )

    if not all_segments:
        rows = "<tr><td colspan='9' style='text-align:center;color:#999;padding:30px'>暂无待打标片段</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>说话人打标 - ASV 声纹识别系统</title>
<style>
* {{ margin:0;padding:0;box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f6fa; color:#2c3e50 }}
.nav {{ background:#2c3e50;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:20px;font-size:14px }}
.nav a {{ color:#bdc3c7;text-decoration:none }}
.nav .user{{margin-left:auto;font-size:13px}}
.container{{max-width:1100px;margin:24px auto;padding:0 16px}}
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
.label-select{{padding:3px 4px;border:1px solid #ddd;border-radius:3px;font-size:11px}}
#audioPlayer{{display:none;margin-bottom:12px}}
audio{{width:100%}}
</style></head>
<body>
<div class="nav">
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">首页</a>
    <a href="/model-manager/segments">录音断句</a>
    <a href="/model-manager/label" class="active">说话人打标</a>
    <a href="/model-manager/vad-config">VAD 参数</a>
    <a href="/model-manager/training">增量训练</a>
    <span class="user">{user.get("username","")}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
    <div class="summary-bar">
        <div class="summary-item"><div class="summary-num">{len(all_segments)}</div><div class="summary-label">总片段</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#e67e22">{unlabeled_count}</div><div class="summary-label">待打标</div></div>
        <div class="summary-item"><div class="summary-num" style="color:#27ae60">{len(all_segments)-unlabeled_count}</div><div class="summary-label">已打标</div></div>
    </div>
    <div class="card">
        <h2>说话人打标</h2>
        <div style="margin-bottom:12px;display:flex;gap:8px">
            <button class="btn btn-primary" onclick="autoLabel()">📌 系统自动打标</button>
        </div>
        <div id="audioPlayer"><audio id="player" controls></audio></div>
        <table>
            <thead><tr><th>录音ID</th><th>坐席</th><th>片段</th><th>时长</th><th>类型</th><th>标签</th><th>播放</th><th>快速标记</th><th>自定义</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
</div>
<script>
function playAudio(segId) {{
    document.getElementById("audioPlayer").style.display = "block";
    document.getElementById("player").src = "/model-manager/segments/audio/" + segId;
    document.getElementById("player").play();
}}
async function setLabel(segId) {{
    const sel = document.getElementById("sel-" + segId);
    const label = sel.value;
    if (!label) return;
    const resp = await fetch("/model-manager/segments/" + segId + "/label", {{
        method: "POST", headers:{{"Content-Type":"application/json"}},
        body: JSON.stringify({{speaker_label: label, label_source: "manual"}})
    }});
    const r = await resp.json();
    if (r.success) document.getElementById("lbl-" + segId).textContent = label;
}}
async function setCustomLabel(segId) {{
    const input = document.getElementById("custom-" + segId);
    const label = input.value.trim();
    if (!label) return;
    const resp = await fetch("/model-manager/segments/" + segId + "/label", {{
        method: "POST", headers:{{"Content-Type":"application/json"}},
        body: JSON.stringify({{speaker_label: label, label_source: "manual"}})
    }});
    const r = await resp.json();
    if (r.success) {{ document.getElementById("lbl-" + segId).textContent = label; input.value = ""; }}
}}
async function autoLabel() {{
    if (!confirm("确定执行系统自动说话人打标？")) return;
    const resp = await fetch("/model-manager/run-label", {{ method:"POST" }});
    const r = await resp.json();
    alert(r.success ? "自动打标完成" : "失败: " + (r.error||JSON.stringify(r)));
    location.reload();
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
<title>增量训练 - ASV</title>
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
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">首页</a>
    <a href="/model-manager/segments">录音断句</a>
    <a href="/model-manager/label">说话人打标</a>
    <a href="/model-manager/training" class="active">增量训练</a>
    <span class="user">{user.get("username","")}</span>
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
    db = _recordings_db()
    versions = []
    try:
        versions = await db.list_model_versions(limit=100)
    except Exception:
        pass
    from collections import defaultdict
    grouped = defaultdict(list)
    for v in versions:
        grouped[v.get("model_name","unknown")].append(v)
    published = {}
    for name in grouped:
        pv = [v for v in grouped[name] if v.get("status") == "published"]
        published[name] = pv[0] if pv else None
    return HTMLResponse(render_model_detail_page(user, grouped, published, versions))


def render_model_detail_page(user, grouped, published, all_versions):
    user = user or {}
    model_infos = {
        "CAM++": {
            "dim": 192, "desc": "CAM++ (Concat-Aggregated MFCC Plus Plus)，电话场景分离最佳",
            "path": "api/models/CAM++_cnceleb",
            "layers": "CAM++ 前端(conv1d/BN/ReLU) → DenseRes2Net → ASP → FC256/192",
            "params": "~7.2M"
        },
        "ResNet34": {
            "dim": 256, "desc": "ResNet34 残差网络，256维embedding，通用性强",
            "path": "api/models/ResNet34_cnceleb",
            "layers": "Conv1x3x3 → 4×[3x3 ResBlock]×{3,4,6,3} → GAP → FC512/256",
            "params": "~21.8M"
        },
        "ECAPA": {
            "dim": 192, "desc": "ECAPA-TDNN，时延神经网络+通道注意力",
            "path": "api/models/ECAPA_cnceleb",
            "layers": "TDNN front-end → SE-Res2Block ×3 → ASP+ChannelAttn → FC192",
            "params": "~6.5M"
        }
    }
    cards = ""
    for mname, info in model_infos.items():
        vlist = grouped.get(mname, [])
        pub = published.get(mname)
        pub_tag = pub.get("version_tag","—") if pub else "未发布"
        latest = vlist[-1] if vlist else None
        vt = ""
        for v in vlist[::-1][:10]:
            sc = f"{v.get('score',''):.4f}" if v.get("score") else "—"
            vt += "<tr>"
            vt += f'<td>{v.get("version_tag","")}</td>'
            vt += f'<td><span class="badge badge-{v.get("status","gray")}">{v.get("status","")}</span></td>'
            vt += f'<td>{sc}</td>'
            vt += f'<td>{v.get("created_at","")[:16]}</td>'
            vt += "</tr>"
        latest_sc = f"{latest['score']:.4f}" if latest and latest.get("score") else "—"
        dim_val = info["dim"]
        cards += f"""<div class="card">
            <h2>{mname} <span class="badge badge-blue">{dim_val}d</span></h2>
            <p style="font-size:13px;color:#666;margin-bottom:12px">{info["desc"]}</p>
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Embedding维度</div><div class="info-value">{dim_val}</div></div>
                <div class="info-item"><div class="info-label">参数量</div><div class="info-value">{info["params"]}</div></div>
                <div class="info-item"><div class="info-label">网络结构</div><div class="info-value" style="font-size:11px">{info["layers"]}</div></div>
                <div class="info-item"><div class="info-label">已发布版本</div><div class="info-value">{pub_tag}</div></div>
                <div class="info-item"><div class="info-label">训练版本数</div><div class="info-value">{len(vlist)}</div></div>
                <div class="info-item"><div class="info-label">最新评分</div><div class="info-value">{latest_sc}</div></div>
                <div class="info-item" style="grid-column:1/-1"><div class="info-label">模型路径</div><div class="info-value" style="font-size:11px;font-family:monospace">{info["path"]}</div></div>
            </div>
            {f'''<table style="margin-top:12px"><thead><tr><th>版本</th><th>状态</th><th>评分</th><th>时间</th></tr></thead><tbody>{vt}</tbody></table>''' if vt else '<p style="color:#999;font-size:13px">暂无训练版本</p>'}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型详情 - ASV</title>
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
</style></head>
<body>
<div class="nav">
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">首页</a>
    <a href="/model-manager/segments">录音断句</a>
    <a href="/model-manager/training">增量训练</a>
    <a href="/model-manager/models" class="active">模型详情</a>
    <a href="/model-manager/publish">发布管理</a>
    <span class="user">{user.get("username","")}</span>
    <a href="/change-password">修改密码</a>
    <a href="/logout">退出</a>
</div>
<div class="container">
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
            sc = f"{v.get('score',''):.4f}" if v.get("score") else "—"
            vt += "<tr>"
            vt += f'<td>{v.get("version_tag","")}</td>'
            vt += f'<td>{v.get("status","")}</td>'
            vt += f'<td>{sc}</td>'
            vt += f'<td>{v.get("created_at","")[:16]}</td>'
            vt += "</tr>"
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
<title>模型发布 - ASV</title>
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
    <strong>ASV 声纹识别系统</strong>
    <a href="/model-manager">首页</a>
    <a href="/model-manager/segments">录音断句</a>
    <a href="/model-manager/models">模型详情</a>
    <a href="/model-manager/publish" class="active">发布管理</a>
    <span class="user">{user.get("username","")}</span>
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
