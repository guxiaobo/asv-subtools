"""
用户认证、用户管理及角色主页路由。

提供以下功能模块：
  1. 登录/注销/修改密码
  2. 系统管理员：用户 CRUD（创建、禁用、重置密码）
  3. 坐席主页：仅查看自己上传的录音 + 上传功能
  4. 模型管理员主页：录音列表/预处理/说话人打标/增量训练/模型对比/发布
"""

from __future__ import annotations

import json
import logging
import os
import asyncio
import subprocess
import sys
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# 先在运行模块下寻找 templates/
_templates_dir = Path(__file__).resolve().parent.parent / "templates"
_templates_dir.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(_templates_dir))

from services.auth import (
    ROLE_ADMIN,
    ROLE_AGENT,
    ROLE_LABELS,
    ROLE_MODEL_MANAGER,
    clear_session_cookie,
    create_session_token,
    get_current_user,
    hash_password,
    is_logged_in,
    require_role,
    verify_password,
)
from services.recording_db import (
    create_user,
    get_user_by_id,
    get_user_by_username,
    list_recordings,
    list_recordings_with_segments,
    get_segments_by_recording,
    update_segment_label,
    list_users,
    update_user,
)

logger = logging.getLogger("asv-api.auth_router")

router = APIRouter(tags=["auth"])

project_root = Path(__file__).resolve().parent.parent.parent  # app/


def _train_env() -> dict:
    """Return env with app/ on PYTHONPATH so train/* scripts can import from train."""
    env = os.environ.copy()
    app_dir = str(project_root)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{app_dir}:{existing}" if existing else app_dir
    return env


# ======================================================================
# 公共页面
# ======================================================================


@router.get("/", response_class=HTMLResponse)
async def root_redirect(request: Request):
    """根路径：已登录跳转到对应主页，否则跳转到登录页。"""
    if is_logged_in(request):
        user = await get_current_user(request)
        return RedirectResponse(url=_home_redirect(user["role"]), status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, err: str = ""):
    """登录页面。"""
    if is_logged_in(request):
        user = await get_current_user(request)
        redirect = _home_redirect(user["role"])
        return RedirectResponse(url=redirect, status_code=302)

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": err},
    )


@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """处理登录表单提交。"""
    try:
        user = await get_user_by_username(username)
        if not user:
            return await _login_error(request, "用户名或密码错误")
        if not user["enabled"]:
            return await _login_error(request, "用户已被禁用，请联系管理员")

        if not verify_password(password, user["password_hash"]):
            return await _login_error(request, "用户名或密码错误")

        token = create_session_token(
            user_id=user["id"],
            username=user["username"],
            role=user["role"],
            agent_id=user.get("agent_id", ""),
        )

        response = RedirectResponse(
            url=_home_redirect(user["role"]), status_code=302
        )
        response.set_cookie(
            key="asv_session",
            value=token,
            max_age=86400,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response
    except Exception as e:
        logger.exception("Login error")
        return await _login_error(request, f"系统错误: {e}")


@router.get("/logout")
async def logout():
    """注销登录。"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="asv_session", path="/")
    return response


@router.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, err: str = "", ok: str = ""):
    """修改密码页面。"""
    user = await get_current_user(request)
    return templates.TemplateResponse(
        "change_password.html",
        {"request": request, "error": err, "ok": ok, "user": user},
    )


@router.post("/change-password")
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
):
    """处理修改密码。"""
    user = await get_current_user(request)
    db_user = await get_user_by_id(user["uid"])
    if not db_user:
        return await _change_password_error(request, "用户不存在")

    if not verify_password(old_password, db_user["password_hash"]):
        return await _change_password_error(request, "原密码错误")

    if len(new_password) < 6:
        return await _change_password_error(request, "新密码至少 6 位")

    await update_user(
        user["uid"],
        password_hash=hash_password(new_password),
    )
    return templates.TemplateResponse(
        "change_password.html",
        {"request": request, "error": "", "ok": "密码修改成功", "user": user},
    )


async def _login_error(request: Request, msg: str) -> HTMLResponse:
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": msg}, status_code=200
    )


async def _change_password_error(request: Request, msg: str) -> HTMLResponse:
    user = await get_current_user(request)
    return templates.TemplateResponse(
        "change_password.html",
        {"request": request, "error": msg, "ok": "", "user": user},
    )


def _home_redirect(role: str) -> str:
    if role == ROLE_ADMIN:
        return "/admin"
    elif role == ROLE_MODEL_MANAGER:
        return "/model-manager"
    return "/agent"


# ======================================================================
# 系统管理员页面
# ======================================================================


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """系统管理员仪表盘 — 用户管理。"""
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    user = await get_current_user(request)
    if user["role"] != ROLE_ADMIN:
        return RedirectResponse(url=_home_redirect(user["role"]), status_code=302)
    users = await list_users()
    from services.recording_db import get_multi_period_call_stats
    call_stats = await get_multi_period_call_stats()
    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "user": user,
            "users": users,
            "roles": ROLE_LABELS,
            "call_stats": call_stats,
        },
    )


@router.post("/admin/user/create")
async def admin_create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    agent_id: str = Form(""),
    display_name: str = Form(""),
):
    """管理员创建用户。"""
    await require_role(ROLE_ADMIN)(request)
    if role not in (ROLE_ADMIN, ROLE_MODEL_MANAGER, ROLE_AGENT):
        return JSONResponse(
            {"success": False, "error": f"无效角色: {role}"}, status_code=400
        )
    if role == ROLE_AGENT and not agent_id.strip():
        return JSONResponse(
            {"success": False, "error": "坐席用户必须填写坐席ID"}, status_code=400
        )
    if len(password) < 6:
        return JSONResponse(
            {"success": False, "error": "密码至少 6 位"}, status_code=400
        )

    try:
        uid = await create_user(
            username=username.strip(),
            password_hash=hash_password(password),
            role=role,
            agent_id=agent_id.strip(),
            display_name=display_name.strip(),
        )
        return {"success": True, "user_id": uid}
    except ValueError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@router.post("/admin/user/toggle")
async def admin_toggle_user(
    request: Request,
    user_id: int = Form(...),
):
    """启用/禁用用户。"""
    await require_role(ROLE_ADMIN)(request)
    user = await get_user_by_id(user_id)
    if not user:
        return JSONResponse({"success": False, "error": "用户不存在"}, status_code=404)

    new_enabled = not user["enabled"]
    await update_user(user_id, enabled=new_enabled)
    return {
        "success": True,
        "enabled": new_enabled,
        "user_id": user_id,
    }


@router.post("/admin/user/reset-password")
async def admin_reset_password(
    request: Request,
    user_id: int = Form(...),
    new_password: str = Form(...),
):
    """管理员重置用户密码。"""
    await require_role(ROLE_ADMIN)(request)
    user = await get_user_by_id(user_id)
    if not user:
        return JSONResponse({"success": False, "error": "用户不存在"}, status_code=404)
    if len(new_password) < 6:
        return JSONResponse(
            {"success": False, "error": "密码至少 6 位"}, status_code=400
        )

    await update_user(user_id, password_hash=hash_password(new_password))
    return {"success": True, "message": f"用户 {user['username']} 密码已重置"}


# ======================================================================
# 坐席主页
# ======================================================================


@router.get("/agent", response_class=HTMLResponse)
async def agent_home(request: Request, err: str = "", ok: str = ""):
    """坐席主页 — 上传录音 + 查看自己录音列表。"""
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    user = await get_current_user(request)
    if user["role"] != ROLE_AGENT:
        return RedirectResponse(url=_home_redirect(user["role"]), status_code=302)
    agent_id_filter = user.get("agent_id", "")
    recordings = await list_recordings(agent_id=agent_id_filter, limit=50)

    return templates.TemplateResponse(
        "agent_home.html",
        {
            "request": request,
            "user": user,
            "recordings": recordings,
            "error": err,
            "ok": ok,
        },
    )


# ======================================================================
# 模型管理员主页
# ======================================================================


@router.get("/model-manager", response_class=HTMLResponse)
async def model_manager_dashboard(
    request: Request,
    err: str = "",
    ok: str = "",
    status_filter: str = "",
):
    """模型管理员主页 — 录音管理 + 训练发布。"""
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=302)
    user = await get_current_user(request)
    if user["role"] != ROLE_MODEL_MANAGER:
        return RedirectResponse(url=_home_redirect(user["role"]), status_code=302)

    from services.recording_db import get_dashboard_stats, get_multi_period_call_stats
    stats = await get_dashboard_stats()
    call_stats = await get_multi_period_call_stats()

    # 格式化运行时长
    uptime_sec = stats["uptime_sec"]
    hours, rem = divmod(int(uptime_sec), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 24:
        days = hours // 24
        hours = hours % 24
        uptime_str = f"{days}天{hours}小时{minutes}分"
    elif hours > 0:
        uptime_str = f"{hours}小时{minutes}分"
    else:
        uptime_str = f"{minutes}分{seconds}秒"

    return templates.TemplateResponse(
        "model_manager_home.html",
        {
            "request": request,
            "user": user,
            "stats": stats,
            "uptime_str": uptime_str,
            "call_stats": call_stats,
            "error": err,
            "ok": ok,
        },
    )


@router.post("/model-manager/run-preprocess")
async def run_preprocess(request: Request):
    """启动批量预处理（VAD 分割 + 说话人分离）。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    recording_id = None
    recording_ids = None
    try:
        body = await request.json()
        recording_id = body.get("recording_id")
        recording_ids = body.get("recording_ids")
    except Exception:
        pass
    try:
        from services.recording_db import set_pre_statuses

        # 先将选中录音的 pre_status 设为 pending，让脚本自动拾取
        ids_to_run = []
        if recording_ids:
            ids_to_run = recording_ids
        elif recording_id:
            ids_to_run = [recording_id]

        # 检查各录音是否有现有片段
        from services.recording_db import get_segment_count_for_recordings
        seg_counts = await get_segment_count_for_recordings(ids_to_run) if ids_to_run else {}

        # 调用训练系统的预处理脚本
        script = project_root / "train" / "preprocess.py"
        if not script.exists():
            return JSONResponse({
                "success": False,
                "error": f"预处理脚本不存在: {script}",
            }, status_code=500)

        env = _train_env()

        from datetime import datetime
        batch_id = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # 读取 VAD 配置中的 min_segment_sec_ignore
        vad_cfg_path = project_root / "data" / "vad_config.json"
        min_ignore_arg = ""
        try:
            if vad_cfg_path.exists():
                vad_cfg = json.loads(vad_cfg_path.read_text())
                mi = float(vad_cfg.get("min_segment_sec_ignore", 0.0))
                if mi > 0:
                    min_ignore_arg = f"--min-ignore={mi}"
        except Exception:
            pass

        first_time_ids = [rid for rid in ids_to_run if seg_counts.get(rid, 0) == 0]
        reseg_ids = [rid for rid in ids_to_run if seg_counts.get(rid, 0) > 0]

        bg_tasks = []

        # 首次断句录音：设 pending，让脚本批量 claim
        if first_time_ids:
            await set_pre_statuses(first_time_ids, "pending")
            args_bt = [sys.executable, str(script), "--batch-id", batch_id]
            if min_ignore_arg:
                args_bt.append(min_ignore_arg)
            bg_tasks.append(_run_preprocess_in_bg(args_bt, env))

        # 重新断句录音：设 reprocessing 让界面显示"重新断句中"，process_single 入口会转为 processing
        if reseg_ids:
            await set_pre_statuses(reseg_ids, "reprocessing")
        for rid in reseg_ids:
            args_re = [sys.executable, str(script), "--recording-id", str(rid), "--batch-id", batch_id]
            if min_ignore_arg:
                args_re.append(min_ignore_arg)
            bg_tasks.append(_run_preprocess_in_bg(args_re, env))

        # 有任务则启动
        if bg_tasks:
            for t in bg_tasks:
                asyncio.create_task(t)

        msg_parts = []
        if first_time_ids:
            msg_parts.append(f"{len(first_time_ids)} 条首次断句")
        if reseg_ids:
            msg_parts.append(f"{len(reseg_ids)} 条重新断句（batch={batch_id}）")

        return {
            "success": True,
            "message": f"预处理任务已启动：{'、'.join(msg_parts)}，请稍后刷新页面查看结果",
            "batch_id": batch_id,
        }
    except Exception as e:
        logger.exception("Preprocess error")
        return JSONResponse({
            "success": False,
            "error": str(e),
        }, status_code=500)


async def _run_preprocess_in_bg(args: List[str], env: dict) -> None:
    """后台执行预处理脚本。"""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        if result.returncode == 0:
            logger.info("后台预处理完成: stdout=%s", result.stdout[-500:])
        else:
            logger.error("后台预处理失败 rc=%d: stderr=%s",
                         result.returncode, result.stderr[-2000:])
    except subprocess.TimeoutExpired:
        logger.error("后台预处理超时")
    except Exception as e:
        logger.exception("后台预处理异常")


# ── 后台自动打标任务 ──


async def _run_auto_label_task(task_id: int, model_name: str, checkpoint_id: int, threshold: float) -> None:
    """Background task for auto-labeling (runs via asyncio.create_task).
    Supports 'all' model_name to run all checkpoints."""
    import numpy as np
    from routers.verify import _get_verifier as _get_vf
    from services.recording_db import _open_conn
    from pathlib import Path

    logger.info("Auto-label task %d started: model=%s cp=%s", task_id, model_name, checkpoint_id)

    verifier = None
    try:
        verifier = _get_vf()
    except RuntimeError:
        pass

    conn = await _open_conn()
    try:
        await conn.execute("UPDATE auto_label_tasks SET status='running', started_at=datetime('now','localtime') WHERE id=?", (task_id,))
        await conn.commit()

        # Load recordings + all segments
        recordings_dict = {}
        cur = await conn.execute("SELECT id, agent_id, customer_id FROM recordings ORDER BY id")
        for r in await cur.fetchall():
            recordings_dict[r["id"]] = {"agent_id": str(r["agent_id"] or ""), "customer_id": str(r["customer_id"] or "")}

        cur = await conn.execute(
            "SELECT id, recording_id, file_path, speaker_label, speaker_type, is_ignored "
            "FROM audio_segments WHERE is_ignored IS NULL OR is_ignored = 0"
        )
        all_seg_rows = [dict(r) for r in await cur.fetchall()]
        total = len(all_seg_rows)
        await conn.execute("UPDATE auto_label_tasks SET total_segments=? WHERE id=?", (total, task_id))
        await conn.commit()
        logger.info("Task %d: %d segments to process", task_id, total)

        # Determine which checkpoints to run
        if model_name == "all":
            # Find latest checkpoint per model
            cur = await conn.execute(
                "SELECT c.id AS cp_id, c.model_name, c.version_tag FROM checkpoints c "
                "INNER JOIN (SELECT model_name, MAX(id) AS max_id FROM checkpoints GROUP BY model_name) latest "
                "ON c.id = latest.max_id ORDER BY c.model_name"
            )
            checkpoints_to_run = [(r["cp_id"], r["model_name"]) for r in await cur.fetchall()]
        else:
            checkpoints_to_run = [(checkpoint_id, model_name)]

        logger.info("Task %d: %d checkpoints to run", task_id, len(checkpoints_to_run))

        # Pre-load checkpoint model file paths
        cp_models = {}  # cp_id -> onnx_path
        for cp_id, cp_mn in checkpoints_to_run:
            cur = await conn.execute("SELECT file_path FROM checkpoints WHERE id = ?", (cp_id,))
            row = await cur.fetchone()
            fpath = (row["file_path"] or "") if row else ""
            if fpath and fpath.endswith(".onnx") and Path(fpath).exists():
                cp_models[cp_id] = fpath
            else:
                # Try to find ONNX by model name in api/models/
                models_dir = Path(__file__).resolve().parent.parent / "models"
                onnx_map = {
                    "CAM++": "campplus.onnx",
                    "ECAPA": "ecapa-speaker-v1.onnx",
                    "ResNet34": "voxceleb_resnet34_LM.onnx",
                }
                onnx_name = onnx_map.get(cp_mn, "")
                if onnx_name:
                    candidate = models_dir / onnx_name
                    if candidate.exists():
                        cp_models[cp_id] = str(candidate)
                        logger.info("Mapped cp %d (%s) to ONNX: %s", cp_id, cp_mn, candidate)
                    else:
                        logger.warning("No ONNX file for cp %d (%s) at %s", cp_id, cp_mn, candidate)
                else:
                    logger.warning("No ONNX mapping for model %s (cp %d)", cp_mn, cp_id)

            if cp_id not in cp_models:
                logger.warning("Task %d: checkpoint %d has no usable ONNX model, will use global fallback", task_id, cp_id)

        # Pre-compute embeddings dict (keyed by (seg_id, cp_id) -> embedding)
        all_embeddings = {}  # (seg_id, cp_id) -> np.ndarray
        seg_meta = {}
        for s in all_seg_rows:
            sid = s["id"]
            rid = s["recording_id"]
            rec = recordings_dict.get(rid, {"agent_id": "", "customer_id": ""})
            seg_meta[sid] = {
                "recording_id": rid,
                "agent_id": rec["agent_id"] or "",
                "customer_id": rec["customer_id"] or "",
                "speaker_label": s.get("speaker_label") or "",
                "speaker_type": s.get("speaker_type") or "",
            }

        # Group segments by (agent_id, customer_id) for reuse
        groups = {}  # (agent_id, customer_id) -> [sid, ...]
        for sid, meta in seg_meta.items():
            key = (meta["agent_id"], meta["customer_id"])
            groups.setdefault(key, []).append(sid)

        processed_total = 0
        for cp_id, cp_model in checkpoints_to_run:
            logger.info("Task %d: processing checkpoint %d (%s)", task_id, cp_id, cp_model)

            # Load this checkpoint's ONNX model (temporary, separate from global verifier)
            tmp_model = None
            cp_model_path = cp_models.get(cp_id, "")
            if cp_model_path and Path(cp_model_path).exists():
                try:
                    from onnx_model import ONNXModel
                    tmp_model = ONNXModel(
                        model_path=cp_model_path,
                        provider="CPUExecutionProvider",
                        intra_op_threads=2,
                        hot_reload_interval_sec=0,
                    )
                    logger.info("Loaded ONNX model for cp %d: %s", cp_id, cp_model_path)
                except Exception as e:
                    logger.warning("Failed to load ONNX model for cp %d: %s", cp_id, e)

            if not tmp_model:
                logger.warning("Task %d: no ONNX model for cp %d, using global verifier fallback", task_id, cp_id)

            # Compute embeddings for all segments with this model
            cp_embeddings = {}
            for sid in seg_meta:
                s = next((x for x in all_seg_rows if x["id"] == sid), None)
                if not s:
                    continue
                audio_path = s.get("file_path", "")
                if not (audio_path and Path(audio_path).exists()):
                    continue
                if (sid, cp_id) in all_embeddings:
                    cp_embeddings[sid] = all_embeddings[(sid, cp_id)]
                else:
                    try:
                        # Load audio via global verifier (use existing audio loader)
                        if not verifier:
                            continue
                        ad = verifier._audio_loader.load_from_file(audio_path)
                        # Extract fbank features
                        feat_config = verifier._config.audio
                        fbank = verifier._audio_loader.extract_fbank(
                            ad,
                            num_filters=feat_config.fbank_num_filters,
                            window_ms=feat_config.fbank_window_ms,
                            hop_ms=feat_config.fbank_hop_ms,
                        )
                        if fbank.ndim == 2:
                            input_tensor = fbank[np.newaxis, ...]
                        else:
                            input_tensor = fbank

                        if tmp_model:
                            # Use temporary checkpoint model
                            primary_input = "feats"
                            if primary_input not in tmp_model.input_names:
                                primary_input = tmp_model.input_names[0]
                            feed_dict = {}
                            for inp_name in tmp_model.input_names:
                                if inp_name == primary_input:
                                    feed_dict[inp_name] = input_tensor.astype(np.float32)
                                elif inp_name.lower() in ("feature_lens", "input_lengths", "lengths"):
                                    feed_dict[inp_name] = np.array([input_tensor.shape[1]], dtype=np.float32)
                                else:
                                    feed_dict[inp_name] = input_tensor.astype(np.float32)
                            output = tmp_model.infer(feed_dict)
                            emb = output[0].flatten() if isinstance(output, list) else list(output.values())[0].flatten()
                        else:
                            # Fallback to global verifier
                            emb = verifier._compute_embedding(ad)

                        cp_embeddings[sid] = emb
                        all_embeddings[(sid, cp_id)] = emb
                    except Exception as ex:
                        logger.warning("Task %d: failed embedding for seg %d cp %d: %s", task_id, sid, cp_id, ex)
                        continue

            tmp_model = None  # Release model

            if not cp_embeddings:
                logger.warning("Task %d: no embeddings for cp %d", task_id, cp_id)
                continue

            # Process each group
            used_new_labels = set()  # track new speaker labels for uniqueness
            group_id = 0
            for (g_agent, g_cust), g_sids in groups.items():
                group_id += 1
                g_sids_with_emb = [sid for sid in g_sids if sid in cp_embeddings]
                if len(g_sids_with_emb) < 2:
                    continue

                # Pairwise similarity matrix
                pairwise = {}
                for i in range(len(g_sids_with_emb)):
                    for j in range(i + 1, len(g_sids_with_emb)):
                        sid_a, sid_b = g_sids_with_emb[i], g_sids_with_emb[j]
                        emb_a, emb_b = cp_embeddings[sid_a], cp_embeddings[sid_b]
                        sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10))
                        pairwise[f"{sid_a}:{sid_b}"] = round(sim, 4)

                # Cluster: group by similarity >= threshold
                clusters = []  # [[sid, ...], ...]
                used = set()
                for sid in g_sids_with_emb:
                    if sid in used:
                        continue
                    cluster = [sid]
                    used.add(sid)
                    for oid in g_sids_with_emb:
                        if oid in used:
                            continue
                        key = f"{min(sid, oid)}:{max(sid, oid)}"
                        sim = pairwise.get(key, 0)
                        if sim >= threshold:
                            cluster.append(oid)
                            used.add(oid)
                    if len(cluster) >= 2:
                        clusters.append(cluster)

                # Process each cluster → determine label
                for cl in clusters:
                    # Check for existing labels
                    label_votes = {}
                    for sid in cl:
                        lbl = seg_meta[sid]["speaker_label"]
                        if lbl:
                            label_votes[lbl] = label_votes.get(lbl, 0) + 1

                    # Determine recommended label
                    if label_votes:
                        best_label = max(label_votes, key=label_votes.get)
                        best_score = max(pairwise.get(f"{min(cl[0], sid)}:{max(cl[0], sid)}", 0) for sid in cl[1:]) if len(cl) > 1 else 0
                        # Strip any old-style prefix to determine type
                        _clean = best_label.replace("@agent_", "").replace("@customer_", "").replace("agent_", "").replace("customer_", "")
                        sp_type = "agent" if "@agent_" in best_label or best_label.startswith("agent_") else \
                                  "customer" if "@customer_" in best_label or best_label.startswith("customer_") else "unknown"
                        best_label = _clean  # store clean label without prefix
                        reason = f"组内{label_votes[best_label]}段已打标一致，继承标签 {best_label}"
                    else:
                        # Check agent/customer pattern
                        cl_recs = set(seg_meta[sid]["recording_id"] for sid in cl if sid in seg_meta)
                        agent_recs = set(rid for rid, rm in recordings_dict.items() if rm["agent_id"] == g_agent) if g_agent else set()
                        cust_recs = set(rid for rid, rm in recordings_dict.items() if rm["customer_id"] == g_cust) if g_cust else set()
                        is_agent = g_agent and cl_recs >= agent_recs and len(agent_recs) >= 2
                        is_cust = g_cust and cl_recs >= cust_recs and len(cust_recs) >= 2

                        if is_agent:
                            best_label = g_agent
                            sp_type = "agent"
                            best_score = max(pairwise.get(f"{min(cl[0], sid)}:{max(cl[0], sid)}", 0) for sid in cl[1:]) if len(cl) > 1 else 0
                            reason = f"覆盖全部{len(agent_recs)}通坐席{g_agent}录音，识别为坐席声纹"
                        elif is_cust:
                            best_label = g_cust
                            sp_type = "customer"
                            best_score = max(pairwise.get(f"{min(cl[0], sid)}:{max(cl[0], sid)}", 0) for sid in cl[1:]) if len(cl) > 1 else 0
                            reason = f"覆盖全部{len(cust_recs)}通客户{g_cust}录音，识别为客户声纹"
                        else:
                            # New speaker: use customer_id or agent_id as base, with numeric suffix for uniqueness
                            _base = g_cust or g_agent or "unknown"
                            _candidate = _base
                            _suffix = 2
                            while _candidate in used_new_labels:
                                _candidate = f"{_base}_{_suffix}"
                                _suffix += 1
                            used_new_labels.add(_candidate)
                            best_label = _candidate
                            sp_type = "customer" if g_cust else ("agent" if g_agent else "unknown")
                            best_score = max(pairwise.get(f"{min(cl[0], sid)}:{max(cl[0], sid)}", 0) for sid in cl[1:]) if len(cl) > 1 else 0
                            reason = f"新说话人候选（组内{len(cl)}段相似），基于录音{('客户'+g_cust) if g_cust else ('坐席'+g_agent if g_agent else '未知')}命名"

                    # Save to auto_label_preview + auto_label_details
                    for sid in cl:
                        meta = seg_meta[sid]
                        if meta["speaker_label"]:
                            continue  # skip already-labeled
                        # Preview table (for confirmation)
                        await conn.execute(
                            "INSERT OR REPLACE INTO auto_label_preview "
                            "(segment_id, speaker_label, speaker_type, score, reason, checkpoint_id) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (sid, best_label, sp_type, best_score, reason, cp_id))
                        # Details table (for graphical display)
                        scores_str = ";".join(f"{k}:{v}" for k, v in pairwise.items())
                        cluster_str = ";".join(":".join(str(x) for x in cl) for cl in clusters)
                        await conn.execute(
                            "INSERT OR REPLACE INTO auto_label_details "
                            "(task_id, segment_id, group_id, group_segs, model_checkpoint_id, "
                            "group_scores, cluster_result, speaker_label, speaker_type, score, reason) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (task_id, sid, group_id, ":".join(str(x) for x in g_sids_with_emb),
                             cp_id, scores_str, cluster_str,
                             best_label, sp_type, best_score, reason))
                        processed_total += 1
                    await conn.execute("UPDATE auto_label_tasks SET processed=? WHERE id=?", (processed_total, task_id))
                await conn.commit()

        # Mark complete
        await conn.execute("UPDATE auto_label_tasks SET status='completed', progress=100, processed=?, completed_at=datetime('now','localtime') WHERE id=?", (processed_total, task_id))
        await conn.commit()
        logger.info("Auto-label task %d completed: %d segments processed", task_id, processed_total)

    except Exception as e:
        logger.exception("Auto-label task %d failed", task_id)
        try:
            await conn.execute("UPDATE auto_label_tasks SET status='failed', error=? WHERE id=?", (str(e), task_id))
            await conn.commit()
        except Exception:
            pass
    finally:
        await conn.close()


@router.post("/model-manager/run-label")
async def run_label_speakers(request: Request):
    """说话人自动打标：预览模式（选模型+版本）或确认保存模式。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    try:
        data = await request.json()
        checkpoint_id = data.get("checkpoint_id")
        preview_only = data.get("preview_only", False)
        confirm_segments = data.get("confirm_segments")
        results_data = data.get("results", {})
        model_name = data.get("model_name", "")
        threshold = data.get("threshold", 0.35)

        if preview_only:
            # ── Launch background auto-label task ──
            from services.recording_db import _open_conn
            conn = await _open_conn()
            try:
                cur = await conn.execute(
                    "SELECT COUNT(*) AS cnt FROM auto_label_tasks WHERE status IN ('pending','running')"
                )
                row = await cur.fetchone()
                if row and row["cnt"] > 0:
                    return JSONResponse({"success": False, "error": "已有自动打标任务正在运行，请等待完成"}, status_code=409)
                
                cur = await conn.execute(
                    "INSERT INTO auto_label_tasks "
                    "(status, checkpoint_id, model_name, threshold, min_segments) "
                    "VALUES ('pending', ?, ?, ?, ?)",
                    (checkpoint_id, model_name, threshold, data.get("min_segments", 2)),
                )
                await conn.commit()
                task_id = cur.lastrowid
            finally:
                await conn.close()

            # Spawn background task
            asyncio.create_task(_run_auto_label_task(task_id, model_name, checkpoint_id, threshold))

            return {"success": True, "task_id": task_id, "status": "pending"}

        if confirm_segments:
            # ── Confirm: save to audio_segments + speaker_voiceprints ──
            # Preview data is preserved (marked confirmed) for historical review.
            from services.recording_db import _open_conn
            conn = await _open_conn()
            saved_count = 0
            conflict_count = 0
            # Collect voiceprint data for batch insert
            vp_rows = []  # (model_name, speaker_type, speaker_id, embedding, seg_count, call_ids)
            try:
                for seg_id_str in confirm_segments:
                    try:
                        seg_id = int(seg_id_str)
                        # Get recommended label from preview table
                        cur = await conn.execute(
                            "SELECT speaker_label, speaker_type, reason FROM auto_label_preview WHERE segment_id = ?",
                            (seg_id,),
                        )
                        row = await cur.fetchone()
                        if not row:
                            continue
                        # Check for conflict
                        cur2 = await conn.execute(
                            "SELECT speaker_label, recording_id FROM audio_segments WHERE id = ?",
                            (seg_id,),
                        )
                        existing = await cur2.fetchone()
                        if existing and existing["speaker_label"]:
                            conflict_count += 1
                        # Update audio_segments (formal table)
                        await conn.execute(
                            "UPDATE audio_segments SET speaker_label=?, speaker_type=?, label_source='auto' WHERE id=?",
                            (row["speaker_label"], row["speaker_type"] or "unknown", seg_id),
                        )
                        # Collect voiceprint info from details table
                        cur3 = await conn.execute(
                            "SELECT d.model_checkpoint_id, d.embedding, c.model_name, a.recording_id, r.call_id "
                            "FROM auto_label_details d "
                            "LEFT JOIN checkpoints c ON d.model_checkpoint_id = c.id "
                            "LEFT JOIN audio_segments a ON d.segment_id = a.id "
                            "LEFT JOIN recordings r ON a.recording_id = r.id "
                            "WHERE d.segment_id = ? AND d.embedding IS NOT NULL "
                            "ORDER BY d.model_checkpoint_id LIMIT 1",
                            (seg_id,),
                        )
                        vp_row = await cur3.fetchone()
                        if vp_row and vp_row["embedding"] and vp_row["model_name"]:
                            vp_rows.append((
                                vp_row["model_name"],
                                row["speaker_type"] or "unknown",
                                row["speaker_label"],
                                vp_row["embedding"],
                                1,
                                str(vp_row["call_id"] or vp_row["recording_id"] or ""),
                            ))
                        # Mark preview as confirmed (not delete) so results stay viewable
                        await conn.execute(
                            "UPDATE auto_label_preview SET reason=? WHERE segment_id=?",
                            (row["reason"] + " [已确认保存]" if row["reason"] else "[已确认保存]", seg_id),
                        )
                        saved_count += 1
                    except Exception:
                        continue
                # Batch insert voiceprints
                if vp_rows:
                    await conn.executemany(
                        "INSERT INTO speaker_voiceprints (model_name, speaker_type, speaker_id, embedding, segment_count, source_call_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        vp_rows,
                    )
                await conn.commit()
            finally:
                await conn.close()
            return {"success": True, "saved_count": saved_count, "conflict_count": conflict_count, "voiceprints": len(vp_rows)}

        # Legacy mode — run external script
        script = project_root / "train" / "preprocess.py"
        if not script.exists():
            return JSONResponse({
                "success": False,
                "error": f"打标脚本不存在: {script}",
            }, status_code=500)

        env = _train_env()
        args = [sys.executable, str(script), "--cross-aggregate-only"]
        result = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "success": False, "error": "说话人打标超时（7200s）",
        }, status_code=500)
    except Exception as e:
        logger.exception("Label error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.post("/model-manager/run-train")
async def run_incremental_train(request: Request):
    """启动增量训练。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    try:
        script = project_root / "train" / "fine_tune.py"

        env = _train_env()
        args = [sys.executable, str(script), "--model", "all"]
        result = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=14400,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "success": False, "error": "增量训练超时（14400s）",
        }, status_code=500)
    except Exception as e:
        logger.exception("Train error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.post("/model-manager/run-evaluate")
async def run_evaluate(request: Request):
    """启动模型评估对比。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    try:
        script = project_root / "train" / "evaluate.py"

        env = _train_env()
        args = [sys.executable, str(script), "--model", "all"]
        result = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "success": False, "error": "模型评估超时（7200s）",
        }, status_code=500)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@router.post("/model-manager/run-publish")
async def run_publish_model(request: Request, version: str = Form("")):
    """发布线上模型。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    try:
        script = project_root / "train" / "model_manager.py"
        env = _train_env()
        args = [sys.executable, str(script), "--publish"]
        if version:
            args.extend(["--version", version])

        result = subprocess.run(
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-1000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ======================================================================
# 初始化默认管理员（首次启动自动创建）
# ======================================================================

async def ensure_default_admin():
    """确保系统至少存在一个默认管理员账号。仅在首次启动时创建。"""
    try:
        admin = await get_user_by_username("admin")
        if admin:
            return

        await create_user(
            username="admin",
            password_hash=hash_password("admin123"),
            role=ROLE_ADMIN,
            display_name="系统管理员",
        )
        logger.info(
            "Default admin user created: admin / admin123. "
            "CHANGE THIS PASSWORD IMMEDIATELY after first login!"
        )
    except Exception as e:
        logger.warning("Failed to create default admin user: %s", e)