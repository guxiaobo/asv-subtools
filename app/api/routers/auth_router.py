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
    try:
        body = await request.json()
        recording_id = body.get("recording_id")
    except Exception:
        pass
    try:
        # 调用训练系统的预处理脚本
        script = project_root / "train" / "preprocess.py"
        if not script.exists():
            return JSONResponse({
                "success": False,
                "error": f"预处理脚本不存在: {script}",
            }, status_code=500)

        env = _train_env()
        args = [sys.executable, str(script)]
        if recording_id:
            args.extend(["--recording-id", str(recording_id)])

        result = await asyncio.to_thread(
            subprocess.run,
            args,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        logger.info("Preprocess result: stdout=%s stderr=%s", result.stdout, result.stderr)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "success": False,
            "error": "预处理超时（3600s）",
        }, status_code=500)
    except Exception as e:
        logger.exception("Preprocess error")
        return JSONResponse({
            "success": False,
            "error": str(e),
        }, status_code=500)


@router.post("/model-manager/run-label")
async def run_label_speakers(request: Request):
    """启动说话人打标（跨录音聚合 + 客户声纹注册）。"""
    user = await require_role(ROLE_MODEL_MANAGER)(request)
    try:
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