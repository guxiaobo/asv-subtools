"""
用户认证服务（JWT + bcrypt）。

提供密码哈希、JWT token 签发/验证、FastAPI 依赖注入。
使用 Cookie-based JWT 实现无状态会话。

用户角色：
  admin          — 系统管理员（用户管理）
  model_manager  — 模型管理员（录音管理、训练、发布）
  agent          — 坐席（仅查看和上传自己的录音）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jwt
from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger("asv-api.auth")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

JWT_SECRET_KEY = ""
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 86400  # 24h
JWT_COOKIE_NAME = "asv_session"

ROLE_ADMIN = "admin"
ROLE_MODEL_MANAGER = "model_manager"
ROLE_AGENT = "agent"

ROLE_LABELS = {
    ROLE_ADMIN: "系统管理员",
    ROLE_MODEL_MANAGER: "模型管理员",
    ROLE_AGENT: "坐席",
}

# -------------------------------------------------------------------
# JWT Secret key (auto-generated or from env)
# -------------------------------------------------------------------


def _get_secret_key() -> str:
    """获取 JWT 签名密钥。优先从环境变量读取，否则生成持久化文件。"""
    global JWT_SECRET_KEY
    if JWT_SECRET_KEY:
        return JWT_SECRET_KEY

    env_key = os.environ.get("ASV_JWT_SECRET")
    if env_key:
        JWT_SECRET_KEY = env_key
        return JWT_SECRET_KEY

    # 持久化到文件
    key_file = Path(__file__).resolve().parent.parent / ".jwt_secret"
    if key_file.exists():
        JWT_SECRET_KEY = key_file.read_text().strip()
    else:
        JWT_SECRET_KEY = hashlib.sha256(os.urandom(64)).hexdigest()
        key_file.write_text(JWT_SECRET_KEY)
        logger.info("JWT secret key generated and saved to %s", key_file)

    return JWT_SECRET_KEY


# -------------------------------------------------------------------
# Password hashing (bcrypt)
# -------------------------------------------------------------------


def _get_bcrypt() -> Any:
    """Lazy import bcrypt (not all deployments have it)."""
    import bcrypt as _bcrypt
    return _bcrypt


def hash_password(password: str) -> str:
    """对密码进行 bcrypt 哈希。"""
    bc = _get_bcrypt()
    return bc.hashpw(password.encode("utf-8"), bc.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """校验密码与哈希是否匹配。"""
    bc = _get_bcrypt()
    return bc.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# -------------------------------------------------------------------
# JWT helpers
# -------------------------------------------------------------------


def create_session_token(user_id: int, username: str, role: str, agent_id: str = "") -> str:
    """创建 JWT session token（存储在 cookie 中）。"""
    payload = {
        "uid": user_id,
        "sub": username,
        "role": role,
        "agent_id": agent_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=JWT_ALGORITHM)


def decode_session_token(token: str) -> Optional[Dict[str, Any]]:
    """解码 JWT token，失败返回 None。"""
    try:
        return jwt.decode(token, _get_secret_key(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# -------------------------------------------------------------------
# Cookie helpers
# -------------------------------------------------------------------


def set_session_cookie(response: Response, token: str) -> None:
    """在响应中设置 session cookie。"""
    response.set_cookie(
        key=JWT_COOKIE_NAME,
        value=token,
        max_age=JWT_EXPIRE_SECONDS,
        httponly=True,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """清除 session cookie。"""
    response.delete_cookie(
        key=JWT_COOKIE_NAME,
        path="/",
    )


# -------------------------------------------------------------------
# FastAPI dependency: get current user from request
# -------------------------------------------------------------------


async def get_current_user(request: Request) -> Dict[str, Any]:
    """从请求 cookie 中解析当前用户信息。用于 FastAPI Depends。"""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录",
        )
    payload = decode_session_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="会话已过期或无效，请重新登录",
        )
    return payload


def require_role(*roles: str):
    """FastAPI 依赖工厂：要求用户具有指定角色之一。"""
    async def _check(request: Request) -> Dict[str, Any]:
        user = await get_current_user(request)
        if user is None or user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足",
            )
        request.state.current_user = user
        return user
    return _check


# -------------------------------------------------------------------
# Check if request should redirect to login
# -------------------------------------------------------------------

def is_logged_in(request: Request) -> bool:
    """检查请求是否已登录（不抛出异常）。"""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        return False
    payload = decode_session_token(token)
    return payload is not None
