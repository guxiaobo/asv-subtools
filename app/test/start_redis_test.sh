#!/usr/bin/env bash
# Wrapper: start API with Redis fetcher on port 6380
export ASV_REDIS_URL="redis://localhost:6380/0"
cd /Users/guxiaobo/Documents/GitHub/asv-subtools/app
exec .venv/bin/python3 -m api.main
