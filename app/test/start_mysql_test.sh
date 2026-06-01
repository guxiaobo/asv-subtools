#!/usr/bin/env bash
# Wrapper: start API with MySQL fetcher on port 3307
export ASV_MYSQL_PORT=3307
cd /Users/guxiaobo/Documents/GitHub/asv-subtools/app
exec .venv/bin/python3 -m api.main
