#!/usr/bin/env bash
# 阿里云 ECS（Ubuntu）一键部署脚本
#
# 用法（在服务器上）：
#   cd /opt/mma
#   cp env.txt .env && vim .env      # 先填 key
#   bash scripts/deploy_ecs.sh
#
# 做的事：装 docker → 校验配置 → 构建并启动 → 打印健康状态
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

say() { printf "\n\033[1;36m== %s ==\033[0m\n" "$1"; }
fail() { printf "\033[1;31m[FAIL] %s\033[0m\n" "$1" >&2; exit 1; }

say "1/5 检查环境"
command -v git >/dev/null || fail "缺少 git：apt install -y git"
if ! command -v docker >/dev/null; then
  say "未安装 docker，开始安装"
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
else
  echo "docker 已安装：$(docker --version)"
fi
docker compose version >/dev/null 2>&1 || fail "缺少 docker compose 插件：apt install -y docker-compose-plugin"

say "2/5 检查 .env"
[ -f .env ] || fail ".env 不存在。先执行：cp env.txt .env 然后填 key"
grep -q "REPLACE_ME" .env && fail ".env 里还有 REPLACE_ME 占位符，先填真实 key"
echo ".env 存在"

say "3/5 校验 compose 配置"
docker compose config --quiet && echo "配置合法"

say "4/5 构建并启动（首次会拉依赖，几分钟）"
docker compose up -d --build

say "5/5 等待健康检查"
for i in $(seq 1 40); do
  sleep 5
  STATUS=$(docker inspect --format '{{.State.Health.Status}}' multi-model-agent-debate 2>/dev/null || echo "starting")
  echo "  [$i/40] $STATUS"
  [ "$STATUS" = "healthy" ] && break
done

if curl -fsS http://127.0.0.1:8000/health | head -c 300; then
  echo
  say "部署完成"
  echo "本机访问：http://127.0.0.1:8000"
  IP=$(curl -fsS --max-time 5 https://api.ipify.org || echo "<你的公网IP>")
  echo "公网访问：http://$IP:8000   （记得在安全组放行 8000）"
  echo
  echo "提醒："
  echo "  · 阿里云内地服务器绑域名走 80/443 需要备案；没备案先用 IP:8000 演示"
  echo "  · 公网暴露前务必改 .env 里的 API_KEY"
else
  fail "健康检查失败，看日志：docker compose logs -f --tail 50"
fi
