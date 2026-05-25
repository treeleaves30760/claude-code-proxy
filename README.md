# Claude Code OpenAI Proxy

這是一個 Anthropic Messages API 相容的 HTTP proxy，讓 Claude Code 可以把請求送到 OpenAI Chat Completions API 或 OpenAI-compatible endpoint。

目前支援：

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`，以近似字元數估算 token
- `GET /v1/models`
- 非串流與 SSE 串流回應
- Claude Code 常用的 tool use / tool result 轉換
- Docker 與 Kubernetes 部署

## 環境變數

| 變數 | 必填 | 說明 |
| --- | --- | --- |
| `OPENAI_API_ENDPOINT` | 是 | OpenAI API base URL，例如 `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | 是 | OpenAI API key |
| `OPENAI_MODEL` | 建議 | 固定轉送到的 OpenAI model；預設 `gpt-5.5` |
| `PROXY_AUTH_TOKEN` | 否 | 若設定，Claude Code 必須用 `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY` 傳入相同 token |
| `MODEL_MAPPING_JSON` | 否 | JSON object，用來把 Claude model name 映射到 OpenAI model |
| `PUBLIC_MODELS` | 否 | `/v1/models` 回傳給 Claude Code 的 model 清單 |
| `OPENAI_MAX_TOKENS_FIELD` | 否 | `auto`、`max_completion_tokens` 或 `max_tokens`；預設 `auto` |
| `REQUEST_TIMEOUT_SECONDS` | 否 | 上游 API timeout，預設 `600` |
| `OPENAI_EXTRA_HEADERS_JSON` | 否 | JSON object，轉送上游時額外加的 headers |
| `OPENAI_EXTRA_BODY_JSON` | 否 | JSON object，轉送上游時額外合併到 OpenAI request body |

範例：

```bash
cp .env.example .env
```

`.env`：

```dotenv
OPENAI_API_ENDPOINT=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.5
PROXY_AUTH_TOKEN=local-shared-token
MODEL_MAPPING_JSON={"proxy-default":"gpt-5.5","claude-sonnet-4-5":"gpt-5.5"}
```

## 本機執行

```bash
uv sync
uv run uvicorn claude_code_proxy.server:app --host 0.0.0.0 --port 8080
```

健康檢查：

```bash
curl http://localhost:8080/healthz
```

## Docker

```bash
docker build -t claude-code-proxy:latest .
docker run --rm --env-file .env -p 8080:8080 claude-code-proxy:latest
```

或：

```bash
docker compose up --build
```

## GHCR Image

此 repo 內建 GitHub Actions workflow：`.github/workflows/docker-image.yml`。推送到 `main` 或推送 `v*.*.*` tag 時會自動 build multi-arch image，並推到 GitHub Container Registry。

產生的 image 名稱會是：

```text
ghcr.io/<owner>/<repo>
```

常用 tag：

- `latest`：default branch 的最新 build
- `sha-<commit>`：指定 commit build
- `<version>`：從 `v1.2.3` 這類 tag 產生，例如 `1.2.3`
- `<major>.<minor>`：例如 `1.2`

下載 image：

```bash
docker pull ghcr.io/<owner>/<repo>:latest
```

直接執行：

```bash
docker run --rm --env-file .env -p 8080:8080 ghcr.io/<owner>/<repo>:latest
```

如果 GHCR package 還是 private，先登入：

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
docker pull ghcr.io/<owner>/<repo>:latest
```

`GHCR_TOKEN` 需要有 `read:packages` 權限。若要讓其他人免登入下載，請到 GitHub package 頁面把 visibility 改成 public。

## Claude Code 設定

Claude Code 官方文件提供 `ANTHROPIC_BASE_URL` 用來把 API 請求導到 proxy/gateway，`ANTHROPIC_AUTH_TOKEN` 會被送成 Bearer token。

```bash
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=local-shared-token
export ANTHROPIC_MODEL=proxy-default
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
claude
```

`ANTHROPIC_BASE_URL` 請填 service root，不要加 `/v1`，Claude Code 會自行呼叫 `/v1/messages`。

如果你想保留 Claude model name，再由 proxy 映射：

```dotenv
MODEL_MAPPING_JSON={"proxy-default":"gpt-5.5","claude-sonnet-4-5":"gpt-5.5"}
```

### 避免走到 Anthropic API

建議啟動 Claude Code 前先 `unset ANTHROPIC_API_KEY`，並固定設定 `ANTHROPIC_BASE_URL` 指向這個 proxy。若你用 `claude --bare` 做測試，Claude Code 會要求 API-key style auth；此時可以把 `ANTHROPIC_API_KEY` 設成 `PROXY_AUTH_TOKEN` 的值，因為請求仍會送到 `ANTHROPIC_BASE_URL` 指定的本機或 Kubernetes proxy。

最小實測範例：

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_API_KEY=local-shared-token \
ANTHROPIC_AUTH_TOKEN=local-shared-token \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
DISABLE_TELEMETRY=1 \
claude --bare --no-session-persistence --tools "" -p \
  --model claude-sonnet-4-6 \
  "Reply with exactly: claude-proxy-ok"
```

Proxy log 會印出實際送到上游的 model，例如：

```text
proxying message request upstream_model=gpt-5.5 stream=True
```

Claude Code 的輸出仍可能把 `modelUsage` 標成 `claude-sonnet-*`，那是 Claude Code 依傳入 model name 做的本地估算欄位；是否有轉到 OpenAI 要看 proxy log 的 `upstream_model`。

## Kubernetes

先複製並修改 secret：

```bash
cp k8s/secret.example.yaml /tmp/claude-code-proxy-secret.yaml
# 編輯 /tmp/claude-code-proxy-secret.yaml
kubectl apply -f /tmp/claude-code-proxy-secret.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

部署到 cluster 前，把 `k8s/deployment.yaml` 裡的 image 換成 GHCR image：

```yaml
image: ghcr.io/<owner>/<repo>:latest
```

本機測試 cluster service：

```bash
kubectl port-forward svc/claude-code-proxy 8080:80
```

然後把 Claude Code 指到 `http://localhost:8080`。

## 限制

- `thinking`、prompt caching、Anthropic server tools 等 Anthropic-only 功能會被忽略或降級。
- `/v1/messages/count_tokens` 是近似值，足夠讓 client 做基本流程判斷，但不是 OpenAI tokenizer 的精確結果。
- 圖片 block 會轉成 OpenAI `image_url` content part，實際可用性取決於你設定的上游 model。
- 若使用非 OpenAI 官方 endpoint，可能需要把 `OPENAI_MAX_TOKENS_FIELD=max_tokens`。

## 參考文件

- Claude Code environment variables: https://code.claude.com/docs/en/env-vars
- Claude streaming Messages API: https://platform.claude.com/docs/en/build-with-claude/streaming
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create
