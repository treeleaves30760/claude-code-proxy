# Claude Code Multi-Provider Proxy

[English](README.md) | 繁體中文

這是一個多 provider AI API proxy。它同時提供 Anthropic Messages API 與 OpenAI-compatible Chat Completions API，讓 Claude Code、opencode 與其他 OpenAI-compatible client 都能透過同一個服務轉接到 OpenAI、OpenRouter、Ollama 或其他相容廠商。

目前支援：

- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `POST /v1/chat/completions`
- `GET /v1/models`
- 非串流與 SSE 串流回應
- Claude Code 常用的 tool use / tool result 轉換
- OpenAI-compatible pass-through，給 opencode、OpenAI SDK、LiteLLM 類工具使用
- YAML model routing，可把公開 model name 對應到不同 provider / upstream model
- Docker 與 Kubernetes 部署

## 環境變數

| 變數 | 必填 | 說明 |
| --- | --- | --- |
| `MODEL_CONFIG_PATH` | 否 | YAML routing config 路徑；預設會找 `config.yaml`，不存在時 fallback 到 legacy env 模式 |
| `PROXY_AUTH_TOKEN` | 否 | 若設定，client 必須用 Bearer token、`x-api-key` 或 `anthropic-api-key` 傳入相同 token |
| `OPENAI_API_KEY` | 視設定 | OpenAI API key；YAML 的 `openai` provider 預設讀這個 |
| `OPENROUTER_API_KEY` | 視設定 | OpenRouter API key；YAML 的 `openrouter` provider 預設讀這個 |
| `OLLAMA_BASE_URL` | 否 | Ollama OpenAI-compatible base URL，例如 `http://localhost:11434/v1` |
| `OPENAI_API_ENDPOINT` | 否 | Legacy env 模式的 OpenAI-compatible base URL，例如 `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 否 | Legacy env 模式固定轉送到的 model；預設 `gpt-5.5` |
| `MODEL_MAPPING_JSON` | 否 | JSON object，用來把 Claude model name 映射到 OpenAI model |
| `PUBLIC_MODELS` | 否 | `/v1/models` 回傳給 client 的 model 清單 |
| `REQUEST_TIMEOUT_SECONDS` | 否 | 上游 API timeout，預設 `600` |

範例：

```bash
cp .env.example .env
cp config.example.yaml config.yaml
```

`.env`：

```dotenv
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
OLLAMA_BASE_URL=http://localhost:11434/v1
PROXY_AUTH_TOKEN=local-shared-token
MODEL_CONFIG_PATH=config.yaml
```

如果沒有提供 `MODEL_CONFIG_PATH` / `config.yaml`，服務會用 legacy env 模式：所有公開 model 都會轉到 `OPENAI_MODEL`。

## YAML 模型路由

`config.example.yaml` 內建 OpenAI、OpenRouter、Ollama 三種 provider 範例：

```yaml
default_model: proxy-default

providers:
  openai:
    type: openai
    api_key_env: OPENAI_API_KEY

  openrouter:
    type: openrouter
    api_key_env: OPENROUTER_API_KEY

  ollama:
    type: ollama
    base_url: ${OLLAMA_BASE_URL:-http://localhost:11434/v1}
    require_api_key: false

models:
  proxy-default:
    provider: openai
    model: gpt-5.5
    aliases: [claude-sonnet-4-6, claude-opus-4-7]

  openrouter-gpt-5.5:
    provider: openrouter
    model: openai/gpt-5.5

  ollama-gpt-oss-20b:
    provider: ollama
    model: gpt-oss:20b
```

Client 對 proxy 送 `model: openrouter-gpt-5.5` 時，proxy 會轉到 OpenRouter 的 `openai/gpt-5.5`。`provider:model` 也支援臨時 passthrough，例如 `openrouter:anthropic/claude-sonnet-4.6`。

## 本機執行

```bash
uv sync
uv run uvicorn claude_code_proxy.server:app --host 0.0.0.0 --port 8080
```

健康檢查：

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

## Docker

```bash
docker build -t claude-code-proxy:latest .
docker run --rm --env-file .env -p 8080:8080 claude-code-proxy:latest
```

使用 YAML routing：

```bash
docker run --rm --env-file .env \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -p 8080:8080 \
  claude-code-proxy:latest
```

或：

```bash
docker compose up --build
```

`docker-compose.yml` 預設會把 `config.example.yaml` 掛成容器內的 `/app/config.yaml`。要使用自己的設定檔時，在 `.env` 加：

```dotenv
LOCAL_MODEL_CONFIG_PATH=./config.yaml
MODEL_CONFIG_PATH=/app/config.yaml
```

## GHCR Image

此 repo 內建 GitHub Actions workflow：`.github/workflows/docker-image.yml`。推送到 `main` 或推送 `v*.*.*` tag 時會自動 build multi-arch image，並推到 GitHub Container Registry。

Image 名稱：

```text
ghcr.io/<owner>/<repo>
```

下載與執行：

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run --rm --env-file .env -p 8080:8080 ghcr.io/<owner>/<repo>:latest
```

如果 GHCR package 是 private，先登入：

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
docker pull ghcr.io/<owner>/<repo>:latest
```

`GHCR_TOKEN` 需要有 `read:packages` 權限。

## Claude Code 設定

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

如果想保留 Claude model name，再由 proxy 映射，請在 `config.yaml` 的 model route 加上 alias：

```yaml
models:
  proxy-default:
    provider: openai
    model: gpt-5.5
    aliases:
      - claude-sonnet-4-6
      - claude-opus-4-7
```

## opencode 設定

opencode 可用 `@ai-sdk/openai-compatible` 對接 proxy。`baseURL` 要指到 `/v1`：

`~/.config/opencode/opencode.jsonc`：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "claude-code-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Claude Code Proxy",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "{env:PROXY_AUTH_TOKEN}"
      },
      "models": {
        "proxy-default": {
          "name": "Proxy default"
        },
        "openrouter-gpt-5.5": {
          "name": "OpenRouter GPT-5.5"
        },
        "openrouter-claude-sonnet-4.6": {
          "name": "OpenRouter Claude Sonnet 4.6"
        },
        "ollama-gpt-oss-20b": {
          "name": "Ollama gpt-oss:20b"
        }
      }
    }
  }
}
```

然後：

```bash
export PROXY_AUTH_TOKEN=local-shared-token
opencode
```

## 預設模型建議

以下名稱是 2026-05-26 重新查官方文件與 OpenRouter model API 後更新的範例：

| Provider | Route | Upstream model |
| --- | --- | --- |
| OpenAI | `proxy-default` / `openai-gpt-5.5` | `gpt-5.5` |
| OpenAI | `openai-gpt-5.5-pro` | `gpt-5.5-pro` |
| OpenRouter | `openrouter-gpt-5.5` | `openai/gpt-5.5` |
| OpenRouter | `openrouter-claude-opus-4.7` | `anthropic/claude-opus-4.7` |
| OpenRouter | `openrouter-claude-sonnet-4.6` | `anthropic/claude-sonnet-4.6` |
| OpenRouter | `openrouter-gemini-3.5-flash` | `google/gemini-3.5-flash` |
| OpenRouter | `openrouter-qwen3-coder-plus` | `qwen/qwen3-coder-plus` |
| Ollama | `ollama-gpt-oss-20b` | `gpt-oss:20b` |

Ollama 使用前需要先 pull model：

```bash
ollama pull gpt-oss:20b
```

## 避免走到 Anthropic API

建議啟動 Claude Code 前先 `unset ANTHROPIC_API_KEY`，並固定設定 `ANTHROPIC_BASE_URL` 指向這個 proxy。若你用 `claude --bare` 做測試，可以把 `ANTHROPIC_API_KEY` 設成 `PROXY_AUTH_TOKEN` 的值，因為請求仍會送到 `ANTHROPIC_BASE_URL` 指定的 proxy。

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
proxying anthropic message provider=openai upstream_model=gpt-5.5 public_model=proxy-default stream=True
```

Claude Code 的 `modelUsage` 仍可能標成 `claude-sonnet-*`，那是 Claude Code 依傳入 model name 做的本地估算欄位；是否有轉到 OpenAI 要看 proxy log 的 `upstream_model`。

## Kubernetes

```bash
cp k8s/secret.example.yaml /tmp/claude-code-proxy-secret.yaml
# 編輯 /tmp/claude-code-proxy-secret.yaml
kubectl apply -f /tmp/claude-code-proxy-secret.yaml
kubectl apply -f k8s/configmap.example.yaml
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

## 限制

- `thinking`、prompt caching、Anthropic server tools 等 Anthropic-only 功能會被忽略或降級。
- `/v1/messages/count_tokens` 是近似值，不是 OpenAI tokenizer 的精確結果。
- 圖片 block 會轉成 OpenAI `image_url` content part，實際可用性取決於上游 model。
- Ollama 的 OpenAI compatibility 是相容層，具體 tool / vision / JSON mode 支援仍取決於本地 model。
- OpenRouter、Ollama 與其他 OpenAI-compatible provider 可能有不同參數支援；可用 YAML route 的 `extra_body` 與 provider 的 `headers` 做細節調整。

## 參考文件

- Claude Code environment variables: https://code.claude.com/docs/en/env-vars
- Claude streaming Messages API: https://platform.claude.com/docs/en/build-with-claude/streaming
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create
- OpenAI models: https://developers.openai.com/api/docs/models/model-endpoint
- opencode providers: https://opencode.ai/docs/providers/
- OpenRouter chat completions: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
- OpenRouter models API: https://openrouter.ai/docs/api/api-reference/models/get-models
- Ollama OpenAI compatibility: https://docs.ollama.com/openai
