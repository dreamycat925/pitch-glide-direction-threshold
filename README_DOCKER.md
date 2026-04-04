# Pitch Glide Direction Threshold — Docker (Local)

この Streamlit アプリを Docker / Docker Compose でローカル実行するためのメモです。

---

## Quick start

リポジトリ直下で実行します。

```bash
docker compose up -d --build
```

ブラウザで開きます。

- http://localhost:20000

停止:

```bash
docker compose down
```

---

## Dev mode

ローカルの編集内容をコンテナ側へ反映しやすくしたい場合:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

ブラウザで開きます。

- http://localhost:20000

---

## ポート

- コンテナ内部: `8501`
- ホスト側: `127.0.0.1:20000`

`docker-compose.yml` / `docker-compose.dev.yml` は、LAN に公開しないよう `127.0.0.1` バインドになっています。

---

## ポート競合時

`20000` がすでに使われている場合は、compose ファイルの左側のポート番号を変更してください。

例:

```yaml
127.0.0.1:20000:8501
```

を

```yaml
127.0.0.1:21000:8501
```

のように変更します。
