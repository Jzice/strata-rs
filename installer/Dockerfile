# 构建 strata 服务器与 Web 前端，产物为一个静态镜像。
FROM rust:1.85-slim AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release -p strata

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/strata /usr/local/bin/strata
COPY --from=web /web/dist /usr/local/share/strata/web
EXPOSE 8080
ENTRYPOINT ["strata", "serve", "--web-dir", "/usr/local/share/strata/web", "--port", "8080"]
