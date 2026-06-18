FROM rust:1.78-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml .
COPY src/ src/
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/adios-regulatory-ai /app/adios-regulatory-ai
COPY data/ data/
EXPOSE 8000
CMD ["/app/adios-regulatory-ai"]
