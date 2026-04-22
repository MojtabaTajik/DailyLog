# syntax=docker/dockerfile:1

FROM golang:1.26-alpine AS builder
WORKDIR /src

# Cache modules separately from source for faster incremental builds.
COPY go.mod go.sum* ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/dailylog .

FROM gcr.io/distroless/static-debian12:latest
WORKDIR /app
COPY --from=builder /out/dailylog /app/dailylog
ENTRYPOINT ["/app/dailylog"]
