# Published builder: ghcr.io/eloquage/typephp-builder
# Local alias: docker build -t eloquage-typephp-builder -f packages/typephp-builder/Dockerfile packages/typephp-builder
FROM ghcr.io/eloquage/typephp-builder:latest

WORKDIR /src
COPY . .

CMD ["sh", "-c", "test -f project.yml || cp project.yml.example project.yml; tpc.php project.yml"]
