#!/bin/bash
# Generate self-signed certificates for local HTTPS development
#
# Usage: ./generate-certs.sh [domain]
# Default domain: localhost

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS_DIR="${SCRIPT_DIR}/../certs"
DOMAIN="${1:-localhost}"

mkdir -p "$CERTS_DIR"

echo "Generating self-signed certificate for: $DOMAIN"

# Generate private key
openssl genrsa -out "$CERTS_DIR/key.pem" 2048

# Generate certificate signing request config
cat > "$CERTS_DIR/cert.conf" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = US
ST = Development
L = Local
O = ModelStag Development
OU = Development
CN = $DOMAIN

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = *.$DOMAIN
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate self-signed certificate (valid for 365 days)
openssl req -new -x509 -key "$CERTS_DIR/key.pem" \
    -out "$CERTS_DIR/cert.pem" \
    -days 365 \
    -config "$CERTS_DIR/cert.conf"

# Clean up config
rm "$CERTS_DIR/cert.conf"

echo ""
echo "Certificate generated successfully!"
echo "  Certificate: $CERTS_DIR/cert.pem"
echo "  Private key: $CERTS_DIR/key.pem"
echo ""
echo "To trust the certificate on macOS:"
echo "  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $CERTS_DIR/cert.pem"
echo ""
echo "To trust on Linux (Ubuntu/Debian):"
echo "  sudo cp $CERTS_DIR/cert.pem /usr/local/share/ca-certificates/modelstag.crt"
echo "  sudo update-ca-certificates"
echo ""
