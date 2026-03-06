#!/bin/bash
set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV_FILE="$SCRIPT_DIR/.env"
TRANSFORMERLAB_APP_DIR="$HOME/transformerlab-app"
TLAB_ENV_DIR="$HOME/.transformerlab"
TLAB_ENV_FILE="$TLAB_ENV_DIR/.env"

# Install nvm and Node.js 22 if needed
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
  echo "Installing nvm"
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  # Refresh shell configuration so nvm is available
  if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
  fi
  if [ -f "$HOME/.zshrc" ]; then
    . "$HOME/.zshrc"
  fi
fi

# Ensure NVM is available for non-interactive shells
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Ensure Node.js 22 is installed and active
if ! command -v nvm >/dev/null 2>&1; then
  echo "nvm not found after installation; aborting." >&2
  exit 1
fi
if ! nvm ls 22 >/dev/null 2>&1; then
  nvm install 22
fi
nvm alias default 22
nvm use 22

# Ensure transformerlab-app is present in home directory
if [ ! -d "$TRANSFORMERLAB_APP_DIR" ]; then
  echo "Cloning transformerlab-app into $TRANSFORMERLAB_APP_DIR"
  git clone https://github.com/transformerlab/transformerlab-app.git "$TRANSFORMERLAB_APP_DIR"
fi

# First update API and reinstall dependencies
echo "Updating Transformer Lab API"
cd "$TRANSFORMERLAB_APP_DIR"
git checkout main
git pull
cd api
./install.sh
cd ..

# Sync environment configuration
if [ -f "$LOCAL_ENV_FILE" ]; then
  mkdir -p "$TLAB_ENV_DIR"
  touch "$TLAB_ENV_FILE"

  # Skip copy only if ~/.transformerlab/.env has any variable other than the two allowed JWT vars
  skip_copy=false
  if [ -s "$TLAB_ENV_FILE" ]; then
    while IFS= read -r line; do
      key="${line%%=*}"
      key="$(printf '%s' "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      [ -z "$key" ] && continue
      case "$key" in
        TRANSFORMERLAB_JWT_SECRET|TRANSFORMERLAB_REFRESH_SECRET) ;;
        *) skip_copy=true; break ;;
      esac
    done < <(grep -vE '^\s*($|#)' "$TLAB_ENV_FILE")
  fi

  if [ "$skip_copy" = false ]; then
    echo "Merging local .env into $TLAB_ENV_FILE"
    while IFS= read -r line; do
      # skip empty lines
      if [ -z "$line" ]; then
        continue
      fi
      # keep comments and malformed lines as-is
      if printf '%s\n' "$line" | grep -qE '^\s*#'; then
        echo "$line" >>"$TLAB_ENV_FILE"
        continue
      fi
      if ! printf '%s\n' "$line" | grep -q '='; then
        echo "$line" >>"$TLAB_ENV_FILE"
        continue
      fi

      key="${line%%=*}"
      case "$key" in
        TRANSFORMERLAB_JWT_SECRET|TRANSFORMERLAB_REFRESH_SECRET)
          # Prefer local JWT secrets: remove existing and append local
          tmp_file="$(mktemp)"
          grep -v "^${key}=" "$TLAB_ENV_FILE" >"$tmp_file" || true
          mv "$tmp_file" "$TLAB_ENV_FILE"
          echo "$line" >>"$TLAB_ENV_FILE"
          ;;
        *)
          # Append other variables only if not already present
          if ! grep -q "^${key}=" "$TLAB_ENV_FILE"; then
            echo "$line" >>"$TLAB_ENV_FILE"
          fi
          ;;
      esac
    done <"$LOCAL_ENV_FILE"
  else
    echo "$TLAB_ENV_FILE contains variables other than TRANSFORMERLAB_JWT_SECRET and TRANSFORMERLAB_REFRESH_SECRET; skipping merge from local .env"
  fi

  # Ensure transformerlab-app has the same .env file
  if [ -f "$TLAB_ENV_FILE" ]; then
    cp "$TLAB_ENV_FILE" "$TRANSFORMERLAB_APP_DIR/.env"
  fi
fi

# Then update and build the frontend and overwrite the existing static webapp
cd "$TRANSFORMERLAB_APP_DIR"
npm install
npm run build
WEBAPP_DIR="$HOME/.transformerlab/webapp"
rm -rf "$WEBAPP_DIR"
mkdir -p "$WEBAPP_DIR"
cp -r "$TRANSFORMERLAB_APP_DIR/release/cloud/." "$WEBAPP_DIR/"