#!/usr/bin/env bash
# Sign every Mach-O + dylib in a macOS package dir with the Developer ID
# Application identity (hardened runtime), then notarize the .zip with Apple's
# notary service via an App Store Connect API key.
#
# Usage: sign_and_notarize_macos.sh <package-dir> <out-zip>
# Env:   AC_API_KEY_ID, AC_API_KEY_ISSUER, AC_API_KEY (base64 of the .p8)
#        (the identity is resolved at runtime from the keychain the caller
#         imported; see the "Import code-signing certificate" workflow step.)
set -euo pipefail

DIR="${1:?usage: $0 <package-dir> <out-zip>}"
ZIP="${2:?usage: $0 <package-dir> <out-zip>}"
ENT="$(cd "$(dirname "$0")" && pwd)/wmr.entitlements"

[ -d "$DIR" ] || { echo "error: not a directory: $DIR" >&2; exit 1; }
for v in AC_API_KEY_ID AC_API_KEY_ISSUER AC_API_KEY; do
    [ -n "${!v:-}" ] || { echo "error: env $v is empty" >&2; exit 1; }
done
[ -f "$ENT" ] || { echo "error: entitlements not found: $ENT" >&2; exit 1; }

# 1. Resolve the identity at runtime.
IDENTITY="$(security find-identity -p codesigning -v | grep -oE 'Developer ID Application:[^)]+\)' | head -n1 || true)"
[ -n "$IDENTITY" ] || { echo "error: no 'Developer ID Application' identity in keychain" >&2; exit 1; }
echo "Signing as: $IDENTITY"

# 2. Sign innermost-first. Hardened runtime (--options runtime) turns ON library
#    validation for the main executable: every dylib wmr.bin loads (linked OR
#    dlopen'd at runtime, e.g. MoltenVK via the Vulkan loader) must carry this
#    same Team ID. So sign all dylibs BEFORE the executables. --timestamp gives a
#    trusted timestamp (required for notarization).
SIGN_OPTS=(--force --options runtime --sign "$IDENTITY" --timestamp)

# 2a. Dylibs (arm64: libvulkan.1.dylib + libMoltenVK.dylib; x86_64: none).
#     find ... -exec (not xargs -0): BSD xargs has no documented -r, and a bare
#     xargs -0 codesign on empty input is only safe by accident. -exec skips
#     itself when find produces nothing.
if [ -d "$DIR/lib" ]; then
    find "$DIR/lib" -type f -name '*.dylib' -print0 \
        -exec codesign "${SIGN_OPTS[@]}" {} \;
fi

# 2b. Executables: every Mach-O among {wmr, wmr.bin}. Handles BOTH layouts:
#       arm64  -> wmr is bash (skipped), wmr.bin is Mach-O (signed)
#       x86_64 -> wmr is Mach-O (signed), wmr.bin absent
#     grep -q ... || continue is set-e-safe and never aborts on a non-Mach-O.
for exe in "$DIR"/wmr "$DIR"/wmr.bin; do
    [ -e "$exe" ] || continue
    if file -b "$exe" | grep -q 'Mach-O'; then
        codesign "${SIGN_OPTS[@]}" --entitlements "$ENT" "$exe"
    fi
done

# 3. Verify each signed artifact individually.
if [ -f "$DIR/wmr.bin" ]; then
    codesign --verify --strict --verbose=2 "$DIR/wmr.bin"
else
    codesign --verify --strict --verbose=2 "$DIR/wmr"   # x86_64 has only wmr
fi
if [ -d "$DIR/lib" ]; then
    find "$DIR/lib" -type f -name '*.dylib' -exec codesign --verify --strict {} \;
fi

# 4. Zip preserving resource forks / xattrs. ditto -c -k --keepParent is
#    notarytool's native input.
ditto -c -k --keepParent "$DIR" "$ZIP"
echo "Created $ZIP"

# 5. Notarize. Write the .p8 to a private temp file (mode 600).
KEY_FILE="$(mktemp -t wmr_authkey)"
trap 'rm -f "$KEY_FILE"' EXIT
chmod 600 "$KEY_FILE"
# macOS base64 decode flag is -D (BSD), NOT --decode (GNU).
printf '%s' "$AC_API_KEY" | base64 -D > "$KEY_FILE"

echo "Submitting to notary service (blocks until Apple responds)..."
xcrun notarytool submit "$ZIP" \
    --key "$KEY_FILE" \
    --key-id "$AC_API_KEY_ID" \
    --issuer "$AC_API_KEY_ISSUER" \
    --wait
# --wait returns non-zero on rejection; the submission UUID is printed. Fetch log:
#   xcrun notarytool log <UUID> --key <.p8> --key-id <ID> --issuer <UUID>

# 6. Best-effort staple. stapler targets .app/.dmg/.pkg; a loose-file zip may
#    not staple (ticket still retrievable online via cdhash). Non-fatal.
xcrun stapler staple "$ZIP" || echo "staple skipped (loose-file zip; online check still works)"

echo "Done: $ZIP"
