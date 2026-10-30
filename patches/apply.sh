#!/usr/bin/env bash
# The one door into patches/series: this file reads the apply order and applies it.
# Every other consumer calls this file instead of parsing patches/series itself.
#
#   bash patches/apply.sh --list      print the series in apply order, one basename per line
#   bash patches/apply.sh DIR         apply the series to DIR, the vllm PACKAGE directory
#                                     (site-packages/vllm, not site-packages)
#
# Exit codes: 0 success; 1 a patch did not apply (the message names it); 2 a usage error,
# or patches/series and patches/ disagree.
#
# --list always prints the series. When the series and the directory disagree, it also
# names each offender on stderr and exits 2: a patch that is not in the series is never
# applied, and a name in the series with no file is a typo. The apply mode refuses to start
# on that disagreement.
#
# Apply policy: GNU `patch -p1 --forward --fuzz 0 --no-backup-if-mismatch`, in order, stop
# at the first patch that fails. An offset means the context matched exactly and the
# file only grew around it: that is benign, and the output reports it. Fuzz means the
# context did NOT match and GNU patch accepted an approximate anchor: that is a patch cut
# against a tree that no longer exists, so it fails here by name instead of landing by guess.
# --forward makes a patch that is already applied fail by name, instead of a question on
# the terminal.
#
# Output contract (patches/check_vllm_series.sh counts these lines), one line per patch:
#   == <name>
#   == <name> (<N> hunk(s) at an offset)
# On a failure, the patch output goes to stderr, indented, after a FAILED: line.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  echo "usage: bash patches/apply.sh --list | DIR  (DIR = the installed vllm package directory)" >&2
  exit 2
}

series() {
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series"
}

# agree WHAT NAMES DIR: 0 when NAMES (one per line) and DIR/*.patch are the same set; else
# name each offender on stderr and return 2.
agree() {
  local what=$1 names=$2 dir=$3 on_disk listed x
  on_disk=$(for f in "$dir"/*.patch; do [ -e "$f" ] && basename "$f"; done | sort)
  listed=$(printf '%s\n' "$names" | sort)
  [ "$on_disk" = "$listed" ] && return 0
  echo "ERROR: $what and the $(basename "$dir")/ directory disagree:" >&2
  for x in $(comm -23 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$listed")); do
    echo "    not in $what (never applied): $x" >&2
  done
  for x in $(comm -13 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$listed")); do
    echo "    in $what, no such file (or listed twice): $x" >&2
  done
  return 2
}

list() {
  local names
  names=$(series)
  printf '%s\n' "$names"
  agree patches/series "$names" "$HERE"
}

# apply_one DIR PATCH_FILE NAME HINT: apply one patch by the policy above, or exit 1 by name.
apply_one() {
  local dir=$1 file=$2 name=$3 hint=$4 out n
  if ! out=$(patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$dir" -i "$file" 2>&1); then
    {
      echo "FAILED: $name does not apply to $dir with exact context. Possible causes:"
      printf '%s\n' "$hint"
      printf '%s\n' "$out" | sed 's/^/    /'
    } >&2
    exit 1
  fi
  n=$(printf '%s\n' "$out" | grep -c 'offset' || true)
  if [ "$n" -gt 0 ]; then echo "== $name ($n hunk(s) at an offset)"; else echo "== $name"; fi
}

SERIES_HINT="    the patch is already applied,
    DIR is not the vllm package directory, or the vLLM version is not the pin."

apply_series() {
  local dir=$1 names name
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  names=$(list) || exit 2
  while IFS= read -r name; do
    apply_one "$dir" "$HERE/$name" "$name" "$SERIES_HINT"
  done <<<"$names"
}

case "${1:-}" in
  --list)
    if [ $# -eq 1 ]; then list
    else usage; fi ;;
  ""|-*) usage ;;
  *) [ $# -eq 1 ] || usage; apply_series "$1" ;;
esac
