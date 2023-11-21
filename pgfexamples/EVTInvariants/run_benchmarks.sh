#!/bin/zsh
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
for f in $SCRIPTPATH/*.pgcl
do
  echo "Processing $f"
  echo -n "6\n5\n" | poetry run prodigy "$@" main "$f"
done
