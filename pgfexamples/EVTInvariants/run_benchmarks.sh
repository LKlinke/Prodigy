#!/bin/zsh
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
for f in $SCRIPTPATH/*.pgcl
do
  if [[ $f == $SCRIPTPATH/skip_*.pgcl ]]
  then
    echo "Skipped"
  else
    echo "Processing $f"
    poetry run prodigy "$@" invariant_synthesis "$f"
  fi
done
