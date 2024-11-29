#!/bin/zsh
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
for f in $SCRIPTPATH/*.pgcl
do
  if [[ $f == $SCRIPTPATH/skip_*.pgcl ]]
  then
      printf "Processing $f\n\e[32mResult:\t\e[m \e[36mSkipped!\e[m\n"
  else
    echo "Processing $f"
    poetry run prodigy "$@" invariant_synthesis "$f"
  fi
done
