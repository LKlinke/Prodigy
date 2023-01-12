#!/bin/bash

for file in pgfexamples/equivalence/*
do
  if [[ $file == *_invariant.pgcl ]]
  then
    echo ""
  elif [[ $file == skip_*.pgcl ]]
  then
    echo "Skipped"
  else
    echo "Run example ${file}"
    printf "\e[32mResult:\t\e[m"
    python prodigy/cli.py $@ check_equality $file ${file%".pgcl"}"_invariant.pgcl"
  fi
done
