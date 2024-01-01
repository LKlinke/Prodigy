#!/bin/bash

for file in pgfexamples/Table\ 4/*
do
  if [[ "$file" == *invariant* ]]
  then
    echo -n
  else
    echo "Run example ${file}"
    printf "\e[32mResult:\t\e[m"
    python prodigy/cli.py $@ check_equality $file ${file%".pgcl"}"_invariant.pgcl"
  fi
done
