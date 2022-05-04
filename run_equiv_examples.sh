#!/bin/bash

for file in pgfexamples/equivalence/*
do
  if [[ $file == *_invariant.pgcl ]]
  then
    echo ""
  else
    echo "Run example ${file}"
    printf "\e[32mResult:\t\e[m"
    time python prodigy/cli.py $@ check_equality $file ${file%".pgcl"}"_invariant.pgcl"
  fi
done
