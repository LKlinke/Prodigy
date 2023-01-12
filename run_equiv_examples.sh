#!/bin/bash

for file in pgfexamples/equivalence/*
do
  if [[ $file == *_invariant.pgcl ]]
  then
    echo ""
  else
    echo "Run example ${file}"
    if [[ $file == */skip_*.pgcl ]]
    then
      printf "\e[32mResult:\t\e[m \e[36mSkipped!\e[m\n"
    else
      printf "\e[32mResult:\t\e[m"
      python prodigy/cli.py $@ check_equality $file ${file%".pgcl"}"_invariant.pgcl"
    fi
  fi
done
