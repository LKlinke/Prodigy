#!/bin/bash

for file in pgfexamples/inference/*
do
  echo "Run example ${file}"
  if [[ $file == pgfexamples/inference/skip_*.pgcl ]]
  then
    echo "Skipped!"
  else
    python prodigy/cli.py $@ main $file
  fi
done
