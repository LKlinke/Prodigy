#!/bin/bash

for file in pgfexamples/Appendix/*
do
  echo "Run example ${file}"
  python prodigy/cli.py $@ main $file
done
