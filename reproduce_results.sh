#!/bin/bash
echo
echo
echo -e '\033[92m----------------------------- REPLICATING RESULTS FROM THE OOPSLA 2024 PAPER -----------------------------\033[0m'
echo
echo
echo -e '\033[96m--------------------------- Inference of Loopy Programs (Table 4 using the GiNaC engine) --------------------------\033[0m'
echo
let count=1
for file in pgfexamples/Table\ 4/*
do
  if [[ $file == *invariant* ]]
  then
     echo -n
  else
    filename=$(basename -- "$file")
    echo -e "\033[93mEXAMPLE #$count (${filename%.*}):\n\033[0m"
    for i in {1..20}
    do
      output=$( echo -e "1\npgfexamples/Table 4/${filename%.*}_invariant.pgcl" | python prodigy/cli.py --engine ginac main "$file" )
      elapsed=$( echo -e "$output" | tail -n 1 |  awk '{ print $3 }')
      echo "Run $i: $elapsed"
    done
    echo
    echo
    count=$((count + 1))
  fi
done
echo
echo
echo -e '\033[96m--------------------------- Inference of Loopy Programs (Table 4 using the Sympy engine) --------------------------\033[0m'
echo
let count=1
for file in pgfexamples/Table\ 4/*
do
  if [[ $file == *invariant* ]]
  then
     echo -n
  else
    if [[ $file == *brp* ]]
    then
      echo "Timeout files will be benchmarked at the end."
    else
      filename=$(basename -- "$file")
      echo -e "\033[93mEXAMPLE #$count (${filename%.*}):\n\033[0m"
      for i in {1..20}
      do
        output=$( echo -e "1\npgfexamples/Table 4/${filename%.*}_invariant.pgcl" | python prodigy/cli.py --engine sympy main "$file" )
        elapsed=$( echo -e "$output" | tail -n 1 |  awk '{ print $3 }')
        echo "Run $i: $elapsed"
      done
      echo
      echo
      count=$((count + 1))
      fi
  fi
done
echo
echo
echo -e '\033[96m--------------------------- Inference of Loop-free Programs (Appendix Table 5 using the GiNaC engine) --------------------------\033[0m'
echo
let count=1
for file in pgfexamples/Appendix/*
do
  if [[ $file == *invariant* ]]
  then
     echo -n
  else
    filename=$(basename -- "$file")
    echo -e "\033[93mEXAMPLE #$count (${filename%.*}):\n\033[0m"
    for i in {1..20}
    do
      output=$( python prodigy/cli.py --engine ginac main "$file" )
      elapsed=$( echo -e "$output" | tail -n 1 |  awk '{ print $3 }')
      echo "Run $i: $elapsed"
    done
    echo
    echo
    count=$((count + 1))
  fi
done
echo
echo
echo -e '\033[96m--------------------------- Inference of Loop-free Programs (Appendix Table 5 using the SymPy engine) --------------------------\033[0m'
echo
let count=1
for file in pgfexamples/Appendix/*
do
  if [[ $file == *invariant* ]]
  then
     echo -n
  else
    filename=$(basename -- "$file")
    echo -e "\033[93mEXAMPLE #$count (${filename%.*}):\n\033[0m"
    for i in {1..20}
    do
      output=$( python prodigy/cli.py --engine ginac main "$file" )
      elapsed=$( echo -e "$output" | tail -n 1 |  awk '{ print $3 }')
      echo "Run $i: $elapsed"
    done
    echo
    echo
    count=$((count + 1))
  fi
done
echo
echo
echo -e '\033[96m--------------------------- Inference of Loopy Timeout Programs (brp_obs_param, engine:SymPy) --------------------------\033[0m'
echo
echo -e "\033[93mEXAMPLE (brp_obs_parameter):\n\033[0m"
output=$( echo -e "1\npgfexamples/Table 4/brp_obs_parameter_invariant.pgcl" | python prodigy/cli.py --engine ginac main pgfexamples/Table\ 4/brp_obs_parameter.pgcl )
echo -e "$output" | tail -n 1
echo
echo
done