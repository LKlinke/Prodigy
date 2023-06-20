#!/bin/zsh
for file in dep_bern dueling_cowboys geometric geometric_parameter n_geometric n_geometric_parameter random_walk random_walk_parameter trivial_iid
do
  echo $file
  for i in {1..20}
  do
    echo -e "1\npgfexamples/equivalence/${file}_invariant.pgcl" | python prodigy/cli.py $@ main pgfexamples/equivalence/$file.pgcl | tail -n 1 |  awk '{ print $3 }' | sed 's/\./,/g'
  done
  echo next
done
