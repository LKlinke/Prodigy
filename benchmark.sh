#!/bin/zsh
for file in \
            brp_obs_parameter \
            bit_flip_conditioning bit_flip_conditioning_parameter bit_flip_conditioning_generalized_parameter \
            dep_bern \
            dueling_cowboys \
            endless_conditioning \
            geometric geometric_observe geometric_observe_parameter geometric_parameter geometric_shifted \
            ky_die ky_die_2 ky_die_parameter \
            n_geometric n_geometric_parameter \
            negative_binomial_parameter negative_binomial_reversed_parameter \
            random_walk random_walk_parameter \
            trivial_iid
do
  echo $file
  for i in {1..20}
  do
    echo -e "1\npgfexamples/equivalence/${file}_invariant.pgcl" | python prodigy/cli.py $@ main pgfexamples/equivalence/$file.pgcl | tail -n 1 |  awk '{ print $3 }' | sed 's/\./,/g'
  done
  echo next
done
