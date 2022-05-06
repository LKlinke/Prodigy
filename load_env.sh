#!/bin/bash

echo "__/\\\\\\\\\\\\\\\\\\\\\\\\\\_______________________________________/\\\\\\___________________________________########";
echo "#_\\/\\\\\\/////////\\\\\\____________________________________\\/\\\\\\___________________________________#######";
echo "##_\\/\\\\\\_______\\/\\\\\\____________________________________\\/\\\\\\___/\\\\\\___/\\\\\\\\\\\\\\\\_____/\\\\\\__/\\\\\\_######";
echo "###_\\/\\\\\\\\\\\\\\\\\\\\\\\\\\/___/\\\\/\\\\\\\\\\\\\\______/\\\\\\\\\\___________\\/\\\\\\__\\///___/\\\\\\////\\\\\\___\\//\\\\\\/\\\\\\__#####";
echo "####_\\/\\\\\\/////////____\\/\\\\\\/////\\\\\\___/\\\\\\///\\\\\\____/\\\\\\\\\\\\\\\\\\___/\\\\\\_\\//\\\\\\\\\\\\\\\\\\____\\//\\\\\\\\\\___####";
echo "#####_\\/\\\\\\_____________\\/\\\\\\___\\///___/\\\\\\__\\//\\\\\\__/\\\\\\////\\\\\\__\\/\\\\\\__\\///////\\\\\\_____\\//\\\\\\____###";
echo "######_\\/\\\\\\_____________\\/\\\\\\_________\\//\\\\\\__/\\\\\\__\\/\\\\\\__\\/\\\\\\__\\/\\\\\\__/\\\\_____\\\\\\__/\\\\_/\\\\\\_____##";
echo "#######_\\/\\\\\\_____________\\/\\\\\\__________\\///\\\\\\\\\\/___\\//\\\\\\\\\\\\\\/\\\\_\\/\\\\\\_\\//\\\\\\\\\\\\\\\\__\\//\\\\\\\\/______#";
echo "########_\\///______________\\///_____________\\/////______\\///////\\//__\\///___\\////////____\\////________";
echo "#                                                                                                    #";
echo "#                                                                                                    #";
echo -e "#                                  \033[92mVIRTUAL PYTHON ENVIRONMENT LOADED\033[0m                                 #";
echo "------------------------------------------------------------------------------------------------------";
sleep 1
echo "#                                                                                                    #";
echo -e "#     \033[96m->\033[0m You can run the experiments presented in the paper by executing \033[33m./reproduce_results.sh\033[0m      #";
echo "#        The measured timings do not include the time for parsing the inputs.                        #";
echo "#                                                                                                    #";
echo -e "#     \033[96m->\033[0m Prodigy has two modes: \033[95mcheck_equality\033[0m and \033[95mmain\033[0m. You can call them via                       #"
echo -e "#        \033[33mpython prodigy/cli.py <target>\033[0m where \033[33m<target>\033[0m is either:                                    #";
echo -e "#     \t     \033[97m*\033[0m \033[33mcheck_equality\033[0m [OPTIONS] PROGRAM_FILE INVARIANT_FILE                                  #";
echo -e "#     \t     \033[97m*\033[0m \033[33mmain\033[0m           [OPTIONS] PROGRAM_FILE [INPUT_DIST]                                    #";
echo "#                                                                                                    #";
echo -e "#        \033[95mcheck_equality\033[0m: just performs the equivalence check described in the paper.                 #";
echo -e "#        \033[95mmain\033[0m: computes the final distribution after termination of a given pGCL program.            #";
echo -e "#           When no input distribution is specified it assumes all variables to be                   #";
echo -e "#           initialized with 0, i.e., implicitly assumes the input distribution '1'.                 #";
echo "#                                                                                                    #";
echo "#                                                                                                    #";
echo -e "#     \033[96m->\033[0m Verbose output can be switched on by \033[33mpython prodigy/cli.py --intermediate-results <target>\033[0m  #"
echo "#                                                                                                    #";
echo "------------------------------------------------------------------------------------------------------";
echo "#                                                                                                    #";
echo -e "#        For more details please visit our Github repository: \033[97m\033[4mhttps://github.com/LKlinke/prodigy\033[0m     #";
echo "#               (C) Mingshuai Chen, Joost-Pieter Katoen, Lutz Klinkenberg, Tobias Winkler            #";
echo "------------------------------------------------------------------------------------------------------";
echo
echo -e "\033[96mWe spawned you in /root/artifact.\033[0m"
echo

cd /root/artifact

exec bash -l -c "poetry run bash"