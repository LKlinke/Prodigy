#!/bin/bash

echo
echo "You are now in /root/artifact."
echo

cd /root/artifact

exec bash -l -c "poetry run bash"