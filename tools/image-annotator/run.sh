#! /usr/bin/env sh
# you need to install nix-shell https://nixos.org/nix/manual/
cd $(dirname $0)
mkdir -p ../output

for i in $(seq 2 20);
do
    ./main.py ../resource/message${i}.png ../output/message${i}.svg
    ./main.py ../resource/message${i}.png ../output/message${i}.txt
done