#!/bin/bash

datasets=("SmoothSubspace" "Strawberry" "Crop" "FiftyWords")
models=("SamplerGAN" "RCGAN")
architectures=("Linear" "RNN" "TCN")

for dataset in "${datasets[@]}"
do
    if [ "$dataset" == "FiftyWords" ]
    then
        for model in "${models[@]}"
        do
            for architecture in "${architecture[@]}"
            do
                if [ "$model" == "RCGAN" ] && [ "$architecture" == "Linear" ]
                then
                    continue
                else
                    python main.py --domain "time-series" --dataset "$dataset" --dataroot "./datasets" --gan_model "$model" --architecture "$architecture" --batch_size 10 --num_epochs 2000
                fi
            done
        done
    else
        for model in "${models[@]}"
        do
            for architecture in "${architecture[@]}"
            do
                if [ "$model" == "RCGAN" ] && [ "$architecture" == "Linear" ]
                then
                    continue
                else
                    python main.py --domain "time-series" --dataset "$dataset" --dataroot "./datasets" --gan_model "$model" --architecture "$architecture" --batch_size 10 --num_epochs 500
                fi
            done
        done
    fi
done
