#!/bin/bash

datasets=("MNIST" "CIFAR10")
models=("SamplerGAN" "CGAN" "InfoGAN" "ACGAN")

for dataset in "${datasets[@]}"
do
    if [ "datasets" == "MNIST" ]
    then
        for model in "${models[@]}"
        do
            if [ "$model" == "InfoGAN" ]
            then
                python main.py --domain "image" --dataset "$dataset" --gan_model "$model" --batch_size 128 --noise_dim 62 --num_epochs 25
            else
                python main.py --domain "image" --dataset "$dataset" --gan_model "$model" --batch_size 128 --noise_dim 100 --num_epochs 25
            fi
        done
    elif [ "datasets" == "CIFAR10" ]
    then
        for model in "${models[@]}"
        do
            if [ "$model" == "InfoGAN" ]
            then
                python main.py --domain "image" --dataset "$dataset" --gan_model "$model" --batch_size 100 --noise_dim 62 --num_epochs 500
            else
                python main.py --domain "image" --dataset "$dataset" --gan_model "$model" --batch_size 100 --noise_dim 100 --num_epochs 500
            fi
        done
    fi
done
