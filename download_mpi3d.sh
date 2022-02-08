#!/usr/bin/env sh

DEFAULTPATH=./data
DEFAULTSPLIT=real

HERE=$PWD
DATAPATH="${1:-$DEFAULTPATH}"
SPLIT="${2:-$DEFAULTSPLIT}"

mkdir -p "${DATAPATH}"
cd "${DATAPATH}"
wget https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_${SPLIT}.npz -O mpi3d.npz
cd "${HERE}"

python split_mpi3d.py $DATAPATH mpi3d.npz $SPLIT

rm -f $DATAPATH/mpi3d.npz

