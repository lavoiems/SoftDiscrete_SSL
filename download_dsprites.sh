#!/usr/bin/env sh

DEFAULTPATH=./data
DEFAULTSPLIT=composition
DLPATH="https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

HERE=$PWD
DATAPATH="${1:-$DEFAULTPATH}"
K="${2:-1}"

mkdir -p "${DATAPATH}"
cd "${DATAPATH}"
wget $DLPATH -O dsprites.npz
cd "${HERE}"
python split_dsprites.py $DATAPATH dsprites.npz $K

rm -f $DATAPATH/dsprites.npz
