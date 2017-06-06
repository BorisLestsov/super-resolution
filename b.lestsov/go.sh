#!/bin/bash
#FILES=/home/b.lestsov/datasets/test/super-res-test-div4/*
FILES=/home/b.lestsov/BSD4LR/*
trap "exit" INT
for f in $FILES
do
  python testit.py $f
  #echo $f
  # take action on each file. $f store current file name
  #cat $f
done
