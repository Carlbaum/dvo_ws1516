#!/bin/bash
# This script runs all the datasets listed in datasets.txt
while read setname; do
	#get url component
	if [[ $setname == *"freiburg1"* ]]
	then
  		urlname="freiburg1/"$setname;
	elif [[ $setname == *"freiburg2"* ]]
	then
  		urlname="freiburg2/"$setname;
	elif [[ $setname == *"freiburg3"* ]]
	then
  		urlname="freiburg3/"$setname;		
	fi &&
	# get file, uncommpress and delete compressed version
	wget http://vision.in.tum.de/rgbd/dataset/$urlname.tgz &&
	tar -xzvf $setname.tgz &&
	rm "$setname".tgz &&
	#copy appropriate K.txt into folder
	if [[ $setname == *"freiburg1"* ]]
	then
  		cp ./K_freiburg1/K.txt "$setname";
	elif [[ $setname == *"freiburg2"* ]]
	then
  		cp ./K_freiburg2/K.txt ./"$setname";
	elif [[ $setname == *"freiburg3"* ]]
	then
  		cp ./K_freiburg3/K.txt ./"$setname";
	fi &&

	#go to src directory and run the program
	cd ../src &&
	./run_all.sh $setname &&
	cd - &&
	rm -r "$setname"
done < datasets.txt
