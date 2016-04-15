#!/bin/bash
while read setname; do
	wget http://vision.in.tum.de/rgbd/dataset/freiburg1/$setname.tgz &&
	tar -xzvf $setname.tgz &&
	rm "$setname".tgz &&
	if [[ $setname == *"freiburg1"* ]]
	then
  		cp ./K_freiburg1/K.txt "$setname";
		echo "1";
	elif [[ $setname == *"freiburg2"* ]]
	then
  		cp ./K_freiburg2/K.txt ./"$setname";
		echo "2";
	elif [[ $setname == *"freiburg3"* ]]
	then
  		cp ./K_freiburg3/K.txt ./"$setname";
		echo "3";		
	fi
	cd ../src
	./run_all.sh $setname
	cd -
	rm -r "$setname"
done < datasets.txt
