#!/bin/bash
function terminate() {
	exit
}
trap 'terminate' {1,2,3,15}

python main.py test
python main.py experiment2
