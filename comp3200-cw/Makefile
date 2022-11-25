run-single: convert-all
	python single_node.py 9000
run-swarm: convert-all
	python start_swarm.py
convert-all:
	jupyter nbconvert --to python single_node.ipynb
	jupyter nbconvert --to python start_swarm.ipynb