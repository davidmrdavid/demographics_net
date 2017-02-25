all: enable_gui build run

enable_gui:
	xhost +local:root
build:
	sudo docker build -t demographics_net_cpu .
run:
	sudo docker run -it demographics_net_cpu
