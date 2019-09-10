all: detcircle datagen

detcircle: circleDetector.cpp
	g++ -o detcircle circleDetector.cpp `pkg-config --cflags --libs opencv`

datagen: VideoImages.cpp
	g++ -o datagen VideoImages.cpp `pkg-config --cflags --libs opencv`

clean:
	rm detcircle
	rm datagen
