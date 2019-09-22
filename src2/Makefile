all: shapes perspective borders

shapes:	Shapes.cpp
	g++ -o shapes Shapes.cpp `pkg-config --cflags --libs opencv`

perspective: perspectiveProjection.cpp
	g++ -o perspective perspectiveProjection.cpp `pkg-config --cflags --libs opencv`	

borders: BorderDetection.cpp
	g++ -o borders BorderDetection.cpp `pkg-config --cflags --libs opencv`

clean:
	rm shapes
	rm perspective
	rm borders
	