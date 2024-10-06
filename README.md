## Raytracer Engine

The program is able to render static images depicting simple scenes with the Ray traced lighting. The scenes are based on previously set configuration, may contain geometric objects. Various materials can be aplied to objects.

# Render image example

![Raytracer_demo](https://github.com/user-attachments/assets/a090db33-173b-40b2-948f-c31012365876)

# Compliation requirements

1.  NVIDIA CUDA toolkit
2.  JSON for modern C++ Niels Lohmann's library

# Scene configuration

The rendering scene can be set using the data.json file. Configuration contains options for the camera location and individual objects' properties such as: location, size, material type, material characteristics.

File content example is provided with the project.
