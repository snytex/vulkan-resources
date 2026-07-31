# vk-space

A C++20 executable, scaffolded by 🦥 **Loris**.

## Build

Requires Vulkan, GLFW, GLM, `glslc`, and [`sn`](/usr/local/include/sn).

```sh
cmake -B build
cmake --build build
cd build && ./vk_space
```

Shaders in `src/shaders/` are compiled to SPIR-V into `build/shaders/`, which the
executable loads relative to the working directory — run it from `build/`.

## Layout

```
vk_space/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── compile_flags.txt
├── src/
│   ├── main.cpp
│   ├── shaders/
│   └── vendor/imgui/
```

---

_Made with a little help from Pip the slow loris._ 🌿
