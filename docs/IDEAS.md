# Enums

```cpp
// Traditional enum
enum Opt : int16 {
    One,
    Two
}; // ; ?
Opt option = Opt.One;

if (option == Opt.two) {
}
```

```cpp
// Anonymous enum declaration
enum : int16 {
    One,
    Two
} option = One;

if (option == Two) {
}
```

```cpp
// Named enum declaration with inline definition
enum Opt : int16 {
    One,
    Two
} option = Opt.One;

if (option == Opt.Two) {
}
```

# Bit packing

struct Packed {
    uint8  a : 4,
    uint8  b : 4,
    uint16 c : 10;
    uint8  d : 6;
    uint8  e;
}

# SVSL API?
svsl.h is empty!