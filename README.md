# Ember

Tool to create embroidery patterns from images.

## Project Structure
```
./src:
ember - application

./src/ember:
embroidery.py - embroidery operations
utils.py - utility functions
```

## Development Tools
- Python 3.12
- [Rye](https://rye.astral.sh/) - `curl -sSf https://rye.astral.sh/get | bash`

## Usage
```
git clone https://github.com/thornewolf/ember
cd ember
mkdir data
wget <image> -O data/image.jpg # or place an image there in some other way
rye run python src/ember/embroidery.py
```

## TODO

- [x] Setup the repo
- [ ] Contour -> Stich functionality is buggy. The stich is not closed when drawn, leading to open shapes. Fix is probably to add an additonal stick connecting the first and last contour element locations.
- [ ] Contours need to be filled when drawn. I think we should do this by identifying "islands" of pixels and filling them in with DFS with a preference order of <move same direction, move right, move left, move down> then trigger DFS going from top left to bottom right. This should cause us to "fill" with stiches going back and forth down each individual shape.
- [ ] Background contour is not defined. This might be related to the previous issue, there is no good point where we can reference the background and fill it in.

## References
- https://zeedigitizing.com/
- https://graphicdesign.stackexchange.com/questions/66375/how-can-i-digitize-my-vector-to-a-dst-for-embroidery-for-free
- https://www.reddit.com/r/Design/comments/150429b/convert_vector_files_to_dst/
- https://www.etsy.com/market/new_york_digitizing?ref=lp_queries_external_top-2
- https://github.com/manthrax/dst-format
- https://github.com/EmbroidePy/pyembroidery/tree/main

## Collaborators
- Thorne Wolfenbarger
- Daniel Bump