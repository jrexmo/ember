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
git clone <url>
cd <repo>
mkdir data
wget <image> -O data/image.jpg # or place an image there in some other way
rye run python src/ember/embroidery.py
```

## Collaborators
- Thorne Wolfenbarger
- Daniel Bump